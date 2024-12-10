"""
 Copyright (c) 2024, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import yaml
import random
import pickle
import json
from transformers import AutoTokenizer
import re
import astunparse
import ray
import os
import gc
import torch
import ast
import contextlib
import datasets
import uuid
import numpy as np
import omegaconf
from omegaconf import OmegaConf
from postproc_utils import general_postprocessing
from accuracy_fns import general_accuracy, GQA_accuracy
import time 
from datetime import datetime
import math
import requests
from PIL import Image
from io import BytesIO   
import socket 
from collections import defaultdict
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")




def code_execution(code_execution_data, this_config, use_fixed_code=None):
    pickle.dump(code_execution_data, open(os.path.join(this_config['output_dir'], 'code_execution_data.pkl'), 'wb'))
    code_execution_path = "python batch_execute_codes.py"
    command = [
        code_execution_path,
        "--input_path", os.path.join(this_config['output_dir'], 'code_execution_data.pkl'),
        "--output_path", os.path.join(this_config['output_dir'], 'code_execution_results.pkl'),
        "--seed", this_config['seed']
    ]
    if use_fixed_code:
        command.extend(["--fixed_code", f"'{use_fixed_code}'"])
    os.system(" ".join([str(c) for c in command]))
    if os.path.isdir(os.path.join(this_config['output_dir'], 'code_execution_results.pkl')):
        code_execution_results = []
        for f in os.listdir(os.path.join(this_config['output_dir'], 'code_execution_results.pkl')):
                code_execution_results.extend(pickle.load(open(os.path.join(this_config['output_dir'], 'code_execution_results.pkl', f), 'rb')))
    else:  
        code_execution_results = pickle.load(open(os.path.join(this_config['output_dir'], 'code_execution_results.pkl'), 'rb'))      
    code_execution_results = {c['sample_id']: c['result'] for c in code_execution_results}
    os.system('rm -r ' + os.path.join(this_config['output_dir'], 'code_execution_results.pkl'))   
    return code_execution_results

def engine_execution(formatted_inputs, this_config, engine_config_key='unit_test_generation', lora_path=None, engine_path="python ./llm_generation_engines/vllm_engine.py"):
    pickle.dump(formatted_inputs, open(os.path.join(this_config['output_dir'], 'engine_input.pkl'), 'wb'))
    engine_config = this_config[engine_config_key]
    if 'gpt4' in engine_config['model_name'].lower():
        index2output = {}
        for index, text in zip(formatted_inputs['index'], formatted_inputs['text']):
            messages = [
                {"role": "user", "content": text},
            ]
            try:
                response = openai.chat.completions.create(
                    model=engine_config['model_name'],
                    messages=messages,
                    seed=this_config['seed'],
                    max_tokens=engine_config['generation_kwargs']['max_new_tokens'],
                    temperature=engine_config['generation_kwargs']['temperature'],
                    top_p=engine_config['generation_kwargs']['top_p'],
                )

                response_content = response.choices[0].message.content
                system_fingerprint = response.system_fingerprint
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.total_tokens - response.usage.prompt_tokens

                index2output[index] = response_content
            except Exception as e:
                index2output[index] = f"Error: {e}"
        return index2output
        
    command = [
        engine_path, 
        '--model_name', engine_config['model_name'],
        '--half', engine_config['half'],
        '--generation_kwargs', f"'{json.dumps(OmegaConf.to_container(engine_config['generation']['generation_kwargs'], resolve=True))}'", 
        '--seed', this_config['seed'], 
        '--batch_size', engine_config['generation']['batch_size'], 
        '--data_path', os.path.join(this_config['output_dir'], 'engine_input.pkl'), 
        '--output_dir', this_config['output_dir']
    ]
    if lora_path:
        command.extend(['--lora_path', lora_path])
    os.system(" ".join([str(c) for c in command]))
    
    outputs = []
    for out_f in os.listdir(os.path.join(this_config['output_dir'], 'engine_output')):
        outputs.extend([json.loads(l.strip()) for l in open(os.path.join(this_config['output_dir'], 'engine_output', out_f)).readlines()])
    os.system('rm -r ' + os.path.join(this_config['output_dir'], 'engine_output'))
    index2output = {o['index']: o['generated_text'] for o in outputs}
    return index2output

def max_count_accuracy(outputs, answer, fixed_code=False, accuracy_fn=general_accuracy):
    if isinstance(outputs, dict):
        outputs = outputs.values()
    with_errors = [o for o in outputs]
    no_errors = [o for o in outputs if o['error'] is None]
    answers = [o['output'] for o in no_errors]
    if len(answers) == 0:
        if len(with_errors)>0 and fixed_code:
            return accuracy_fn([with_errors[0]['output']], [answer])
        else:
            return 0
    counts = {a:answers.count(a) for a in set(answers)}
    max_count = max(counts.values())
    max_count_answers = [a for a in counts if counts[a] == max_count]
    if len(max_count_answers) > 1:
       return 0
    else:
        max_count_answer = max_count_answers[0]
    return accuracy_fn([max_count_answer], [answer])

def get_penalty(error, compilation_error_penalty=0.0, runtime_error_penalty=0.0):
    if error is None:
        return 0
    elif 'execute_command' in error.lower():
        return compilation_error_penalty
    else:
        return runtime_error_penalty

def unit_test_score(selected_unit_tests, unit_test_results, compilation_error_penalty=0.1, runtime_error_penalty=0.1, acc_fn=general_accuracy,aggregator=np.mean):
    scores = defaultdict(int)        
    for code_index in range(len(unit_test_results[0])):
        curr_res = unit_test_results[0][code_index]['results']
        if isinstance(selected_unit_tests, dict):
            selected_unit_tests = selected_unit_tests[0][code_index]         
        curr_scores = []
        for t in range(len(curr_res)):
            curr_scores.append(sum([(acc_fn([ut['output']], [selected_unit_tests[t][1]]) \
                if ut['error'] is None else 0) -get_penalty(ut['error']) \
                    for ut in curr_res[t]['results']])/len(curr_res[t]['results']))
        scores[code_index] = aggregator(np.array(curr_scores))
    max_index = max(scores, key=scores.get)
    return max_index

            

def find_free_port():
    # Create a new socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Bind it to port 0, which tells the OS to select a free port
        s.bind(('localhost', 0))
        # Get the port number assigned
        port = s.getsockname()[1]
        return port


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    

def get_prompt_max_length(prompt_input, tokenizer, offset=3):
    if isinstance(prompt_input, list):
        prompt_input = prompt_input[0]
    prompt_input = tokenizer(
        prompt_input, return_tensors="pt", truncation=False)
    max_length = 100*((prompt_input['input_ids'].shape[1]//100)+offset)
    return max_length

def tokenize_dataset(dataset, tokenizer):
    # find max length 
    first_text = dataset['text'][0]
    prompt_input = tokenizer(first_text, return_tensors="pt", truncation=False)
    max_length = 100*((prompt_input['input_ids'].shape[1]//100)+3)
    
    dataset = datasets.Dataset.from_dict(dataset)
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.rename_column('text', 'prompt_text')   

    return tokenized_dataset

def create_unique_folder(base_path):
    # Generate a unique ID for the folder name
    unique_folder_name = str(uuid.uuid4())
    # Combine the base path with the unique folder name to form the full path
    full_path = os.path.join(base_path, unique_folder_name)
    # Create the folder
    os.makedirs(full_path)
    return full_path

def parse_config_options(config_options):
    if isinstance(config_options, list):
        config_options = ' '.join(config_options)
    options = {o.split('=')[0]: o.split('=')[1] for o in config_options.split(' ')}
    return options

def load_config(base_path, config_path, options):
    config = omegaconf.OmegaConf.load(base_path)
    config = omegaconf.OmegaConf.merge(config, omegaconf.OmegaConf.load(config_path))
    if options is not None:
        config_options = parse_config_options(options)
        config = omegaconf.OmegaConf.merge(config, omegaconf.OmegaConf.create(config_options))
    return config

def get_base_prompt(prompt_file, in_context_examples_file, num_in_context_examples):
    with open(prompt_file) as f:
        base_prompt = f.read().strip()

    # Load in-context examples
    with open(in_context_examples_file) as f:
        in_context_examples = json.load(f)
        in_context_examples = random.sample(in_context_examples, num_in_context_examples)
        in_context_examples = [f"Query: {example['question']}\nProgram:\n{example['code']}" for example in in_context_examples]

    base_prompt = base_prompt.replace("INSERT_CONTEXT_HERE", "\n\n".join(in_context_examples))
    return base_prompt
    
def get_visual_program_prompt(batch, base_prompt, model_type, tokenizer=None):
    if tokenizer == None and 'gpt4' not in model_type.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_type, trust_remote_code=True)
    if 'gpt4' in model_type.lower():
        batch = [base_prompt.replace('INSERT_QUERY_HERE', s) for s in batch]
    elif 'llama' in model_type.lower():
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        batch = ['<s>'+base_prompt.replace('INSERT_QUERY_HERE', s) for s in batch]
        # batch = [
        #     [
        #         # {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": base_prompt.replace('INSERT_QUERY_HERE', s)}
        #     ] for s in batch
        # ]
        # batch = [tokenizer.apply_chat_template(s, tokenize=False) for s in batch]
    elif tokenizer.chat_template is None:
        return ['<s>' + base_prompt.replace('INSERT_QUERY_HERE', s) for s in batch]
    else:
        tokenizer.padding_side = "left"
        batch = [
            [
            # {'role': "system", "content": system_prompt},
            { 'role': 'user', 'content': base_prompt.replace('INSERT_QUERY_HERE', s)}
            ] for s in batch
        ]
        batch = [tokenizer.apply_chat_template(s, tokenize=False) for s in batch]
    return batch

def get_visual_program_error_correction_prompt(question, correction_prompt, code, output, model_type, tokenizer=None):
    if tokenizer == None:
        tokenizer = AutoTokenizer.from_pretrained(model_type, trust_remote_code=True)
    code = extract_python_code(code)
    error = output['error']
    correction_prompt = correction_prompt.replace('INSERT_QUERY_HERE', question).replace('INSERT_CODE_HERE', code).replace('INSERT_ERROR_HERE', error)
    return correction_prompt

def get_visual_program_correction_prompt(question, correction_prompt, code, unit_tests, unit_test_results, model_type, tokenizer=None):
    if tokenizer == None:
        tokenizer = AutoTokenizer.from_pretrained(model_type, trust_remote_code=True)
    unit_test_template = 'Test INDEX_LETTER\nImage Content: INSERT_UNIT_TEST_HERE\nGround Truth Answer: "INSTERT_UNIT_TEST_ANSWER_HERE"\nProgram Output: "INSERT_UNIT_TEST_RESULT_HERE"'
    
    unit_test_outputs = []
    code = extract_python_code(code)
    for unit_test_result in unit_test_results:
        unit_test_result = unit_test_result
        if unit_test_result['error'] != None:
            unit_test_outputs.append(f"Error: {unit_test_result['error']}")
        else:
            unit_test_outputs.append(f"{unit_test_result['output']}")
    unit_test_input = "\n".join([unit_test_template.replace("INDEX_LETTER", chr(ord('A')+i))
                                 .replace('INSERT_UNIT_TEST_HERE', unit_tests[i][0])
                                 .replace('INSTERT_UNIT_TEST_ANSWER_HERE', unit_tests[i][1])
                                 .replace('INSERT_UNIT_TEST_RESULT_HERE', unit_test_outputs[i]) for i in range(len(unit_tests))])
    correction_prompt = correction_prompt.replace('INSERT_QUERY_HERE', question).replace('INSERT_CODE_HERE', code).replace('INSERT_UNIT_TEST_OUTPUTS_HERE', unit_test_input)
    return correction_prompt

def extract_python_code(text):
    text = text.replace('"""', '')
    code = text.split('Query:')[0]
    if len(code.split('def execute_command(image) -> str:')) > 1:
        code = code.split('def execute_command(image) -> str:')[1]
    elif len(code.split('def execute_command(image):')) > 1:
        code = code.split('def execute_command(image):')[1]
    # Define regex pattern to identify potential code blocks
    code_pattern = re.compile(r"(?:^|\n)( {4}|\t).*", re.MULTILINE)
    
    # Split the text into lines
    lines = code.split('\n')
    
    # Initialize variables to store code blocks and a flag to indicate if we are inside a code block
    code_blocks = []
    current_code_block = []
    inside_code_block = False
    
    for line in lines:
        if code_pattern.match(line):
            # If the line matches the code pattern, add it to the current code block
            current_code_block.append(line)
            inside_code_block = True
        else:
            if inside_code_block:
                # If we were inside a code block and encounter a non-code line, save the current code block
                code_blocks.append('\n'.join(current_code_block))
                current_code_block = []
                inside_code_block = False
    
    # If the text ends while still inside a code block, save the current code block
    if inside_code_block:
        code_blocks.append('\n'.join(current_code_block))
    
    return f'def execute_command(image):\n' +"\n".join(code_blocks)


def exec_code(pred_code, im, fixed_code=None):
    from image_patch import (ImagePatch, 
                             bool_to_yesno,
                             best_image_match, 
                             distance,
                             coerce_to_numeric,
                             process_guesses, 
                             llm_query
    )
    code_result = {"code": pred_code, "error": None, "output": "error"}
    # code = extract_python_code(code)
    if 'def execute_command(image)' not in pred_code:
        code = f'def execute_command(image):\n' + pred_code
    else:
        code = pred_code
    try:
        code = astunparse.unparse(ast.parse(code))
        exec(compile(code, 'Codex', 'exec'), globals())
        x = execute_command(im)
        code_result["output"] = x
        code_result['error'] = None
        return code_result
    except Exception as e:
        if fixed_code:
            try:
                if 'def execute_command(image)' not in fixed_code:
                    fixed_code = f'def execute_command(image):\n' + fixed_code
                else:
                    fixed_code = fixed_code
                code = astunparse.unparse(ast.parse(fixed_code))
                exec(compile(code, 'Codex', 'exec'), globals())
                x = execute_command(im)
                code_result["error"] = str(e)
                code_result["output"] = x
                return code_result
            except:
                code_result['error'] = str(e)
                return code_result
        else:
            code_result['error'] = str(e)
            return code_result
            

def get_fixed_code(task):
    if task == 'VQA':
        code ="\n    image_patch = ImagePatch(image)\n    return image_patch.simple_query(\"{}\")\n"
    elif task == 'ITM':
        code = "\n    image_patch = ImagePatch(image)\n    text = \"{}\"\n    text=text.split(\"text=\")[-1].lower().strip()\n    if image_patch._detect(text, .7):\n        return \"yes\"\n    else:\n        return \"no\"\n"
    else:
        code = "\n    image_patch = ImagePatch(image)\n    return image_patch.simple_query(\"{}\")\n"
    return code


class SynonymChecker:
    def __init__(self, model_name='all-MiniLM-L6-v2', threshold=0.9):
        """
        Initializes the SynonymChecker with a specific model and similarity threshold.
        
        Parameters:
        - model_name: str, the name of the Sentence Transformer model to use.
        - threshold: float, the cosine similarity threshold for considering phrases as synonymous.
        """
        from sentence_transformers import SentenceTransformer, util
        import nltk
        try:
            nltk.download('wordnet')
        except:
            pass
        from nltk.corpus import wordnet as wn
        self.wn = wn
        self.cos_sim = util.pytorch_cos_sim
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold

    def set_threshold(self, new_threshold):
        """
        Updates the similarity threshold.
        
        Parameters:
        - new_threshold: float, the new threshold value.
        """
        self.threshold = new_threshold
    
    def is_hyponym(self, word1, word2):
        """
        Checks if word1 is a hyponym of word2 using WordNet.
        """
        hyponyms = []
        for synset in self.wn.synsets(word2):
            for hyponym in synset.hyponyms():
                hyponyms.extend(hyponym.lemma_names())
        return word1 in hyponyms

    def are_synonymous(self, phrase1, phrase2, postproc=True):
        """
        Determines if two phrases are synonymous based on the cosine similarity of their embeddings.
        
        Parameters:
        - phrase1: str, the first phrase to compare.
        - phrase2: str, the second phrase to compare.
        
        Returns:
        - bool, True if the phrases are considered synonymous, False otherwise.
        """
        if any([p is None for p in [phrase1, phrase2]]):
            return 0.0
        if postproc:
            phrase1 = general_postprocessing(phrase1)
            phrase2 = general_postprocessing(phrase2)
        embedding1 = self.model.encode(phrase1, convert_to_tensor=True)
        embedding2 = self.model.encode(phrase2, convert_to_tensor=True)
        cosine_sim = self.cos_sim(embedding1, embedding2)
        return float(cosine_sim.item() > self.threshold) or self.is_hyponym(phrase1, phrase2) or self.is_hyponym(phrase2, phrase1)


def initialize_image_generator(this_config, **kwargs):
    if this_config['image_generation']['image_source'] == 'web':
        from unit_test_generation.unit_test_image import WebRetriever
        image_retriever = WebRetriever(
            model_name = this_config['image_generation']['reranking_model_name'],
            top_k = this_config['image_generation']['top_k'],
            return_k = this_config['image_generation']['return_k'],
            scorer = this_config['image_generation']['image_scorer'],
            return_image=this_config['image_generation']['return_image']
            )
    elif this_config['image_generation']['image_source'] == 'diffusion':
        from unit_test_generation.unit_test_image import DiffusionRetriever
        image_retriever = DiffusionRetriever(
            model_name = this_config['image_generation']['diffusion_model_name'],
            return_k=this_config['image_generation']['return_k'],
            guidance_scale=this_config['image_generation']['guidance_scale'],
            num_inference_steps=this_config['image_generation']['num_inference_steps'],
            gligen_scheduled_sampling_beta=this_config['image_generation']['gligen_scheduled_sampling_beta'],
            seed = this_config['seed'],
            return_image=this_config['image_generation']['return_image'],
            output_dir=kwargs.get('output_dir', f"{this_config['output_dir']}/unit_test_images"),
            batch_size=this_config['image_generation']['image_batch_size'],
            distributed=this_config['image_generation']['distributed']
        )
    elif this_config['image_generation']['image_source'] == 'cc12m':
        from unit_test_generation.unit_test_image import ConceptualCaptionsRetriever
        image_retriever = ConceptualCaptionsRetriever(
            model_name = this_config['image_generation']['reranking_model_name'],
            search_by = this_config['image_generation']['search_by'],
            top_k = this_config['image_generation']['top_k'],
            return_k = this_config['image_generation']['return_k'],
            reprompt_with_description=this_config['image_generation']['reprompt_with_description'],
            return_image=this_config['image_generation']['return_image']
            )
    return image_retriever



def create_unique_file(directory, base_filename="myfile", extension=".txt"):
    # Get the current datetime
    now = datetime.now()
    datetime_str = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Convert to milliseconds
    
    # Create the initial filename
    filename = f"{base_filename}_{datetime_str}{extension}"
    file_path = os.path.join(directory, filename)
    
    # Increment a counter until a unique filename is found
    counter = 1
    while os.path.exists(file_path):
        filename = f"{base_filename}_{datetime_str}_{counter}{extension}"
        file_path = os.path.join(directory, filename)
        counter += 1
    return file_path


def load_image(image_path_or_url):
    if image_path_or_url.startswith('http'):
        response = requests.get(image_path_or_url)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path_or_url)
    return image