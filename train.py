"""
 Copyright (c) 2024, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import argparse
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
import gc
import os
import copy
import ray
import pickle
import json
import random
from omegaconf import OmegaConf
from PIL import Image
from collections import defaultdict, Counter
from utils import (load_config,
                    get_base_prompt,
                    get_visual_program_prompt,
                    extract_python_code,
                    get_visual_program_correction_prompt,
                    SynonymChecker,
                    set_seed,
                    get_fixed_code,
                    initialize_image_generator,
                    engine_execution,
                    code_execution,
                    unit_test_score, 
                    get_penalty
                    )
from my_datasets import get_dataset_class
from unit_test_generation.processing import extract_unit_tests, get_unit_test_prompt, get_grounded_diffusion_prompt
from unit_test_generation.unit_test_sampling import TextSampler
from accuracy_fns import get_accuracy_fn


### ARGUMENT PARSING ###
parser = argparse.ArgumentParser(
    description='Train visual programming with unit test feedback')
parser.add_argument('--base_path', type=str, default='viunit_configs/base.yaml',
                    help='Path to the configuration file')
parser.add_argument('--config_path', type=str,
                    default='viunit_configs/base.yaml', help='Path to the configuration file')
parser.add_argument('--config_options', nargs='+',
                    help='Options to override the configuration file')
parser.add_argument('--data_path', type=str, default='',
                    help='Precomputed Data path')
parser.add_argument('--unit_test_path', type=str, default='',
                    help='Precomputed Unit Test path')
parser.add_argument('--load_images',type=int, default=1,
                    help='Load precomputed images')
parser.add_argument('--load_selected',type=int, default=1,
                    help='Load selection unit tests')
parser.add_argument('--execute_unit_tests',type=int, default=1,
                    help='reexecute unit tests')
parser.add_argument('--code_path', type=str, default='',
                    help='Precomputed Code path')
parser.add_argument('--only_first_iter', type=int, default=0,
                    help='Precomputed Code path')
parser.add_argument('--reload', type=int, default=0,
                    help='Precomputed Code path')
parser.add_argument('--recompute_unit_tests', type=int, default=0,
                    help='Recompute unit test score')
parser.add_argument('--lora_path', type=str, default=None,
                    help='Lora Path')

args = parser.parse_args()

## CONFIG SETUP ##
print(args.config_path)
this_config = load_config(
    args.base_path, args.config_path, args.config_options)

set_seed(this_config['seed'])

## TASK SETUP ## 
task = this_config['data']['task']
acc_fn = get_accuracy_fn(this_config['data']['dataset_name'])
fixed_code = get_fixed_code(task)

os.makedirs(this_config['output_dir'], exist_ok=True)

## LOGGING ## 
log_file = open(os.path.join(this_config['output_dir'], 'log.txt'), 'a')
json.dump(OmegaConf.to_container(this_config, resolve=True), open(
    os.path.join(this_config['output_dir'], 'config.json'), 'w'), indent=4)


#### SET UP DATASET ####
dataset_class = get_dataset_class(this_config['data']['dataset_name'])
dataset = dataset_class(**this_config['data']['dataset_args'])
save_file = os.path.join(this_config['output_dir'], 'data.p')
if os.path.exists(save_file) and args.reload:
    dataset.data = pickle.load(open(save_file, 'rb'))
elif args.data_path:
    dataset.data = pickle.load(open(args.data_path, 'rb'))
## add indices
if 'index' not in dataset.data[0]:
    for i, d in enumerate(dataset.data):
        d['index'] = i

## LOAD CACHED CODE ##
if args.code_path:
    code_data = pickle.load(open(args.code_path, 'rb'))
    for i, d in enumerate(dataset.data):
        if 'correction_prompt' in code_data[i]:     
            d['correction_prompt'] = code_data[i]['correction_prompt']
        if args.only_first_iter:
            d['generated_code'] = {0: code_data[i]['generated_code'][0]}
        else:        
            d['generated_code'] = code_data[i]['generated_code']

## LOAD CACHED UNIT_TESTS ## 
if this_config['do_unit_test'] and args.unit_test_path:
    print("Loading unit tests...")
    if args.load_images:
        print("with images...")
    if args.load_selected:
        print("with selected unit tests...")
    unit_test_data = pickle.load(open(args.unit_test_path, 'rb'))
    for i, d in enumerate(dataset.data):
        dataset.data[i]['unit_tests'] = unit_test_data[i]['unit_tests']
        if args.load_selected:
            dataset.data[i]['selected_unit_tests'] = unit_test_data[i]['selected_unit_tests']
            if args.load_images:
                dataset.data[i]['unit_test_images'] = unit_test_data[i]['unit_test_images']

## SETUP ENGINES ##
if this_config['llm_engine'] == 'vllm':
    engine_path = 'python llm_generation_engines/vllm_engine.py'
else:
    engine_path = 'accelerate launch llm_generation_engines/hf_engine.py'
train_path = "accelerate launch train_virep.py"
      

## SETUP PROMPTS and TOKENIZERS ##
if this_config['do_unit_test']:
    # unit test generation
    unit_test_system_prompt = open(
        this_config['unit_test_generation']['generation']['prompt_file']).read()
    unit_test_in_context_examples = open(
        this_config['unit_test_generation']['generation']['in_context_examples_file']).read()
    with open(os.path.join(this_config['output_dir'], 'unit_test_system_prompt.txt'), 'w') as f:
        f.write(unit_test_system_prompt)
    with open(os.path.join(this_config['output_dir'], 'unit_test_in_context_examples.txt'), 'w') as f:
        f.write(unit_test_in_context_examples)
        
    if 'gpt' not in this_config['unit_test_generation']['model_name']:
        unit_test_tokenizer = AutoTokenizer.from_pretrained(
            this_config['unit_test_generation']['model_name'], trust_remote_code=True)
    else:
        unit_test_tokenizer = None

    correction_prompt = open(
    this_config['visual_program_generator']['generation']['correction_prompt_file']).read()
    with open(os.path.join(this_config['output_dir'], 'correction_prompt.txt'), 'w') as f:
        f.write(correction_prompt)
    
    if this_config['image_generation']['image_source'] == 'diffusion' and any([k in this_config['image_generation']['diffusion_model_name'].lower() for k in ['lmd', 'gligen']]):
        lm_grounded_diffusion_in_context_prompt = open(this_config['image_generation']['generation']['in_context_examples_file']).read().strip()
        lm_grounded_diffusion_system_prompt = open(this_config['image_generation']['generation']['prompt_file']).read().strip()
        if 'gpt' not in this_config['image_generation']['model_name'].lower():
            lm_grounded_diffusion_tokenizer = AutoTokenizer.from_pretrained(this_config['image_generation']['model_name'], trust_remote_code=True)
        else:
            lm_grounded_diffusion_tokenizer = None


base_prompt = get_base_prompt(this_config['visual_program_generator']['generation']['prompt_file'],
                              this_config['visual_program_generator']['generation']['in_context_examples_file'],
                              this_config['visual_program_generator']['generation']['num_in_context_examples']
                              )
with open(os.path.join(this_config['output_dir'], 'base_prompt.txt'), 'w') as f:
    f.write(base_prompt)
program_tokenizer = AutoTokenizer.from_pretrained(
    this_config['visual_program_generator']['model_name'], trust_remote_code=True)


start_iter = 0
if len(dataset.data) > 0 and 'generated_code' in dataset.data[0]:
    start_iter = max([max(list(d['generated_code'].keys())) for d in dataset.data])


for iter in range(start_iter, this_config['execution']['feedback_max_iters']):
    print(f"Iteration {iter}")
    log_file.write(f"Iteration {iter}\n")
    
    if len(dataset.data) > 0 and 'generated_code' in dataset.data[0] and any([iter in dataset.data[idx]['generated_code'] for idx in range(len(dataset))]):
        print('Visual programs already generated..')
    else:
        print("Generating visual programs....")
        formatted_inputs = {
            'text': get_visual_program_prompt(
                [b_['text'] for b_ in dataset], 
                base_prompt, 
                this_config['visual_program_generator']['model_name'], 
                program_tokenizer
                ),
            'index': [b_['index'] for b_ in dataset]}
        if iter > 0 and this_config['do_train']:
            index2out = engine_execution(formatted_inputs, this_config, 'visual_program_generator', engine_path=engine_path, lora_path=args.lora_path)
        else:
            index2out = engine_execution(formatted_inputs, this_config, 'visual_program_generator', engine_path=engine_path)
        for i in index2out:
            if isinstance(index2out[i], str):
                index2out[i] = [index2out[i]]
                
        index2input = {i:t for i,t in zip(formatted_inputs['index'], formatted_inputs['text'])}
        for i in range(len(dataset.data)):
            if 'generated_code' not in dataset.data[i]:
                dataset.data[i]['generated_code'] = {}
            if 'code_prompt' not in dataset.data[i]:
                dataset.data[i]['code_prompt'] = {}
            dataset.data[i]['code_prompt'][iter] = index2input[dataset.data[i]['index']]
            dataset.data[i]['generated_code'][iter] = index2out[dataset.data[i]['index']]  
        pickle.dump(dataset.data, open(save_file, 'wb'))
    
    
    if this_config['do_unit_test']:
        ### Generate candidate unit tests ###
        if this_config['unit_test_generation']['use_program'] and \
            ('unit_tests' not in dataset.data[0] or \
                not any([iter in dataset.data[idx]['unit_tests'] for idx in range(len(dataset))])):
                formatted_inputs = {'text': [], 'index': []}
                for i, d in enumerate(dataset):
                    if iter in d['generated_code']:
                        formatted_inputs['text'].extend(
                            get_unit_test_prompt([d['text']]* len(d['generated_code'][iter]),
                                                    unit_test_system_prompt,
                                                    unit_test_in_context_examples, 
                                                    this_config['unit_test_generation']['model_name'], 
                                                    unit_test_tokenizer, 
                                                    program=[extract_python_code(p) for p in d['generated_code'][iter]]
                                                    )
                            )
                        formatted_inputs['index'].extend([f"{d['index']}-{c}" for c in range(len(d['generated_code'][iter]))])
                if len(formatted_inputs['text']) == 0:
                    exit(0)
                index2out = engine_execution(formatted_inputs, this_config, 'unit_test_generation', engine_path=engine_path)
                index2input = {i:t for i,t in zip(formatted_inputs['index'], formatted_inputs['text'])}
                for i in range(len(dataset.data)):
                    if iter in dataset.data[i]['generated_code']:
                        idx = dataset.data[i]['index']
                        if 'unit_tests' not in dataset.data[i]:
                            dataset.data[i]['unit_tests'] = {}
                        if 'unit_test_prompt' not in dataset.data[i]:
                            dataset.data[i]['unit_test_prompt'] = {}
                        dataset.data[i]['unit_test_prompt'][iter] = index2input[f'{idx}-{c}']
                        dataset.data[i]['unit_tests'][iter] = [
                            extract_unit_tests(
                                index2out[f'{idx}-{c}']) for c in range(len(dataset.data[i]['generated_code'][iter]))
                            ]
                pickle.dump(dataset.data, open(save_file, 'wb'))
        elif 'unit_tests' not in dataset.data[0] and \
            not this_config['unit_test_generation']['use_program']: 
            print("Generating unit tests...")
            formatted_inputs = {
                'text': get_unit_test_prompt(
                    [b_['text'] for b_ in dataset], 
                    unit_test_system_prompt,
                    unit_test_in_context_examples, 
                    this_config['unit_test_generation']['model_name'], 
                    unit_test_tokenizer
                    ),
                'index': [b_['index'] for b_ in dataset]}
            if len(formatted_inputs['text']) == 0:
                exit(0)
            index2out = engine_execution(formatted_inputs, this_config, 'unit_test_generation', engine_path=engine_path)
            index2input = {i:t for i,t in zip(formatted_inputs['index'], formatted_inputs['text'])}
            for i in range(len(dataset.data)):
                idx = dataset.data[i]['index']
                dataset.data[i]['unit_test_prompt'] = index2input[idx]
                dataset.data[i]['unit_tests'] = extract_unit_tests(
                    index2out[idx])
            pickle.dump(dataset.data, open(save_file, 'wb'))
        else:
            print("Unit tests already generated...")
        
        ### Sample unit tests ###
        if this_config['unit_test_generation']['use_program'] and \
            ('selected_unit_tests' not in dataset.data[0] or \
                not any([iter in dataset.data[idx]['selected_unit_tests'] for idx in range(len(dataset))])):
            print("Sampling unit tests....")
            text_sampler = TextSampler(
                model_name = this_config['unit_test_sampling']['model_name'],
                sampling_strategy=this_config['unit_test_sampling']['strategy'],
                filter_long_answers=this_config['unit_test_sampling']['filter_long_answers']
                )
            for i in tqdm(range(len(dataset.data))):
                if 'selected_unit_tests' not in dataset.data[i]:
                        dataset.data[i]['selected_unit_tests'] = {}
                if iter in dataset.data[i]['unit_tests']:
                    dataset.data[i]['selected_unit_tests'][iter] = []
                    for code_index, code in enumerate(dataset.data[i]['generated_code'][iter]):
                        dataset.data[i]['selected_unit_tests'][iter].append(text_sampler.sample(
                                                dataset.data[i]['unit_tests'][iter][code_index], 
                                                num_samples=this_config['unit_test_sampling']['num_unit_tests'],
                                                ))
            pickle.dump(dataset.data, open(save_file, 'wb'))
            text_sampler.clear_sampler()
            del text_sampler
            gc.collect()
            torch.cuda.empty_cache()
        elif 'selected_unit_tests' not in dataset.data[0] and \
            not this_config['unit_test_generation']['use_program']:
            print("Sampling unit tests....")
            text_sampler = TextSampler(
                model_name = this_config['unit_test_sampling']['model_name'],
                sampling_strategy=this_config['unit_test_sampling']['strategy'],
                filter_long_answers=this_config['unit_test_sampling']['filter_long_answers']
                )
            for i in tqdm(range(len(dataset.data))):
                dataset.data[i]['selected_unit_tests'] = text_sampler.sample(
                                        dataset.data[i]['unit_tests'], 
                                        num_samples=this_config['unit_test_sampling']['num_unit_tests'],
                                        )
            pickle.dump(dataset.data, open(save_file, 'wb'))
            text_sampler.clear_sampler()
            del text_sampler
            gc.collect()
            torch.cuda.empty_cache()
        else:
            print("Unit tests already sampled...")
        
        ## Generate images for unit tests ##
        if this_config['unit_test_generation']['use_program']  and ('unit_test_images' not in dataset.data[0] or not any([iter in dataset.data[idx]['unit_test_images'] for idx in range(len(dataset))])):
            print("Sampling images with program...")
            image_retriever = initialize_image_generator(this_config)
            queries, idx2lengths = [], {}
            for d in dataset.data:
                if iter in d['selected_unit_tests']:
                    idx2lengths[d['index']] = [len(u) for u in d['selected_unit_tests'][iter]]
                    for code_ut in d['selected_unit_tests'][iter]:
                        queries.extend([uu[0] for uu in code_ut])
            
            if this_config['image_generation']['image_source'] == 'diffusion' and any([k in this_config['image_generation']['diffusion_model_name'].lower() for k in ['lmd', 'gligen']]):
                formatted_inputs = {
                    'text': get_grounded_diffusion_prompt(
                        queries ,
                        lm_grounded_diffusion_system_prompt,
                        lm_grounded_diffusion_in_context_prompt, 
                        this_config['unit_test_generation']['model_name'], 
                        lm_grounded_diffusion_tokenizer) ,
                    'index': list(range(len(queries)))}
                index2out = engine_execution(formatted_inputs, 'image_generation')
                llm_response = [index2out[i] for i in range(len(queries))]
                images = []
                image_batch_size = this_config['image_generation']['image_batch_size']*torch.cuda.device_count()
                if this_config['image_generation']['distributed']:
                    diffuser_input_queries = queries
                    diffuser_llm_response = llm_response
                    images = image_retriever.batch_fetch_image(diffuser_input_queries, diffuser_llm_response)
                else:
                    for i in tqdm(range(0, len(queries), image_batch_size)):
                        images.extend(image_retriever.batch_fetch_image(queries[i:i+image_batch_size], llm_response[i:i+image_batch_size]))
            else:
                if this_config['image_generation']['distributed']:
                    diffuser_input_queries = queries
                    images = image_retriever.batch_fetch_image(diffuser_input_queries)
                else:
                    images = []
                    image_batch_size = this_config['image_generation']['image_batch_size']*torch.cuda.device_count()
                    for i in tqdm(range(0, len(queries), image_batch_size)):
                        images.extend(image_retriever.batch_fetch_image(queries[i:i+image_batch_size]))
            image_retriever.clear_retriever()
            gc.collect()
            torch.cuda.empty_cache()
                
            start = 0
            for idx,d in enumerate(dataset.data):
                if iter in d['selected_unit_tests']:
                    if 'unit_test_images' not in d:
                        dataset.data[idx]['unit_test_images'] = {}
                    dataset.data[idx]['unit_test_images'][iter] = []
                    for j in range(len(idx2lengths[d['index']])): # code index
                            dataset.data[idx]['unit_test_images'][iter].append(images[start:start+idx2lengths[d['index']][j]])
                            start += idx2lengths[d['index']][j]      
        elif 'unit_test_images' not in dataset.data[0] and not this_config['unit_test_generation']['use_program']:
            print("Sampling images...")
            image_retriever = initialize_image_generator(this_config)
            queries = [u[0] for d in dataset.data for u in d['selected_unit_tests']]
            if this_config['image_generation']['image_source'] == 'diffusion' and any([k in this_config['image_generation']['diffusion_model_name'].lower() for k in ['lmd', 'gligen']]):
                formatted_inputs = {
                    'text': get_grounded_diffusion_prompt(
                        queries ,
                        lm_grounded_diffusion_system_prompt,
                        lm_grounded_diffusion_in_context_prompt, 
                        this_config['unit_test_generation']['model_name'], 
                        lm_grounded_diffusion_tokenizer) ,
                    'index': list(range(len(queries)))}
                index2out = engine_execution(formatted_inputs, this_config, 'image_generation', engine_path=engine_path)
                llm_response = [index2out[i] for i in range(len(queries))]
                images = []
                image_batch_size = this_config['image_generation']['image_batch_size']*torch.cuda.device_count()
                if this_config['image_generation']['distributed']:
                    diffuser_input_queries = queries
                    diffuser_llm_response = llm_response
                    images = image_retriever.batch_fetch_image(diffuser_input_queries, diffuser_llm_response)
                else:
                    for i in tqdm(range(0, len(queries), image_batch_size)):
                        images.extend(image_retriever.batch_fetch_image(queries[i:i+image_batch_size], llm_response[i:i+image_batch_size]))
            else:
                if this_config['image_generation']['distributed']:
                    diffuser_input_queries = queries
                    images = image_retriever.batch_fetch_image(diffuser_input_queries)
                else:
                    images = []
                    image_batch_size = this_config['image_generation']['image_batch_size']*torch.cuda.device_count()
                    for i in tqdm(range(0, len(queries), image_batch_size)):
                        images.extend(image_retriever.batch_fetch_image(queries[i:i+image_batch_size]))
            image_retriever.clear_retriever()
            gc.collect()
            torch.cuda.empty_cache()
            start = 0
            lengths = [len(d['selected_unit_tests']) for d in dataset.data]
            for idx in range(len(lengths)):
                dataset.data[idx]['unit_test_images'] = images[start:start+lengths[idx]]
                start += lengths[idx]
            pickle.dump(dataset.data, open(save_file, 'wb'))  
        else:
            print("Images already sampled...")
            

        if this_config['execution']['synonym_checker']:
            checker = SynonymChecker()
        if len(dataset.data) > 0 and 'unit_test_results' in dataset.data[0] and \
            any([iter in dataset.data[idx]['unit_test_results'] for idx in range(len(dataset))]) and \
                not args.execute_unit_tests and not args.recompute_unit_tests:
            print('Unit tests already executed..')
        elif not args.execute_unit_tests and args.recompute_unit_tests and len(dataset.data) > 0 and \
            'unit_test_results' in dataset.data[0] and \
                any([iter in dataset.data[idx]['unit_test_results'] for idx in range(len(dataset))]):
            print('Recomputing unit tests...')
            for idx in range(0, len(dataset.data)):
                if iter > 0 and iter not in dataset.data[idx]['generated_code']:
                    continue
                unit_test_pre = copy.deepcopy(dataset.data[idx]['unit_test_results'][iter])
                dataset.data[idx]['unit_test_results'][iter] = []
                for code_index, code in enumerate(dataset.data[idx]['generated_code'][iter]):
                    unit_tests = dataset.data[idx]['selected_unit_tests']
                    images = dataset.data[idx]['unit_test_images']
                    if this_config['unit_test_generation']['use_program']:
                        unit_tests = unit_tests[iter][code_index]
                        images = dataset.data[idx]['unit_test_images'][iter][code_index]
                    unit_test_results = []
                    for ut_index, ut in enumerate(unit_tests):
                        curr_unit_test = []
                        if isinstance(images[ut_index], tuple):
                            try:
                                curr_images = images[ut_index][1]
                            except:
                                curr_images = images[ut_index][0] 
                        else:
                            curr_images = images[ut_index]
                            
                        for im_idx,im in enumerate(curr_images):
                            if im is None or im=='':
                                print(f"No image for unit test for data index {dataset.data[idx]['index']}, Code Index: {code_index}, UT Index: {ut_index}")
                                log_file.write(f"No image for unit test for data index {dataset.data[idx]['index']}, Code Index: {code_index}, UT Index: {ut_index}\n")
                                continue
                            
                            res = unit_test_pre[code_index]['results'][ut_index]['results'][im_idx]
                            penalty = get_penalty(res['error'], this_config['execution']['error_penalty']['syntax'], this_config['execution']['error_penalty']['runtime'])
                            if res['error'] is not None:
                                res['acc'] = -penalty
                            else:
                                if this_config['execution']['synonym_checker']:
                                    res['acc'] = checker.are_synonymous(res['output'], ut[1])
                                else:
                                    res['acc'] = acc_fn([res['output']], [ut[1]])
                            curr_unit_test.append(res)
                        if len(curr_unit_test) == 0:
                            print(f"No images for unit test {dataset.data[idx]['index']}-{code_index}-{ut_index}")
                            curr_unit_test = {'results': curr_unit_test, 'acc': 0.}
                        else:
                            curr_scores = [r['acc'] for r in curr_unit_test]
                            if this_config['execution']['image_agg'] == 'mean':
                                curr_unit_test = {'results': curr_unit_test, 'acc': sum(curr_scores)/len(curr_unit_test)}
                            elif this_config['execution']['image_agg'] == 'max':
                                curr_unit_test = {'results': curr_unit_test, 'acc': max(curr_scores)}
                            elif this_config['execution']['image_agg'] == 'majority':
                                acc_count = Counter(curr_scores)
                                curr_unit_test = {'results': curr_unit_test, 'acc': acc_count.most_common(1)[0][0]}
                        unit_test_results.append(curr_unit_test)
                    
                    curr_ut_scores = [r['acc'] for r in unit_test_results]
                    dataset.data[idx]['unit_test_results'][iter].append({
                        'results': unit_test_results, 
                        'acc': sum(curr_ut_scores)/len(unit_test_results)
                        })
            pickle.dump(dataset.data, open(save_file, 'wb'))
        else:
            print("Executing unit tests....")
            code_execution_data = [] #sample_id, answer, possible_answers, query_type, query, img
            for idx in tqdm(range(0, len(dataset.data))):
                if iter > 0 and iter not in dataset.data[idx]['generated_code']:
                    continue
                for code_index, code in enumerate(dataset.data[idx]['generated_code'][iter]):
                    unit_tests = dataset.data[idx]['selected_unit_tests']
                    images = dataset.data[idx]['unit_test_images']
                    if this_config['unit_test_generation']['use_program']:
                        unit_tests = unit_tests[iter][code_index]
                        images = dataset.data[idx]['unit_test_images'][iter][code_index]
                    generated_code = extract_python_code(code)
                    if len(generated_code.split('def execute_command(image) -> str:\n')) > 1:
                        generated_code = generated_code.split('def execute_command(image) -> str:\n')[1]
                    elif len(generated_code.split('def execute_command(image):\n')) > 1:
                        generated_code = generated_code.split('def execute_command(image):\n')[1]
                    unit_test_results = []
                    for ut_index, ut in enumerate(unit_tests):
                        curr_unit_test = []
                        if isinstance(images[ut_index], tuple):
                            try:
                                curr_images = images[ut_index][1]
                            except:
                                curr_images = images[ut_index][0]
                        else:
                            curr_images = images[ut_index]
                        for im_idx,im in enumerate(curr_images):
                            if im is None:
                                continue
                            code_execution_data.append({'sample_id': f"{idx}_{code_index}_{ut_index}_{im_idx}", 
                                                        'answer': ut[1], 
                                                        'possible_answers': [ut[1]], 
                                                        'query_type': 'image', 
                                                        'codes': generated_code,
                                                        'query': dataset.data[idx]['question'], 
                                                        'image': im
                                                        })
            

            code_execution_results = code_execution(code_execution_data, this_config, use_fixed_code=None)
            
            for idx in tqdm(range(0, len(dataset.data))):
                if iter > 0 and iter not in dataset.data[idx]['generated_code']:
                    continue
                if 'unit_test_results' not in dataset.data[idx]:
                        dataset.data[idx]['unit_test_results'] = {}
                dataset.data[idx]['unit_test_results'][iter] = []
                for code_index, code in enumerate(dataset.data[idx]['generated_code'][iter]):
                    unit_tests = dataset.data[idx]['selected_unit_tests']
                    images = dataset.data[idx]['unit_test_images']
                    if this_config['unit_test_generation']['use_program']:
                        unit_tests = unit_tests[iter][code_index]
                        images = dataset.data[idx]['unit_test_images'][iter][code_index]
                    unit_test_results = []
                    for ut_index, ut in enumerate(unit_tests):
                        curr_unit_test = []
                        if isinstance(images[ut_index], tuple):
                            try:
                                curr_images = images[ut_index][1]
                            except:
                                curr_images = images[ut_index][0]
                        else:
                            curr_images = images[ut_index]
                        for im_idx,im in enumerate(curr_images):
                            if im is None or im=='':
                                print(f"No image for unit test for data index {dataset.data[idx]['index']}, Code Index: {code_index}, UT Index: {ut_index}")
                                log_file.write(f"No image for unit test for data index {dataset.data[idx]['index']}, Code Index: {code_index}, UT Index: {ut_index}\n")
                                continue
                            
                            res = code_execution_results[f"{idx}_{code_index}_{ut_index}_{im_idx}"]
                            penalty = get_penalty(res['error'], this_config['execution']['error_penalty']['syntax'], this_config['execution']['error_penalty']['runtime'])
                            if res['error'] is not None:
                                res['acc'] = -penalty
                            else:
                                if this_config['execution']['synonym_checker']:
                                    res['acc'] = checker.are_synonymous(res['output'], ut[1])
                                else:
                                    res['acc'] = acc_fn([res['output']], [ut[1]])
                            curr_unit_test.append(res)
                        if len(curr_unit_test) == 0:
                            print(f"No images for unit test {dataset.data[idx]['index']}-{code_index}-{ut_index}")
                            curr_unit_test = {'results': curr_unit_test, 'acc': 0.}
                        else:
                            curr_scores = [r['acc'] for r in curr_unit_test]
                            if this_config['execution']['image_agg'] == 'mean':
                                curr_unit_test = {'results': curr_unit_test, 'acc': sum(curr_scores)/len(curr_unit_test)}
                            elif this_config['execution']['image_agg'] == 'max':
                                curr_unit_test = {'results': curr_unit_test, 'acc': max(curr_scores)}
                            elif this_config['execution']['image_agg'] == 'majority':
                                acc_count = Counter(curr_scores)
                                curr_unit_test = {'results': curr_unit_test, 'acc': acc_count.most_common(1)[0][0]}
                        unit_test_results.append(curr_unit_test)
                    
                    curr_ut_scores = [r['acc'] for r in unit_test_results]
                    dataset.data[idx]['unit_test_results'][iter].append({
                        'results': unit_test_results, 
                        'acc': sum(curr_ut_scores)/len(unit_test_results)
                        })
            pickle.dump(dataset.data, open(save_file, 'wb'))

    print('Executing codes....')
    code_execution_data = [] #sample_id, answer, possible_answers, query_type, query, img, codes
    for idx in tqdm(range(0, len(dataset.data))):
        if iter > 0 and iter not in dataset.data[idx]['generated_code']:
            continue
        for iter, codes in dataset[idx]['generated_code'].items():
            for code_index, code in enumerate(codes):
                if 'code_outputs' in dataset.data[idx] and \
                    iter in dataset.data[idx]['code_outputs'] and \
                        code_index in dataset.data[idx]['code_outputs'][iter]:
                            continue
                generated_code = extract_python_code(code)
                if len(generated_code.split('def execute_command(image) -> str:\n')) > 1:
                    generated_code = generated_code.split('def execute_command(image) -> str:\n')[1]
                elif len(generated_code.split('def execute_command(image):\n')) > 1:
                    generated_code = generated_code.split('def execute_command(image):\n')[1]
                code_execution_data.append({'sample_id': f"{idx}_{iter}_{code_index}", 
                                            'answer': dataset.data[idx]['answer'], 
                                            'possible_answers': [dataset.data[idx]['answer']], 
                                            'query_type': 'image',
                                            'codes': generated_code,
                                            'query':dataset.data[idx]['question'], 
                                            'image': dataset[idx]['image_path']
                                            })
    if len(code_execution_data) > 0:
        code_execution_results = code_execution(code_execution_data, this_config, use_fixed_code=False) 
    correct = 0
    for idx in tqdm(range(len(dataset))):
        dataset.data[idx]['max_accuracy_output'] = []
        dataset.data[idx]['max_accuracy_code'] = []
        dataset.data[idx]['max_accuracy_iter'] = []
        prev_max = -10e10 # large negative integer
        answers = [dataset.data[idx]['answer']]
        if 'code_outputs' not in dataset.data[idx]:
            dataset.data[idx]['code_outputs'] = {} 
        if iter in dataset.data[idx]['generated_code']:
            codes = dataset.data[idx]['generated_code'][iter]
        else: # get last iter
            codes = dataset.data[idx]['generated_code'][max(list(dataset.data[idx]['generated_code'].keys()))]
        if iter not in dataset.data[idx]['code_outputs']:
            dataset.data[idx]['code_outputs'][iter] = {}
        for code_index,code in enumerate(codes):
            if code_index in dataset.data[idx]['code_outputs'][iter]:
                res = dataset.data[idx]['code_outputs'][iter][code_index]
            else:
                res = code_execution_results[f"{idx}_{iter}_{code_index}"]
                dataset.data[idx]['code_outputs'][iter][code_index] = res
            res['acc'] = acc_fn([res['output']], answers)
            dataset.data[idx]['output'] = res
            if res['acc'] > prev_max:
                dataset.data[idx]['max_accuracy_output'] = [res]
                dataset.data[idx]['max_accuracy_code'] = [code]
                dataset.data[idx]['max_accuracy_iter'] = [iter]
                prev_max = res['acc']
            elif res['acc'] == prev_max:
                dataset.data[idx]['max_accuracy_output'].append(res)
                dataset.data[idx]['max_accuracy_code'].append(code)
                dataset.data[idx]['max_accuracy_iter'].append(iter) 
        
        if len(dataset.data[idx]['max_accuracy_output'])==0:
            continue
        if this_config['execution']['code_agg'] == 'max' or this_config['execution']['code_agg'] == 'single':
            acc = max([r['acc'] if r['error'] is None else 0 for r in dataset.data[idx]['max_accuracy_output']])
        elif this_config['execution']['code_agg'] == 'majority' or this_config['execution']['code_agg'] == 'vote':
            outputs = [r['output'] if r['error'] is None else 'error' for r in dataset.data[idx]['max_accuracy_output']]
            most_common_output = Counter(outputs).most_common(1)[0][0]
            acc = acc_fn([most_common_output], answers)
        else:
            acc = dataset.data[idx]['max_accuracy_output'][0]['acc'] \
                if dataset.data[idx]['max_accuracy_output'][0]['error'] is None else 0
        dataset.data[idx]['final_accuracy'] = acc
        correct+=acc
    pickle.dump(dataset.data, open(save_file, 'wb'))
    print("Training Iteration accuracy: ", correct/len(dataset.data))
    log_file.write(f"Training Iteration accuracy: {correct/len(dataset.data)}\n")
    ## training
    print('Training....')
    formatted_inputs = []
    for d in dataset.data:
        for code_index,code in enumerate(d['generated_code'][iter]):
            if this_config['do_unit_test']:
                formatted_inputs.append({'question': d['question'], 'pred_code': extract_python_code(code), 'unit_test_results': d['unit_test_results'][iter][code_index], 'true_accuracy': float(d['code_outputs'][iter][code_index]['acc'] if d['code_outputs'][iter][code_index]['error'] is None else 0)})
            else:
                formatted_inputs.append({'question': d['question'], 'pred_code': extract_python_code(code), 'unit_test_results': {'acc': float(d['code_outputs'][iter][code_index]['acc'])}, 'true_accuracy':float(d['code_outputs'][iter][code_index]['acc'] if d['code_outputs'][iter][code_index]['error'] is None else 0)})
    pickle.dump(formatted_inputs, open(os.path.join(this_config['output_dir'], 'engine_input.pkl'), 'wb'))
    command = [train_path,
                '--model_name', this_config['visual_program_generator']['model_name'],
                '--train_kwargs', f"'{json.dumps(OmegaConf.to_container(this_config['train']['train_kwargs'], resolve=True))}'", 
                '--seed', this_config['seed'],
                '--iter', iter,
                '--batch_size', this_config['train']['batch_size'],
                '--data_path', os.path.join(this_config['output_dir'], 'engine_input.pkl'),
                '--output_dir', this_config['output_dir'],
                '--train_prompt_file', this_config['train']['train_prompt_file'],
                '--num_improve_steps', this_config['train']['num_improve_steps']
                ]
    os.system(" ".join([str(c) for c in command]))
    os.system('rm ' + os.path.join(this_config['output_dir'], 'engine_input.pkl'))
    
    
log_file.close()
