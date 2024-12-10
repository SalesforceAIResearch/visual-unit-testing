"""
 Copyright (c) 2024, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import os
import torch
from datasets import Dataset
import pandas as pd
from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer,
                          BitsAndBytesConfig,
)
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
import pickle
import json
import argparse
from accelerate import Accelerator
from accelerate.utils import gather_object

import time
import math

accelerator = Accelerator()


                             
parser = argparse.ArgumentParser(
    description='Generate llm outputs')
parser.add_argument('--model_name', type=str, default='', help="huggingface model name to use")
parser.add_argument('--half', type=bool, default=False, help="use float 16")
parser.add_argument('--generation_kwargs', type=str, default="{}", help="generation kwargs")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--model_kwargs', type=str, default="{}", help="model kwargs")
parser.add_argument('--lora_path', type=str, default=None, help="path to lora adapter")                        
parser.add_argument('--batch_size', type=int, default=None, help="per device batch size")
parser.add_argument('--data_path', type=str, default='', help='input data path')
parser.add_argument('--output_dir', type=str, default='.', help='output directory')
args = parser.parse_args()


input_dataset = pickle.load(open(os.path.join(args.data_path), 'rb'))

generation_kwargs = json.loads(args.generation_kwargs)
model_kwargs = json.loads(args.model_kwargs)


text_key = 'text'

def convert_to_hf_dataset(batch):
    if isinstance(batch, Dataset):
        # If already a Hugging Face dataset, return it as is
        hf_dataset = batch
    elif isinstance(batch, torch.utils.data.Dataset):
        # Convert torch Dataset to list of dictionaries (each dictionary representing a sample)
        # Assuming that the dataset returns a dictionary where keys are column names
        # and values are data points corresponding to the keys.
        batch_list = [batch[i] for i in range(len(batch))]
        hf_dataset = Dataset.from_dict({k: [dic[k] for dic in batch_list] for k in batch_list[0]})
    elif isinstance(batch, pd.DataFrame):
        # Convert directly from pandas DataFrame
        hf_dataset = Dataset.from_pandas(batch)
    elif isinstance(batch, dict):
        # Assume batch is a dictionary where each key has a list of items
        hf_dataset = Dataset.from_dict(batch)
    elif isinstance(batch, list):
        if len(batch)>0 and isinstance(batch[0], dict):
            hf_dataset = Dataset.from_dict({k: [dic[k] for dic in batch] for k in batch[0]})
        elif len(batch)>0 and isinstance(batch[0], str):
            hf_dataset = Dataset.from_dict({'text': batch})
    else:
        raise ValueError("Unsupported batch type for conversion.")
    return hf_dataset
    


dataset = convert_to_hf_dataset(input_dataset)
if text_key in dataset.column_names:
    input_dataset = KeyDataset(dataset, text_key)
else:
    input_dataset = dataset

tokenizer = AutoTokenizer.from_pretrained(args.model_name, 
                                          trust_remote_code=True, 
                                          use_fast=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side='left'
quantization_config = None
if model_kwargs:
    if model_kwargs.get('load_in_kbit', False):
        kbit  = model_kwargs.get('kbit', 8)
        quantization = {f'load_in_{kbit}bit': True,
                        f'bnb_{kbit}bit_quant_type':f"nf{kbit}",
                        f'bnb_{kbit}bit_use_double_quant':True,
                        f'bnb_{kbit}bit_compute_dtype':torch.float16 if half\
                            else torch.float32
                        }
        quantization_config = BitsAndBytesConfig(**quantization)


model = AutoModelForCausalLM.from_pretrained(args.model_name, 
                                            quantization_config=quantization_config,
                                            torch_dtype=torch.float16 if args.half else torch.float32,
                                            device_map={"": accelerator.process_index},
                                            attn_implementation="flash_attention_2",
                                            trust_remote_code=True)

if isinstance(args.lora_path, str):
    model.load_adapter(args.lora_path)


if 'max_length' in generation_kwargs:
    max_length = generation_kwargs['max_length']
    del generation_kwargs['max_length']
else:
    max_length = model.config.max_length
# Pre-tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length",max_length=max_length, truncation=True)

batch_size = int(args.batch_size*torch.cuda.device_count()) if args.batch_size else len(input_dataset['text'])

dataset = dataset.map(tokenize_function, batched=True)

dataset = dataset.remove_columns('text')
dataset.set_format(type='torch')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
# Check if CUDA is available and setup model for multi-GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if torch.cuda.device_count() > 1:
#     model = torch.nn.DataParallel(model)
# model.to(device)
if accelerator.is_main_process:
    pbar = tqdm(total=len(dataset))
else:
    pbar = None
   
# sync GPUs and start the timer
accelerator.wait_for_everyone()
start=time.time()
# divide the prompt list onto the available GPUs 
num_devices = torch.cuda.device_count()
max_iter = math.ceil(len(dataloader)//torch.cuda.device_count())*num_devices
with accelerator.split_between_processes(dataloader, apply_padding=True) as batches:
    # store output of generations in dict
    model.eval()
    predictions = dict(outputs=[], indices=[])
    
    with torch.no_grad():
        for batch in batches:
            predictions['indices'].extend(batch.pop('index'))
            batch = {k: v.to('cuda') for k, v in batch.items() if isinstance(v, torch.Tensor)}
            output = model.generate(**batch, **generation_kwargs)
            prediction = tokenizer.batch_decode(output[:,batch['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            predictions['outputs'].extend(prediction)
            if pbar:
                pbar.update(len(predictions)*num_devices) ## approximation
    predictions = [predictions]

# collect results from all the GPUs
accelerator.wait_for_everyone()
predictions_gathered=gather_object(predictions)
if accelerator.is_main_process:
    predictions_flattened = {'outputs':[], 'indices':[]}
    for pred in predictions_gathered:
        predictions_flattened['outputs'].extend(pred['outputs'])
        predictions_flattened['indices'].extend([i.item() for i in pred['indices']])

    results = []
    for i, (index,out) in enumerate(zip(predictions_flattened['indices'], predictions_flattened['outputs'])):
        if isinstance(out, str):
            out = [out]
        results.append({'generated_text': out, 'prompt': input_dataset[index], 'index': index})

    os.makedirs(os.path.join(args.output_dir, 'engine_output'), exist_ok=True)
    with open(os.path.join(args.output_dir, 'engine_output', 'results.jsonl'),'w') as f:
        for res in results:     
            f.write(json.dumps(res)+'\n')       