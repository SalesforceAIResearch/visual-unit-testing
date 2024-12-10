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
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
import pickle
import json
import argparse
from accelerate import Accelerator
from accelerate.utils import gather_object
from accelerate import PartialState

import time
import math
from diffusers import StableDiffusionPipeline, DiffusionPipeline, AutoPipelineForText2Image
from diffusers import DPMSolverMultistepScheduler
from unit_test_generation.processing import get_phrase_indices
from utils import create_unique_file


accelerator = Accelerator()


                             
parser = argparse.ArgumentParser(
    description='Generate images using a diffusion model')
parser.add_argument('--model_name', type=str, default='', help="diffusion model name")
parser.add_argument('--return_k', type=int, default=1, help="return k images")
parser.add_argument('--guidance_scale', type=float, default=7.5, help="generation kwargs")
parser.add_argument('--num_inference_steps', type=int, default=28, help="number of inference steps")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--image_scale', type=float, default=0.7)
parser.add_argument('--gligen_scheduled_sampling_beta', type=float, default=0.5)
parser.add_argument('--return_image', type=int, default=1)
parser.add_argument('--output_dir', type=str, default='.', help='output directory')
parser.add_argument('--batch_size', type=int, default=None, help="Per device batch size")
parser.add_argument('--data_path', type=str, default='.', help='Input data path')
parser.add_argument('--max_retries', type=int, default=3)
args = parser.parse_args()

model_name = args.model_name
return_k = args.return_k
guidance_scale = args.guidance_scale
num_inference_steps = args.num_inference_steps
seed = args.seed
image_scale = args.image_scale
return_image = args.return_image
output_dir = args.output_dir
batch_size = args.batch_size
data_path = args.data_path
max_retries = args.max_retries
gligen_scheduled_sampling_beta = args.gligen_scheduled_sampling_beta


input_dataset = pickle.load(open(os.path.join(args.data_path), 'rb'))

os.makedirs(output_dir, exist_ok=True)  
os.makedirs(os.path.join(output_dir, 'result'), exist_ok=True)

if 'lmd' in model_name:
    batch_size = 1
    generator = DiffusionPipeline.from_pretrained(model_name, # "longlian/lmd_plus"
                                                            custom_pipeline="unit_test_generation/llm_grounded_diffusion",
                                                            # custom_pipeline='llm_grounded_diffusion',
                                                            # custom_revision="main",
                                                            use_onnx=False,
                                                            # load_connected_pipes=False,
                                                            variant="fp16", torch_dtype=torch.float16,
                                                            guidance_scale=guidance_scale, 
                                                            num_inference_steps=num_inference_steps)
    
elif 'stable-diffusion-3' in model_name:
    from diffusers import StableDiffusion3Pipeline

    generator = StableDiffusion3Pipeline.from_pretrained(model_name,
                                                    torch_dtype=torch.float16, 
                                                    variant="fp16", 
                                                    use_onnx=False,
                                                    guidance_scale=guidance_scale,
                                                    num_inference_steps=num_inference_steps,
                                                    
    )
elif 'xl' in model_name:
    generator = AutoPipelineForText2Image.from_pretrained(model_name,
                                                    torch_dtype=torch.float16, 
                                                    variant="fp16",
                                                    use_onnx=False, 
                                                    guidance_scale=guidance_scale,
                                                    use_safetensors=True,
                                                    num_inference_steps=num_inference_steps,
    )
elif 'gligen' in model_name.lower():
    from diffusers import StableDiffusionGLIGENPipeline
    from unit_test_generation.llm_grounded_diffusion.pipeline import LLMGroundedDiffusionPipeline
    batch_size = 1
    generator = StableDiffusionGLIGENPipeline.from_pretrained(model_name,
                                                    torch_dtype=torch.float16, 
                                                    variant="fp16", 
                                                     use_onnx=False, 
                                                    guidance_scale=guidance_scale,
                                                    num_inference_steps=num_inference_steps,
    )
    generator.parse_llm_response = LLMGroundedDiffusionPipeline.parse_llm_response
else:
        generator = DiffusionPipeline.from_pretrained(model_name,
                                                    torch_dtype=torch.float16, 
                                                    variant="fp16", 
                                                    use_onnx=False,
                                                    guidance_scale=guidance_scale,
                                                    num_inference_steps=num_inference_steps,
        )
    
batch_size = batch_size if batch_size else len(input_dataset)
    
distributed_state = PartialState()
generator.to(distributed_state.device)
progress = tqdm(total=len(input_dataset)//(distributed_state.num_processes*batch_size), disable=not distributed_state.is_local_main_process)  # only show progress on the main process
generator.set_progress_bar_config(leave=False)
generator.set_progress_bar_config(disable=True)
                                                  
            
    
def preprocess_llm_response(llm_response, caption):
    ## TODO: there is problem when passing batch
    images = []
    llm_responses_formatted = []
    for response in llm_response:
        if isinstance(response, list):
            curr_formatted_reponse = []
            for r in response:
                ## find reponse with all requirements
                if "background prompt" not in r.lower() or "negative prompt" not in r.lower() or "[]" in "\n".join(r.split('\n')[:3]):
                    continue
                lines = r.split('\n')
                curr_formatted_reponse.append('\n'.join(lines[:3]))
                break
            llm_responses_formatted.append(curr_formatted_reponse)
        else:
            lines = response.split('\n')
            llm_responses_formatted.append('\n'.join(lines[:3]))
    phrases, boxes, bg_prompt, neg_prompt, phrase_indices = [], [], [], [], []
    for r,response in enumerate(llm_responses_formatted):
        if isinstance(response, list):
            curr_phrases, curr_boxes, curr_bg_prompt, curr_neg_prompt, curr_phrase_indices = [], [], [], [], []
            for i in range(len(response)):
                try:
                    phrase, box, bg, neg = generator.parse_llm_response(response[i])
                except:
                    continue
                if 'lmd' in model_name:
                    try:
                        phrase_index = get_phrase_indices(generator,bg, phrase)
                    except:
                        continue
                else:
                    phrase_index = []
                
                curr_phrases.extend(phrase)
                curr_boxes.extend(box)
                curr_bg_prompt.extend(bg)
                curr_neg_prompt.extend(neg)
            if curr_boxes == []:
                curr_boxes = [[0, 512, 0, 512] ]
                curr_phrases = [caption[r]]
                curr_bg_prompt = "a photorealistic image"
                curr_neg_prompt = []
                if 'lmd' in model_name:
                    try:
                        phrase_index = get_phrase_indices(generator, curr_bg_prompt, curr_phrases)
                    except:
                        continue
                else:
                    phrase_index = []
            curr_phrase_indices.extend(phrase_index)
            phrases.append(curr_phrases)
            boxes.append(curr_boxes)
            bg_prompt.append(curr_bg_prompt)
            neg_prompt.append(" ".join(curr_neg_prompt))
            phrase_indices.append(curr_phrase_indices)
        else:
            try:
                phrase, box, bg, neg = generator.parse_llm_response(response)
            except:
                box = [[0, 512, 0, 512] ]
                phrase = [caption[r]]
                curr_bg_prompt = "a photorealistic image of "
                curr_neg_prompt = ""
            if 'lmd' in model_name:
                phrase_index = get_phrase_indices(generator, bg, phrase)
            else:
                phrase_index = []
            phrases.append(phrase)
            boxes.append(box)
            bg_prompt.append(bg)
            neg_prompt.append(neg)
            phrase_indices.append(phrase_index)
    return phrases, boxes, bg_prompt, neg_prompt, phrase_indices
                                                        
    
def batch_fetch_image(data):
    if len(data)>0 and 'llm_response' in data[0]:
        llm_response = [d['llm_response'] for d in data]
    else:
        llm_response = None
    queries = [d['query'] for d in data]
    indices = [d['index'] for d in data]
    
    queries = ["a photorealistic image of " + q.replace('"', '') for q in queries]
    if "lmd" in model_name:
        ## TODO: there is problem when passing batch
        images = []
        phrases, boxes, bg_prompt, neg_prompt, phrase_indices = preprocess_llm_response(llm_response, queries)
        for r, response in enumerate(llm_response):
            # Use `LLMGroundedDiffusionPipeline` to generate an image
            curr_seed = seed
            for attempt in range(max_retries):
                diffusion_output = generator(
                    prompt=queries[r],
                    negative_prompt=neg_prompt[r],
                    phrases=phrases[r],
                    boxes=boxes[r],
                    gligen_scheduled_sampling_beta=gligen_scheduled_sampling_beta,
                    output_type="pil",
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale, 
                    lmd_guidance_kwargs={},
                    phrase_indices = phrase_indices[r],
                    num_images_per_prompt=return_k, 
                    generator  = torch.Generator(device=generator.device).manual_seed(curr_seed)
                )
                if True in diffusion_output.nsfw_content_detected:
                    curr_seed += 1
                    print(f"NSFW content detected, retrying with seed {curr_seed}")
                    continue
                else:
                    images.extend(diffusion_output.images)
                    break
    elif 'gligen' in model_name.lower():
        phrases, boxes, bg_prompt, neg_prompt, phrase_indices = preprocess_llm_response(llm_response, queries)
        images = []
        for r, response in enumerate(llm_response):
            curr_seed = seed
            for attempt in range(max_retries):
                diffusion_output = generator(
                    prompt=queries[r],
                    gligen_phrases=phrases[r],
                    gligen_boxes=boxes[r],
                    gligen_scheduled_sampling_beta=gligen_scheduled_sampling_beta,
                    output_type="pil",
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator  = torch.Generator(device=generator.device).manual_seed(curr_seed),
                    height=512,
                    width=512)
                if True in diffusion_output.nsfw_content_detected:
                    curr_seed += 1
                    print(f"NSFW content detected, retrying with seed {curr_seed}")
                    continue
                else:
                    images.extend(diffusion_output.images)
                    break
    else:
        curr_seed = seed
        images = [None for _ in range(len(queries))]
        for attempt in range(max_retries):
            diffusion_output = generator(queries,  
                                num_images_per_prompt=return_k, 
                                guidance_scale=guidance_scale, 
                                inference_steps=num_inference_steps,
                                seed = curr_seed,
                                output_type='pil',
                                height=512,
                                width=512,
                                return_dict=True)
            if 'nsfw_content_detected' in diffusion_output and True in diffusion_output.nsfw_content_detected:
                curr_seed += 1
                for i, nsfw in enumerate(diffusion_output.nsfw_content_detected):
                    if nsfw:
                        print(f"NSFW content detected, retrying with seed {curr_seed} for query {queries[i]}")
                    else:
                        images[i] = diffusion_output.images[i] if images[i] is None else images[i]
                continue
            else:
                for i in range(len(queries)):
                    images[i] = diffusion_output.images[i] if images[i] is None else images[i]
                break
    
    output = []
    return_images = []
    paths = []
    images = [im.resize((int(im.width*image_scale), int(im.height*image_scale))) for im in images]
    count = 0
    for i in range(0,len(images), return_k):
        curr_paths = []
        for j in range(i, i+return_k):
            im_path = create_unique_file(output_dir, 'image', '.png')
            
            images[j].save(im_path)
            curr_paths.append(im_path)
        paths.append(curr_paths)
        return_images.append(images[i:i+return_k])
        if return_image:
            output.append((indices[count], paths[-1], return_images[-1]))
        else:
            output.append((indices[count], paths[-1]))
        count += 1
    return output

with distributed_state.split_between_processes(input_dataset) as queries:
    batch_queries = [queries[i:i+batch_size] for i in range(0, len(queries), batch_size)]
    for i in range(len(batch_queries)):
        curr_batch = batch_queries[i]
        result = batch_fetch_image(curr_batch)
        with open(os.path.join(output_dir, f'result/{distributed_state.process_index}_batch_{i}.json'), 'wb') as f:
            pickle.dump(result,f)
        progress.update(1)

    