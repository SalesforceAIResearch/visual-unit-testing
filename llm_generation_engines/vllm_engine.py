"""
This example shows how to use Ray Data for running offline batch inference
distributively on a multi-nodes cluster.

Learn more about Ray Data in https://docs.ray.io/en/latest/data/data.html
"""

from typing import Any, Dict, List

import numpy as np
import ray
from packaging.version import Version
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

assert Version(ray.__version__) >= Version(
    "2.22.0"), "Ray version must be at least 2.22.0"

import torch
import os
import pandas as pd
import datasets
import json
import argparse
import pickle



                             
parser = argparse.ArgumentParser(
    description='Generate llm outputs')
parser.add_argument('--model_name', type=str, default='', help="huggingface model name to use")
parser.add_argument('--half', type=bool, default=False, help="use float 16")
parser.add_argument('--generation_kwargs', type=str, default="{}", help="generation kwargs")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--model_kwargs', type=str, default="{}", help="model kwargs")
parser.add_argument('--lora_path', type=str, default=None, help="path to lora adapter")                        
parser.add_argument('--batch_size', type=float, default=None, help="per device batch size")
parser.add_argument('--data_path', type=str, default='', help='input data path')
parser.add_argument('--output_dir', type=str, default='.', help='output directory')
args = parser.parse_args()


args = parser.parse_args()

dataset = pickle.load(open(os.path.join(args.data_path), 'rb'))

from functools import partial   

num_gpus = torch.cuda.device_count()                       
num_cpus = torch.cuda.device_count() * 8

RAY_STOP = ray.shutdown
RAY_INIT = partial(ray.init, address='local', num_cpus=num_cpus, num_gpus=num_gpus, _temp_dir=os.path.join('/nlpgpu/data/artemisp/ray')) #f'ray start --head'

ctx = RAY_INIT()

with ctx:
    
    LORA_PATH = args.lora_path

    generation_kwargs = json.loads(args.generation_kwargs)
    if generation_kwargs['top_k'] == 0:
        generation_kwargs['top_k'] = -1


        
    SAMPLING_PARAMS = SamplingParams(
        temperature=generation_kwargs['temperature'], 
        top_p=generation_kwargs['top_p'], 
        top_k=generation_kwargs['top_k'],
        presence_penalty=0.0,
        frequency_penalty=0.0,
        spaces_between_special_tokens=False,
        truncate_prompt_tokens=None,
        min_tokens=10, 
        max_tokens=generation_kwargs['max_new_tokens'],
        best_of=generation_kwargs['num_return_sequences'], 
        n=generation_kwargs['num_return_sequences'], 
        use_beam_search=generation_kwargs['num_beams']>1,
        ) 

    print("Device count: ", torch.cuda.device_count())
    
    model=args.model_name
    tensor_parallel_size=torch.cuda.device_count()
    seed=args.seed
    half=args.half
    enable_lora=args.lora_path is not None
    num_instances=1

    # Create a class to do batch inference.
    class LLMPredictor:

        def __init__(self, 
                    ):
        
            # Create an LLM.
            self.llm = LLM(model=model,
                        tokenizer=model,
                            tensor_parallel_size=tensor_parallel_size,
                            trust_remote_code=True,
                            seed=seed,
                            dtype=torch.float16 if half else torch.float32, 
                            max_num_seqs=128, 
                            enable_lora=enable_lora,
                            distributed_executor_backend='ray',
                            enforce_eager=True,
                            # max_parallel_loading_workers=4
                            )
            # self.sampling_params = sampling_params
            # self.lora_path = None
            # self.text_key = 'text'

        def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
            # Generate texts from the prompts.
            # The output is a list of RequestOutput objects that contain the prompt,
            # generated text, and other information.
            if LORA_PATH:
                outputs = self.llm.generate(prompts=batch['text'], 
                                            sampling_params=SAMPLING_PARAMS, 
                                            lora_request=LoRARequest('rl-lora', 1, LORA_PATH),
                                            use_tqdm = False
                                            )

            else:
                outputs = self.llm.generate(batch['text'],
                                            SAMPLING_PARAMS,
                                            use_tqdm=False
                                            )
            
            prompt: List[str] = []
            generated_text: List[str] = []
            index: List[int] = []
            for i,output in enumerate(outputs):
                prompt.append(output.prompt)
                generated_text.append([o.text for o in output.outputs])
                index.append(batch['index'][i])
            return {
                "index": index,
                "prompt": prompt,
                "generated_text": generated_text,
            }
            


    devices = list(range(torch.cuda.device_count()))

    text_key = 'text'
    if isinstance(dataset, datasets.Dataset):
        ds = ray.data.from_datasets(dataset)
    elif isinstance(dataset, dict):
        ds = ray.data.from_items([{'text': d, 'index': i} for d,i in zip(dataset['text'], dataset['index'])])
    elif isinstance(dataset, torch.utils.data.Dataset):
        ds = ray.data.from_torch(dataset)
    elif isinstance(dataset, pd.DataFrame):
        ds = ray.data.from_pandas(dataset)
    else:
        ds = ray.data.from_items(dataset)
        

        
    def scheduling_strategy_fn():
        # One bundle per tensor parallel worker
        pg = ray.util.placement_group(
            [{
                "GPU": 1,
                "CPU": 1
            }] * tensor_parallel_size,
            strategy="STRICT_PACK",
        )
        return dict(scheduling_strategy=PlacementGroupSchedulingStrategy(
            pg, placement_group_capture_child_tasks=True))

    resources_kwarg: Dict[str, Any] = {}
    if len(devices) == 1:
        # For tensor_parallel_size == 1, we simply set num_gpus=1.
        resources_kwarg["num_gpus"] = 1
    else:
        # Otherwise, we have to set num_gpus=0 and provide
        # a function that will create a placement group for
        # each instance.
        resources_kwarg["num_gpus"] = 0
        resources_kwarg["ray_remote_args_fn"] = scheduling_strategy_fn

    # outputs = self.model(ds)
    ds = ds.map_batches(
        LLMPredictor,
        # Set the concurrency to the number of LLM instances.
        concurrency=num_instances,
        # Specify the batch size for inference.
        batch_size=int(args.batch_size*torch.cuda.device_count()) if args.batch_size else ds.count(),
        **resources_kwarg
    )

    ds.write_json(os.path.join(args.output_dir, 'engine_output'))
    RAY_STOP()
