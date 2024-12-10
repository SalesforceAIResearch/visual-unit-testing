import datetime
import math
import os
import pathlib
from functools import partial
import warnings
import traceback
import pickle

import torch.multiprocessing as mp
from joblib import Memory
from num2words import num2words
import numpy as np
from omegaconf import OmegaConf
from rich.console import Console
from torch.utils.data import DataLoader
from tqdm import tqdm

from viper_configs import viper_config
from viper_utils import seed_everything
import torch
from postproc_utils import general_postprocessing
from utils import load_image
import re
import gc
import psutil
import signal
import ast
import astunparse
import time

# See https://github.com/pytorch/pytorch/issues/11201, https://github.com/pytorch/pytorch/issues/973
# Not for dataloader, but for multiprocessing batches
mp.set_sharing_strategy('file_system')
queue_results = None

cache = Memory('cache/' if viper_config.use_cache else None, verbose=0)
runs_dict = {}
console = Console(highlight=False)

def my_collate(batch):
    # Avoid stacking images (different size). Return everything as a list
    to_return = {k: [d[k] for d in batch] for k in batch[0].keys()}
    return to_return

class MyDataset(torch.utils.data.Dataset):
        def __init__(self, results):
            self.results = results
        def __getitem__(self, idx):
            if isinstance(self.results[idx]['image'], list):
                self.results[idx]['image'] = self.results[idx]['image'][0]
            if isinstance(self.results[idx]['image'], str):
                self.results[idx]['image'] = load_image(self.results[idx]['image'].replace('/diffusion_unit_tests/', '/stable_diffusion_unit_tests/').replace('/GQA/data/images/', '/GQA/'))

            if self.results[idx]['image'].mode != 'RGB':
                self.results[idx]['image'] = self.results[idx]['image'].convert('RGB')
            return self.results[idx]
        def __len__(self):
            return len(self.results)

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Execution timed out!")

def run_program(parameters, queues_in_, input_type_, retrying=False, fixed_code=None, timeout_duration=120):
    from image_patch import ImagePatch, llm_query, best_image_match, distance, bool_to_yesno, coerce_to_numeric
    from video_segment import VideoSegment

    code, sample_id, image, possible_answers, query = parameters

    # Decide which code to use
    if retrying and fixed_code:
        code_to_use = fixed_code.format(query.replace('"', '\\"').replace("'", "\\'"))
    else:
        code_to_use = code

    # Define the function header dynamically
    function_name = f'execute_command_{sample_id}'
    code_header = (
        f'def {function_name}('
        f'{input_type_}, possible_answers, query, '
        'ImagePatch, VideoSegment, '
        'llm_query, bool_to_yesno, distance, best_image_match, coerce_to_numeric):\n'
    )
    code_full = code_header + code_to_use

    result = {"error": None, "output": None}

    # Function to compile and execute the code
    def compile_and_exec(code_str):
        local_namespace = {}
        try:
            code_parsed = ast.parse(code_str)
            code_unparsed = astunparse.unparse(code_parsed)
            exec(compile(code_unparsed, '<string>', 'exec'), globals(), local_namespace)
            return local_namespace
        except Exception as e:
            result["error"] = f"Compilation error: {str(e)}"
            return None

    # Attempt to compile the code
    local_namespace = compile_and_exec(code_full)
    if not local_namespace:
        # Compilation failed
        if not retrying and fixed_code:
            # Retry with fixed_code
            return run_program(parameters, queues_in_, input_type_, True, fixed_code, timeout_duration)
        else:
            return result, sample_id, code_full

    # Execute the function
    try:
        # Set the signal for timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_duration)  # Set the timeout duration in seconds

        # Prepare partial functions with queues
        queues = [queues_in_, queue_results]
        image_patch_partial = partial(ImagePatch, queues=queues)
        video_segment_partial = partial(VideoSegment, queues=queues)
        llm_query_partial = partial(llm_query, queues=queues)

        func = local_namespace[function_name]
        output = func(
            image, possible_answers, query,
            image_patch_partial, video_segment_partial,
            llm_query_partial, bool_to_yesno, distance, best_image_match, coerce_to_numeric
        )
        result['output'] = output
    except TimeoutException:
        result["error"] = f"Timed out after {timeout_duration} seconds."
        if not retrying and fixed_code:
            # Retry with fixed_code
            return run_program(parameters, queues_in_, input_type_, True, fixed_code, timeout_duration)
    except Exception as e:
        result['error'] = f"Execution error: {str(e)}"
        if not retrying and fixed_code:
            # Retry with fixed_code
            return run_program(parameters, queues_in_, input_type_, True, fixed_code, timeout_duration)
    finally:
        # Disable the alarm
        signal.alarm(0)

    # Post-processing of the output
    if result.get('output') is not None:
        result['output'] = general_postprocessing(result['output'])

    return result, sample_id, code_full

def run_program_wrapper(args):
    return run_program(*args)

def worker_init(queues_in_):
    global queue_results
    index_queue = mp.current_process()._identity[0] % len(queues_in_)
    queue_results = queues_in_[index_queue]


def main():
    mp.set_start_method('spawn', force=True)
    from vision_processes import queues_in, finish_all_consumers, forward, manager
    import argparse

    parser = argparse.ArgumentParser(description='Run the VLLM engine')
    parser.add_argument('--input_path', type=str, default='')
    parser.add_argument('--output_path', type=str, default='')
    parser.add_argument('--fixed_code', type=str, default=None)
    parser.add_argument('--input_type', type=str, default='image')
    parser.add_argument('--seed', type=int, default=42)
    parser_args = parser.parse_args()

    seed_everything(parser_args.seed)

    fixed_code = parser_args.fixed_code
    batch_size = viper_config.dataset.batch_size
    num_processes = min(batch_size, 50)
    timeout_duration = viper_config.timeout_duration  # Ensure this is defined in your config

    if viper_config.multiprocessing:
        queues_results = [manager.Queue() for _ in range(num_processes)]
    else:
        queues_results = [None for _ in range(num_processes)]

    if viper_config.clear_cache:
        cache.clear()
    viper_config.execute_code = True

    results = pickle.load(open(parser_args.input_path, 'rb'))
    
    dataloader = DataLoader(MyDataset(results), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, collate_fn=my_collate)
    input_type = parser_args.input_type

    results_dir = pathlib.Path(parser_args.output_path)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Conditional 'with' statement using mp.Pool or a dummy context manager
    with mp.Pool(processes=num_processes, initializer=worker_init, initargs=(queues_results,)) \
            if viper_config.multiprocessing else open(os.devnull, "w") as pool:
        try:
            n_batches = len(dataloader)
            pbar = tqdm(total=n_batches)  # Initialize tqdm progress bar

            for i, batch in enumerate(dataloader):
                codes = batch['codes']
                # Run the code
                if viper_config.execute_code:
                    results = []

                    if viper_config.multiprocessing:
                        # Use multiprocessing pool to execute tasks
                        tasks = []
                        for c, sample_id, img, possible_answers, query in zip(
                                codes, batch['sample_id'], batch['image'], batch['possible_answers'], batch['query']
                        ):
                            args = (
                                [c, sample_id, img, possible_answers, query],
                                queues_in,
                                input_type,
                                False,
                                fixed_code,
                                timeout_duration
                            )
                            task = pool.apply_async(run_program_wrapper, args=(args,))
                            tasks.append((task, sample_id, args, False))  # (task, sample_id, args, retrying)

                        # Collect results
                        for idx, (task, sample_id, args, retrying) in enumerate(tasks):
                            try:
                                result = task.get(timeout=timeout_duration + 5)  # Slight buffer over function timeout
                                results.append(result)
                            except mp.TimeoutError:
                                # Handle timeout
                                print(f"Sample {sample_id} timed out.")
                                if not retrying and fixed_code:
                                    # Retry with fixed_code
                                    args_list = list(args)
                                    args_list[0][0] = fixed_code.format(args_list[0][4].replace('"', '\\"').replace("'", "\\'"))
                                    args_list[3] = True  # Set retrying to True
                                    retry_task = pool.apply_async(run_program_wrapper, args=(args_list,))
                                    # Collect retry result
                                    try:
                                        retry_result = retry_task.get(timeout=timeout_duration + 5)
                                        retry_result[0]['error'] = f"Execution timed out after {timeout_duration} seconds."
                                        results.append(retry_result)
                                    except mp.TimeoutError:
                                        print(f"Sample {sample_id} timed out again on retry.")
                                        result = (
                                            {"error": f"Execution timed out after {timeout_duration} seconds.", "output": 'error'},
                                            sample_id,
                                            None
                                        )
                                        results.append(result)
                                    except Exception as e:
                                        print(f"Sample {sample_id} failed on retry with exception: {e}")
                                        result = ({"error": str(e), "output": 'error'}, sample_id, None)
                                        results.append(result)
                                else:
                                    result = (
                                        {"error": f"Execution timed out after {timeout_duration} seconds.", "output": 'error'},
                                        sample_id,
                                        None
                                    )
                                    results.append(result)
                            except Exception as e:
                                print(f"Sample {sample_id} failed with exception: {e}")
                                if not retrying and fixed_code:
                                    # Retry with fixed_code
                                    args_list[0][0] = fixed_code.format(args_list[0][4].replace('"', '\\"').replace("'", "\\'"))
                                    args_list = list(args)
                                    args_list[3] = True  # Set retrying to True
                                    retry_task = pool.apply_async(run_program_wrapper, args=(args_list,))
                                    # Collect retry result
                                    try:
                                        retry_result = retry_task.get(timeout=timeout_duration + 5)
                                        retry_result[0]['error'] = f"Execution failed with exception: {str(e)}"
                                        results.append(retry_result)
                                    except mp.TimeoutError:
                                        print(f"Sample {sample_id} timed out again on retry.")
                                        result = (
                                            {"error": f"Execution timed out after {timeout_duration} seconds.", "output": 'error'},
                                            sample_id,
                                            None
                                        )
                                        results.append(result)
                                    except Exception as e2:
                                        print(f"Sample {sample_id} failed on retry with exception: {e2}")
                                        result = ({"error": str(e2), "output": 'error'}, sample_id, None)
                                        results.append(result)
                                else:
                                    result = ({"error": str(e), "output": 'error'}, sample_id, None)
                                    results.append(result)
                    else:
                        # Execute tasks without multiprocessing
                        results = []
                        for c, sample_id, img, possible_answers, query in zip(
                                codes, batch['sample_id'], batch['image'], batch['possible_answers'], batch['query']
                        ):
                            args = (
                                [c, sample_id, img, possible_answers, query],
                                queues_in,
                                input_type,
                                False,
                                fixed_code,
                                timeout_duration
                            )
                            result = run_program_wrapper(args)
                            results.append(result)

                else:
                    results = [(None, c) for c in codes]
                    warnings.warn("Not executing code! This is only generating the code. We set the flag "
                                  "'execute_code' to False by default, because executing code generated by a language "
                                  "model can be dangerous. Set the flag 'execute_code' to True if you want to execute "
                                  "it.")

                sample_id2result = {r[1]: r for r in results}
                all_results = [sample_id2result[sample_id][0] for sample_id in batch['sample_id']]

                all_codes = batch['codes']
                all_ids = batch['sample_id']
                all_answers = batch['answer']
                all_possible_answers = batch['possible_answers']
                all_query_types = batch['query_type']
                all_queries = batch['query']
                output_results = []
                for r in range(len(all_results)):
                    output_results.append({'result': all_results[r], 'answer': all_answers[r], 'code': all_codes[r],
                                            'sample_id': all_ids[r], 'query': all_queries[r],
                                            'possible_answers': all_possible_answers[r]})
                filename = f'results_{i}_{mp.current_process().name}.pkl'
                with open(os.path.join(parser_args.output_path, filename), 'wb') as f:
                    pickle.dump(output_results, f)

                # Update progress bar
                pbar.update(1)

            pbar.close()

        except Exception as e:
            # Print full stack trace
            traceback.print_exc()
            console.print(f'Exception: {e}')
            console.print("Completing logging and exiting...")

        finally:
            finish_all_consumers()

if __name__ == '__main__':
    main()
