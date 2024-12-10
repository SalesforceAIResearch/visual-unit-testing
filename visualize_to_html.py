
"""
 Copyright (c) 2024, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import os
os.environ['PYTHONPATH'] = f"/nlpgpu/data/artemisp/visual_unit_testing:{os.environ['CONDA_PREFIX']}/"
os.environ['HF_HOME'] = '/nlpgpu/data/artemisp/.cache/huggingface'
os.environ['TORCH_HOME'] = '/nlpgpu/data/artemisp/visual_unit_testing/.cache/'
os.environ['HF_ACCESS_TOKEN'] = '<HF_TOKEN>'
os.environ['HF_TOKEN'] = '<HF_TOKEN>'
os.environ['CUDA_HOME'] = os.environ['CONDA_PREFIX']
os.environ['CONFIG_NAMES'] = 'demo_config'
os.environ["GQA_IMAGE_PATH"] = "/nlp/data/vision_datasets/GQA"
os.environ["WINOGROUND_IMAGE_PATH"] = "/nlp/data/vision_datasets/winoground/data/images"
os.environ["COCO_VAL2017"] = "/nlp/data/vision_datasets/winoground/data/images"
os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'
os.environ['CUDA_HOME'] = os.environ['CONDA_PREFIX']
os.environ['LD_LIBRARY_PATH'] = f"{os.environ['CONDA_PREFIX']}/lib"
os.environ['ld'] = f"ld -l {os.environ['LD_LIBRARY_PATH']}"

import os
from PIL import Image
import sys
import re
import pickle
import omegaconf
from unit_test_generation.processing import extract_unit_tests, get_unit_test_prompt, get_grounded_diffusion_prompt
from unit_test_generation.unit_test_sampling import TextSampler
from utils import exec_code, extract_python_code
from viper_configs import viper_config
from tqdm import tqdm
import ast
import copy
import random
from main_simple_lib import *
import argparse
from IPython.display import display
import io
import base64

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default=None)
parser.add_argument('--num_examples', type=int, default=10)
args = parser.parse_args()

image_root = os.getenv('WINOGROUND_IMAGE_PATH') if 'winoground' in args.data_path.lower() \
    else os.getenv('GQA_IMAGE_PATH') if 'gqa' in args.data_path.lower() else \
    os.getenv('COCO_VAL2017') if 'sugarcrepe' in args.data_path.lower() else None

def image_to_html(image_path):
    buffered = io.BytesIO()
    if isinstance(image_path, Image.Image):
        img = image_path
    elif 'http' in image_path:
        img = Image.open(requests.get(image_path, stream=True).raw).convert('RGB')
    else:   
        img = Image.open(image_path).convert('RGB')
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    img_html = f'<img src="data:image/jpeg;base64,{img_str}" width="128" height="128">'
    return img_html

def save_html(output_str, filename="output.html"):
    with open(filename, "w") as f:
        f.write(output_str)

def exec_code(image, code):
    code = extract_python_code(code)
    try:
        code = ast.unparse(ast.parse(code))
    except Exception as e:
        output_html  = f"<p style='color:red;'>Error: {e}</p>"
        return output_html

    syntax_2 = Syntax(code, "python", theme="light", line_numbers=True, start_line=0)
    
    code = code.replace("def execute_command(image)","def execute_command(image, my_fig, time_wait_between_lines, syntax)") 
    output_html =''

    if 'Issue' in code:
        output_html += f"<p style='color:red;'>Issue: {code.split('Issue:')[1]}</p>"
    
    output_html += '<hr>'
    
    if isinstance(image, Image.Image):
        img = image
    elif 'http' in image:
        img = Image.open(requests.get(image, stream=True).raw)
    else:
        img = Image.open(image)
    
    if img.mode != 'RGB':
        img = img.convert('RGB')

    try:
        output_html += execute_code_html((code, syntax_2), img, show_intermediate_steps=True)
        # Instezad of executing the code and displaying intermediate steps,
        # capture the output into HTML here (assume `execute_code` can be adjusted similarly)
        # output_html += "<p>Code executed successfully.</p>"
    except Exception as e:
        output_html += f"<p style='color:red;'>Error: {e}</p>"

    return output_html

def print_query(data, index):
    output_html = '<h3>Query Information</h3>'
    if 'index' in data[index]:
        assert index == data[index]['index']
    output_html += f"<p>Question: {data[index]['question']}</p>"
    output_html += f"<p>Answer: {data[index]['answer']}</p>"
    if os.path.exists(os.path.join(image_root, data[index]['image']+'.jpg')):
        image_path = os.path.join(image_root, data[index]['image']+'.jpg')
    elif os.path.exists(os.path.join(image_root, data[index]['image']+'.png')):
        image_path = os.path.join(image_root, data[index]['image']+'.png')
    
    output_html += image_to_html(image_path)
    return output_html

def print_unit_tests(data, index, exec=False):
    output_html = '<h3>Unit Tests</h3>'
    output_html += f"<p>Question: {data[index]['question']}</p>"

    if 'generated_code' in data[index] and 'selected_unit_tests' in data[index]:
        if isinstance(data[index]['selected_unit_tests'], dict):
            for iter in data[index]['generated_code']:
                output_html += f"<h4>Iteration: {iter}</h4>"
                if 'selected_unit_tests' in data[index]:
                    for i, code in enumerate(data[index]['generated_code'][iter]):
                        output_html += f"<pre>{extract_python_code(code)}</pre>"
                        ut = data[index]['selected_unit_tests'][iter][i]
                        if isinstance(ut, list):
                            for t, u in enumerate(ut):
                                output_html += f"<p>Unit test: {u}</p>"
                                for img_idx, img in zip(ut, data[index]['unit_test_images'][iter][i][t][1]):
                                    if isinstance(img, list):
                                        for im_idx, im in enumerate(img):
                                            output_html += image_to_html(im)
                                            if exec:
                                                output_html += exec_code(im, code)
                                            if 'unit_test_results' in data[index]:
                                                output_html += f"<p>Unit test acc: {data[index]['unit_test_results'][iter][i]['results'][t]['results'][im_idx]['acc']}</p>"
                                    else:
                                        output_html += image_to_html(img)
                                        if exec:
                                            output_html += exec_code(img, code)
                                        if 'unit_test_results' in data[index]:
                                            output_html += f"<p>Unit test acc: {data[index]['unit_test_results'][iter][i]['results'][t]['acc']}</p>"
                        else:
                            output_html += f"<p>Unit test: {ut}</p>"
                            for im_idx, img in enumerate(data[index]['unit_test_images'][iter][i]):
                                if isinstance(img[1], list):
                                    for im in img[1]:
                                        output_html += image_to_html(im)
                                        if exec:
                                            output_html += exec_code(im, code)
                                else:
                                    output_html += image_to_html(img[1])
                                    if exec:
                                        output_html += exec_code(img[1], code)
                                if 'unit_test_results' in data[index]:
                                    output_html += f"<p>Unit test acc: {data[index]['unit_test_results'][iter][i]['results'][t][im_idx]['acc']}</p>"
                            if 'unit_test_results' in data[index]:
                                output_html += f"<p>Unit test output: {data[index]['unit_test_results'][iter][i]['results'][t]}</p>"
                                output_html += f"<p>Unit test acc: {data[index]['unit_test_results'][iter][i]['results'][t]['acc']}</p>"
                            output_html += '<hr>'
        else:
            for iter in data[index]['generated_code']:
                for i, code in enumerate(data[index]['generated_code'][iter]):
                    output_html += f"<pre>{extract_python_code(code)}</pre>"
                    for t, ut in enumerate(data[index]['selected_unit_tests']):
                        output_html += f"<p>Unit test: {ut}</p>"
                        if isinstance(data[index]['unit_test_images'][t], tuple):
                            images = data[index]['unit_test_images'][t][1]
                        else:
                            images = [Image.open(im) if not isinstance(im, list) else [Image.open(imm) for imm in im] for im in data[index]['unit_test_images'][t]]
                        for img in images:
                            if isinstance(img, list):
                                for im in img:
                                    output_html += image_to_html(im)
                                    if exec:
                                        output_html += exec_code(im, code)
                                    if 'unit_test_results' in data[index]:
                                        output_html += f"<p>Unit test acc: {data[index]['unit_test_results'][iter][i]['results'][t][im_idx]['acc']}</p>"
                            else:
                                output_html += image_to_html(img)
                                if exec:
                                    output_html += exec_code(img, code)
                        if 'unit_test_results' in data[index]:
                            output_html += f"<p>Unit test output: {data[index]['unit_test_results'][iter][i]['results'][t]}</p>"
                            output_html += f"<p>Unit test acc: {data[index]['unit_test_results'][iter][i]['results'][t]['acc']}</p>"
                        output_html += '<hr>'
    else:
        output_html += "<p>No unit tests</p>"

    return output_html

def print_codes(data, index, exec=False):
    output_html = '<h3>Code Outputs</h3>'
    output_html += f"<p>Question: {data[index]['question']}</p>"
    output_html += f"<p>Answer: {data[index]['answer']}</p>"
    if 'final_accuracy' in data[index]:
        output_html += f"<p>Final Accuracy: {data[index]['final_accuracy']}</p>"
    
    if os.path.exists(os.path.join(image_root, data[index]['image']+'.jpg')):
        image_path = os.path.join(image_root, data[index]['image']+'.jpg')
    elif os.path.exists(os.path.join(image_root, data[index]['image']+'.png')):
        image_path = os.path.join(image_root, data[index]['image']+'.png')
    elif os.path.exists(os.path.join(image_root, data[index]['image'])):
        image_path = os.path.join(image_root, data[index]['image'])
    
    output_html += image_to_html(image_path)
    
    for iter in data[index]['generated_code']:
        output_html += f'<h4>Iteration: {iter}</h4>'
        for j, code in enumerate(data[index]['generated_code'][iter]):
            output_html += '<hr>'
            output_html += f"<pre>{extract_python_code(code)}</pre>"
            if exec:
                output_html += exec_code(image_path, code)
            output_html += '<hr>'
    
    return output_html

# Load the data
data = pickle.load(open(args.data_path, 'rb'))
if args.num_examples is None:
    args.num_examples = len(data)
if args.num_examples > len(data):
    args.num_examples = len(data)
sample_data = random.sample(data, args.num_examples)

html_content = ""
for index in range(len(sample_data)): 
    html_content += print_query(sample_data, index)
    html_content += print_codes(sample_data, index, exec=True)
    html_content += print_unit_tests(sample_data, index, exec=True)
    html_content += '-'*100 + '<br>'
# Save HTML to file
save_html(html_content, "unit_tests_output.html")

# To see the result, open `unit_tests_output.html` in a web browser
