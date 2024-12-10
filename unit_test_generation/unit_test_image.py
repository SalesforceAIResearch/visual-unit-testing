
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import torch
from torchvision.transforms import ToTensor
import gc
import pickle
import os
import json
from tqdm import tqdm
from torch.nn import functional as F
from utils import create_unique_file, find_free_port

class WebRetriever:
    def __init__(self,
                 model_name='ViT-L/14@336px',
                 top_k=1, 
                 return_k=1, 
                 scorer=None, 
                 return_image=True,
                 ):
        self.search_url = "https://www.google.com/search?hl=en&tbm=isch&q="
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
        self.top_k = top_k
        self.return_k = return_k
        if self.top_k < self.return_k:
            print(f"Top k {self.top_k} is less than return k {self.return_k}. Setting top k to return k.")
            self.top_k = self.return_k
        if scorer:
            from vision_models import CLIPModel
            self.scorer = CLIPModel(version=model_name)
        else:
            self.scorer = None
        self.return_image = return_image

        
    def batch_fetch_image(self, queries):
        outputs = []
        for query in queries:
            outputs.append(self.fetch_image(query))
        return outputs
    
    def fetch_image(self,query):
        # Construct the Google Search URL
        search_url = self.search_url + query
        # Make an HTTP GET request
        response = requests.get(search_url, headers=self.headers, timeout=(3.05, 27))
        response.raise_for_status()
        
        # Parse the response content with BeautifulSoupz
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all <img> tags in the HTML and extract the source URLs
        images_src = soup.find_all('img')
        
        # get top k images
        image_urls = []
        images = []
        counter = 0
        for i in range(1, len(images_src)):  #The first image is typically a Google logo or irrelevant, so we use the second item
            try:
                url = images_src[i]['src']
                image_response = requests.get(url)
                image = Image.open(BytesIO(image_response.content))
                images.append(image)
                image_urls.append(url)
            except:
                continue
            counter += 1
            if counter >= self.top_k:
                break
        if self.scorer:
            sim = self.scorer.compare([ToTensor()(im) for im in images], query, return_scores=True)
            sorted_idx = torch.argsort(sim, descending=True)
            image_urls = [image_urls[i] for i in sorted_idx[:self.return_k]]
            images = [images[i] for i in sorted_idx[:self.return_k]]
            
            if self.return_image:
                return image_urls, [im for im in images]
            else:
                return image_urls[:self.return_k]
        
        if self.return_image:         
            return image_urls[:self.return_k], images[:self.return_k]
        else:
            return image_urls[:self.return_k]
    
    def clear_retriever(self):
        if self.scorer:
            del self.scorer
        gc.collect()
        torch.cuda.empty_cache()

class DiffusionRetriever:
    def __init__(self, model_name='stabilityai/stable-diffusion-3-medium-diffusers', 
                #  scorer=None, # TODO: potentially add rescoring by clip
                #  top_k=1,
                 return_k=1,
                 guidance_scale=7.5, 
                 num_inference_steps=28,
                 gligen_scheduled_sampling_beta=0.4,
                 seed=42,
                 image_scale=0.7,
                 return_image=True,
                 output_dir=None,
                 max_retries=10,
                 distributed=True,
                 batch_size=None):
        self.distributed = distributed
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.return_k = return_k
        self.seed = seed
        self.gligen_scheduled_sampling_beta = gligen_scheduled_sampling_beta
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.return_image = return_image
        self.max_retries = max_retries
        if distributed:
            self.output_dir = output_dir
            self.data_path = os.path.join(self.output_dir, 'input_data.pkl')
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'result'), exist_ok=True)
            self.command = [f"accelerate launch --main_process_port {find_free_port()} unit_test_generation/unit_test_diffusers_distributed.py",
                        f"--model_name {model_name}",
                        f"--return_k {return_k}",
                        f"--guidance_scale {guidance_scale}",
                        f"--num_inference_steps {num_inference_steps}",
                        f"--gligen_scheduled_sampling_beta {gligen_scheduled_sampling_beta}",
                        f"--seed {seed}",
                        f"--image_scale {image_scale}",
                        f"--return_image {int(return_image)}",
                        f"--output_dir {output_dir}",
                        f"--max_retries {max_retries}",
                        f"--batch_size {batch_size}", 
                        f"--data_path {self.data_path}"]
        else:
            from diffusers import StableDiffusionPipeline, DiffusionPipeline, AutoPipelineForText2Image
            from diffusers import DPMSolverMultistepScheduler
            from unit_test_generation.processing import get_phrase_indices
            self.model_name = model_name
            ##TODO: add support for https://github.com/hohonu-vicml/DirectedDiffusion/blob/master/bin/DDGradio.py

            if 'lmd' in model_name:
                self.generator = DiffusionPipeline.from_pretrained(model_name, # "longlian/lmd_plus"
                                                                        custom_pipeline="unit_test_generation/llm_grounded_diffusion",
                                                                        # custom_pipeline='llm_grounded_diffusion',
                                                                        # custom_revision="main",
                                                                        variant="fp16", torch_dtype=torch.float16,
                                                                        guidance_scale=guidance_scale, 
                                                                        num_inference_steps=num_inference_steps)
                self.generator.enable_model_cpu_offload()
                
            elif 'stable-diffusion-3' in model_name:
                from diffusers import StableDiffusion3Pipeline

                self.generator = StableDiffusion3Pipeline.from_pretrained(model_name,
                                                                torch_dtype=torch.float16, 
                                                                variant="fp16", 
                                                                guidance_scale=guidance_scale,
                                                                num_inference_steps=num_inference_steps,
                                                                device_map="balanced")
            elif 'xl' in model_name:
                self.generator = AutoPipelineForText2Image.from_pretrained(model_name,
                                                                torch_dtype=torch.float16, 
                                                                variant="fp16", 
                                                                guidance_scale=guidance_scale,
                                                                use_safetensors=True,
                                                                num_inference_steps=num_inference_steps,
                                                                device_map="balanced")
            elif 'gligen' in model_name.lower():
                from diffusers import StableDiffusionGLIGENPipeline
                from unit_test_generation.llm_grounded_diffusion.pipeline import LLMGroundedDiffusionPipeline
                self.generator = StableDiffusionGLIGENPipeline.from_pretrained(model_name,
                                                                torch_dtype=torch.float16, 
                                                                variant="fp16", 
                                                                guidance_scale=guidance_scale,
                                                                num_inference_steps=num_inference_steps,
                                                                device_map="balanced")
                self.generator.parse_llm_response = LLMGroundedDiffusionPipeline.parse_llm_response
            else:
                self.generator = DiffusionPipeline.from_pretrained(model_name,
                                                                torch_dtype=torch.float16, 
                                                                variant="fp16", 
                                                                guidance_scale=guidance_scale,
                                                                num_inference_steps=num_inference_steps,
                                                                device_map="balanced")
            self.image_scale = image_scale
            self.get_phrase_indices = get_phrase_indices
            
                
                                                            
                
            # self.generator.scheduler = DPMSolverMultistepScheduler.from_config(self.generator.scheduler.config)
            
            # self.guidance_scale = guidance_scale
            # self.num_inference_steps = num_inference_steps
            # self.return_k = return_k
            # self.seed = seed
            # self.get_phrase_indices = get_phrase_indices
            # self.gligen_scheduled_sampling_beta = gligen_scheduled_sampling_beta
            # self.output_dir = output_dir
            # os.makedirs(self.output_dir, exist_ok=True)
            # self.return_image = return_image
            # self.max_retries = max_retries
        
    def preprocess_llm_response(self, llm_response, caption):
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
                        phrase, box, bg, neg = self.generator.parse_llm_response(response[i])
                    except:
                        continue
                    if 'lmd' in self.model_name:
                        phrase_index = self.get_phrase_indices(self.generator,bg, phrase)
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
                    if 'lmd' in self.model_name:
                        phrase_index = self.get_phrase_indices(self.generator, curr_bg_prompt, curr_phrases)
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
                    phrase, box, bg, neg = self.generator.parse_llm_response(response)
                except:
                    box = [[0, 512, 0, 512] ]
                    phrase = [caption[r]]
                    curr_bg_prompt = "a photorealistic image of "
                    curr_neg_prompt = ""
                if 'lmd' in self.model_name:
                    phrase_index = self.get_phrase_indices(self.generator, bg, phrase)
                else:
                    phrase_index = []
                phrases.append(phrase)
                boxes.append(box)
                bg_prompt.append(bg)
                neg_prompt.append(neg)
                phrase_indices.append(phrase_index)
        return phrases, boxes, bg_prompt, neg_prompt, phrase_indices
                                                        
    
    def batch_fetch_image(self, queries, llm_response=None):
        if self.distributed:
            if llm_response is None:
                data = [{'query': q, 'index': i} for i, q in enumerate(queries)]
            else:
                data = [{'query': q, 'llm_response': llm_response[i], 'index': i} for i, q in enumerate(queries)]
            with open(self.data_path, 'wb') as f:
                pickle.dump(data, f)
            os.system(' '.join(self.command))
            return_data = []
            for file in os.listdir(os.path.join(self.output_dir, 'result')):
                return_data.extend(pickle.load(open(os.path.join(self.output_dir, 'result', file), 'rb')))
            if not self.return_image:
                return_data = {d[0]: d[1] for d in return_data}    
            else:
                return_data= {d[0]: d[1:] for d in return_data}
            return_data = [return_data[i] for i in range(len(queries))]
            os.system(f'rm -r {os.path.join(self.output_dir, "result")}')
            os.system(f'rm -r {self.data_path}')
            return return_data
        queries = ["a photorealistic image of " + q.replace('"', '') for q in queries]
        if "lmd" in self.model_name:
            ## TODO: there is problem when passing batch
            images = []
            phrases, boxes, bg_prompt, neg_prompt, phrase_indices = self.preprocess_llm_response(llm_response, queries)
            for r, response in enumerate(llm_response):
                # Use `LLMGroundedDiffusionPipeline` to generate an image
                seed = self.seed
                for attempt in range(self.max_retries):
                    diffusion_output = self.generator(
                        prompt=queries[r],
                        negative_prompt=neg_prompt[r],
                        phrases=phrases[r],
                        boxes=boxes[r],
                        gligen_scheduled_sampling_beta=self.gligen_scheduled_sampling_beta,
                        output_type="pil",
                        num_inference_steps=self.num_inference_steps,
                        guidance_scale=self.guidance_scale, 
                        lmd_guidance_kwargs={},
                        phrase_indices = phrase_indices[r],
                        num_images_per_prompt=self.return_k, 
                        generator  = torch.Generator(device=self.generator.device).manual_seed(seed)
                    )
                    if True in diffusion_output.nsfw_content_detected:
                        seed += 1
                        print(f"NSFW content detected, retrying with seed {seed}")
                        continue
                    else:
                        images.extend(diffusion_output.images)
                        break
        elif 'gligen' in self.model_name.lower():
            phrases, boxes, bg_prompt, neg_prompt, phrase_indices = self.preprocess_llm_response(llm_response, queries)
            images = []
            for r, response in enumerate(llm_response):
                seed = self.seed
                for attempt in range(self.max_retries):
                    diffusion_output = self.generator(
                        prompt=queries[r],
                        gligen_phrases=phrases[r],
                        gligen_boxes=boxes[r],
                        gligen_scheduled_sampling_beta=self.gligen_scheduled_sampling_beta,
                        output_type="pil",
                        num_inference_steps=self.num_inference_steps,
                        guidance_scale=self.guidance_scale,
                        generator  = torch.Generator(device=self.generator.device).manual_seed(seed),
                        height=512,
                        width=512)
                    if True in diffusion_output.nsfw_content_detected:
                        seed += 1
                        print(f"NSFW content detected, retrying with seed {seed}")
                        continue
                    else:
                        images.extend(diffusion_output.images)
                        break
        else:
            seed = self.seed
            images = [None for _ in range(len(queries))]
            for attempt in range(self.max_retries):
                diffusion_output = self.generator(queries,  
                                    num_images_per_prompt=self.return_k, 
                                    guidance_scale=self.guidance_scale, 
                                    inference_steps=self.num_inference_steps,
                                    seed = seed,
                                    output_type='pil',
                                    height=512,
                                    width=512,
                                    return_dict=True)
                if 'nsfw_content_detected' in diffusion_output and True in diffusion_output.nsfw_content_detected:
                    seed += 1
                    for i, nsfw in enumerate(diffusion_output.nsfw_content_detected):
                        if nsfw:
                            print(f"NSFW content detected, retrying with seed {seed} for query {queries[i]}")
                        else:
                            images[i] = diffusion_output.images[i] if images[i] is None else images[i]
                    continue
                else:
                    for i in range(len(queries)):
                        images[i] = diffusion_output.images[i] if images[i] is None else images[i]
                    break
        
        # if 'xl' in self.model_name:
        #     images = self.refiner(
        #             prompt=queries,
        #             num_images_per_prompt=self.return_k, 
        #             num_inference_steps=self.num_inference_steps,
        #             denoising_start=0.8, # TODO: add param to config
        #             image=images,
        #         ).images
        output = []
        return_images = []
        paths = []
        images = [im.resize((int(im.width*self.image_scale), int(im.height*self.image_scale))) for im in images]
        for i in range(0,len(images), self.return_k):
            curr_paths = []
            for j in range(i, i+self.return_k):
                im_path = create_unique_file(self.output_dir, 'image', '.png')
                
                images[j].save(im_path)
                curr_paths.append(im_path)
            paths.append(curr_paths)
            return_images.append(images[i:i+self.return_k])
            if self.return_image:
                output.append((paths[-1], return_images[-1]))
            else:
                output.append(paths[-1])
        return output


    def fetch_image(self, query, llm_response=None):
        if llm_response is None:
            return self.batch_fetch_image([query])
        else:
            return self.batch_fetch_image([query], [llm_response]) 

    def clear_retriever(self):
        if self.distributed:
            return
        del self.generator
        gc.collect()
        torch.cuda.empty_cache()
        return