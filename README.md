# <img src="assets/viunit_logo.png" alt="Logo" height="100"> <br> Visual Unsupervised Unit Testing


## Framework Overview
<img src="assets/ViUnit Demo.png"/>

## Installation

```
conda create -n viunit_env python=3.10
conda activate viunit_env

conda install -c conda-forge gcc_linux-64=9.5.0
conda install -c conda-forge compilers
conda install -c conda-forge glib

export CC=gcc
export CXX=g++

conda install nvidia/label/cuda-12.1.0::cuda-toolkit

export PATH=$CONDA_PREFIX/bin:$PATH
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
alias ld="ld -L $LD_LIBRARY_PATH"

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

conda install pip
unset PYTHONPATH
python -m pip install -r requirements.txt

bash download_models.sh
```

:exclamation::exclamation: Ensure that the `cudatoolkit` version matches the one used to compile `pytorch` otherwise `GroundingDino`'s kernel for multiscale deformable attention will produce an error. 


## Datasets
### GQA
Download the GQA images from [here](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip). 
Set the environment variable `GQA_IMAGE_PATH` to the unzipped directory of the images. 
### WinoGround
Download the WinoGround images from [here](https://huggingface.co/datasets/facebook/winoground). 
Set the environment variable `WINOGROUND_IMAGE_PATH` to the image path. The annotation file should be placed under `my_datasets/annotation_files/WinoGround`.


### SugarCREPE
Download the [COCO-2017](https://cocodataset.org/#download) validation split used in [SugarCREPE](https://github.com/RAIVNLab/sugar-crepe).
Set the environment variable `COCO_VAL2017` to the image path.  

## Configs
Refer to [ViperGPT](https://github.com/cvlab-columbia/viper) for information about the API configs in `viper_configs/`.

`ViUniT` configs are found `/viunit_configs`.

### Config Parameters

#### **General Settings**

- **`seed`**  
  Sets the random seed for reproducibility.

- **`output_dir`**  
  Specifies the directory where the output files, including models and logs, will be saved.

- **`llm_engine`**  
  Defines the language model engine to be used. Options are `"hf"` (Hugging Face) or `"vllm"`. VLLM is about 25x faster, and is used in reported experiments. 

- **`do_unit_test`**  
  Specifies whether to perform unit testing.

---

#### **Data Settings**

- **`task`**  
  Used to define the task in question. Options: `VQA` (Visual Question Answering), `ITM` (Image-Text Matching)

- **`dataset_name`**  
  Specifies the dataset to be used. Options: `GQA`, `WinoGround`, `SugarCREPE`

- **`annotation_file`**  
  Path to the annotation file for the dataset. For training use the train split annotation file. 

- **`image_root`**  
  Defines the root directory for images associated with the dataset.

- **`sample_ids_file: null`**  
  If specified, this file would contain sample IDs for selective data loading. Set to `null` by default. File should consist of sample_ids split by `\n`.

- **`num_samples`**  
  Limits the number of samples to be loaded for the task. Randomly selected

- **`sample_per_type`**  
  Specifies the number of samples per type. `-1` indicates no limit. For GQA type is the question `group`, for WinoGround the instance `tag`, and for SugarCREPE the `question_type`.

---

#### **Training Settings**

- **`train_prompt_file`**  
  Path to the training prompt file. Does not include the full API. 

- **`num_improve_steps`**  
  Specifies the number of improvement steps for training. Namely, at each policy iteration how many times to iterate over the dataset. 

- **`batch_size`**  
  Defines the *per device* batch size for training.

- **`train_max_iters`**  
  Sets the number of iterations for training.

- **`train_kwargs`**  
  Additional training arguments including:
  - **`only_correct: True`**: Focuses training on correct responses on the training datasets. 
  - **`unit_test_reward: 'all'`**: Specifies reward calculation during unit testing. Can be one of `all`, `ratio`, `threshold`. All only accepts examples that pass all unit tests (`reward`=1), `ratio` sets reward equal to unit test score and considers all examples, unless `only_correct` is set, and `threshold` sets `reward`=1 to all examples that pass the threshold score on the unit tests. 
  - **`unit_test_reward_weight`**: Specifies a weight to apply to the reward such that `reward*=unit_test_reward_weight`.  
  - **`threshold`**: Threshold value for `unit_test_reward`=`'threshold'`
  - **`warmup_ratio`**: Warmup period ratio for learning rate scheduling.
  - **`max_grad_norm`**: Limits gradient norm to prevent exploding gradients.
  - **`lr_scheduler_type`**: Specifies the learning rate scheduler type.
  - **`learning_rate`**: Sets the learning rate.
  - **`lora_config`**: Configuration for Low-Rank Adaptation (LoRA) with parameters like `r`, `lora_alpha`, `lora_dropout`, and `target_modules`.
---

#### **Visual Program Generation**

- **`model_name`**  
  Specifies the model used for generating visual programs.

- **`half`**  
  Uses half-precision to save memory.

- **`model_kwargs`**  
  Additional model arguments, such as loading in 4-bit precision.

- **`generation`**  
  Configuration for generating visual programs, including prompt files, in-context examples, and generation parameters like `top_p`, `temperature`, `max_new_tokens`, and sampling strategies.
  - `num_return_sequences` specifies number of programs to generate!

---

#### **Unit Test Generation**

- **`model_name`**  
  Defines the model used for generating unit tests.

- **`half`**  
  Uses half-precision for efficiency.

- **`use_program`**  
  Indicates whether to condition unit test generation on program implementation. 

- **`generation`**  
  Contains generation settings similar to those in visual program generation but tailored for unit testing.

---

#### **Image Generation**

- **`image_scorer`**  
  Specifies whether to use an image scorer.

- **`image_source`**  
  Defines the source for image generation, with options like `'web'`, `'cc12m'`, and `'diffusion'`.

- **`top_k`**  
  Limits the number of images to be considered for rerakning in `'web'` and `'cc12m'`

- **`return_k`**  
  Number of images to be returned for each example.

- **`reranking_model_name`**  
  Specifies the model used for reranking generated images.

- **`generation`**  
  Details the parameters for generating images, including diffusion model settings, inference steps, and prompt configurations.

---

#### **Unit Test Sampling**

- **`strategy`**  
  Defines the strategy for sampling unit tests. Options: `coverage_by_answer` (maximize different answers, then fill rest of options by maximizing difference in inputs), `coverage` (maximize input difference), `random`, `answer_only` (maximize different asnwers, then randomly fill rest of spots).

- **`model_name`**  
  Model used for encoding descriptions for coverage sampling.

- **`filter_long_answers`**  
  Filters out unit tests with long answers (more than 5 words).

- **`num_unit_tests`**  
  Specifies the number of unit tests to generate.

---

#### **Execution Settings**

- **`pass_threshold`**  
  Sets the threshold for passing execution criteria for unit testing.

- **`feedback_max_iters`**  
  Maximum iterations for feedback loops (reprompting for correction). If set at 1 no feedback is conducted.

- **`image_agg`**  
  Aggregation method for images, with options like `'max'` (if at least one passes the unit test it is correct), `'mean'` (average across unit tests), or `'majority'` (if most images pass, then unit test passes).

- **`code_agg`**  
  Aggregation method for code execution, with options like `'single'` (randomly break ties of top scoring programs), `'vote'` (majority output of top voting programs).

- **`use_fixed_code_when_low_confidence`**  
  Option to use fixed code when confidence is low for answer refusal experiments.

- **`synonym_checker`**  
  Option to enable a synonym checker for unit testing instead of accuracy function.

- **`error_penalty`**  
  Penalties for errors in syntax and runtime. Default to 0.


## Run Inference
`
python generate.py --config_path /path/to/config
`
There are additional arguments that can be passed in the `generate.py` file that help with loading cached results. 
```
--data_path          Specify the path to a pickle file containing all data necessary for the generation. This option facilitates replicating previous runs.

--unit_test_path     Indicate the path to a pickle file containing unit tests from a different run that you wish to copy.

--load_images        Set to 0 or 1 to determine whether to load images from the unit tests. The default value is 1.

--load_selected      Set to 0 or 1 to decide whether to load sampled unit tests as they are, or to perform resampling. The default value is 1.

--execute_unit_tests Set to 0 or 1 to control whether to re-execute the unit tests. The default setting is 0.

--recompute_unit_tests Enable this option to recompute the accuracy of code on unit tests without re-execution. This is useful for applying thresholds. Default is set to 0.

--reload             Set to 0 or 1 to decide if stored data in the output directory should be reloaded. The default setting is 0.

--lora_path          Specify this option to use a Lora path for the visual program generator.
```


## Run Train
`
python train.py --config_path /path/to/config
`
There are additional arguments that can be passed in the `train.py` file that help with loading cached results. 
```
--data_path          Specify the path to a pickle file containing all data necessary for the generation. This option facilitates replicating previous runs.

--unit_test_path     Indicate the path to a pickle file containing unit tests from a different run that you wish to copy.

--load_images        Set to 0 or 1 to determine whether to load images from the unit tests. The default value is 1.

--load_selected      Set to 0 or 1 to decide whether to load sampled unit tests as they are, or to perform resampling. The default value is 1.

--execute_unit_tests Set to 0 or 1 to control whether to re-execute the unit tests. The default setting is 0.

--recompute_unit_tests Enable this option to recompute the accuracy of code on unit tests without re-execution. This is useful for applying thresholds. Default is set to 0.

--reload             Set to 0 or 1 to decide if stored data in the output directory should be reloaded. The default setting is 0.

--lora_path          Specify this option to use a Lora path for the visual program generator.
```

## Demo
An `ipython` demo file is provided in `demo.ipynb`

## Disclaimer
* Users need to make their own assessment regarding any obligations or responsibilities under the corresponding licenses or terms and conditions pertaining to the original datasets and data. This repository is being released for research purposes only. 
* The Meta WinoGround dataset is linked out to and not included in the repo. 
* The GQA dataset is released by Standford and the origin of the data is not described - the CC-BY 4.0 logo is simply listed at the bottom of the dataset download page. The dataset is based on the image-text pairs of COCO (https://cocodataset.org/#termsofuse), which follows flickr TOU (https://www.flickr.com/creativecommons/) having a variety of licenses.



## Acknowledgements
API code adapted from [ViperGPT](https://github.com/cvlab-columbia/viper)
