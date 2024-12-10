#!/bin/bash
#
#SBATCH --partition=p_nlp
#SBATCH --job-name=predictions
#SBATCH --output=/nlp/data/artemisp/vision_bias/logs/%x.%j.out
#SBATCH --error=/nlp/data/artemisp/vision_bias/logs/%x.%j.err
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --constraint=48GBgpu
#SBATCH --mem-per-cpu=14G
#SBATCH --cpus-per-task=8

# Initialize conda
__conda_setup="$('/nlp/data/artemisp/miniforge3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/nlp/data/artemisp/miniforge3/etc/profile.d/conda.sh" ]; then
        . "/nlp/data/artemisp/miniforge3/etc/profile.d/conda.sh"
    else
        export PATH="/nlp/data/artemisp/miniforge3/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate viunit_env


export PYTHONPATH=/nlpgpu/data/artemisp/visual_unit_testing:$CONDA_PREFIX

export HF_HOME=/nlpgpu/data/artemisp/.cache/huggingface
export TORCH_HOME=/nlpgpu/data/artemisp/visual_unit_testing/.cache/
export HF_ACCESS_TOKEN=<HF_TOKEN>
export HF_TOKEN=<HF_TOKEN>

export CC=gcc
export CXX=g++
export PATH=$CONDA_PREFIX/bin:$PATH
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
alias ld="ld -L $LD_LIBRARY_PATH"


# export TF_CPP_MIN_LOG_LEVEL=1

# export TORCH_CUDA_ARCH_LIST="8.6"

export GQA_IMAGE_PATH=/nlp/data/vision_datasets/GQA
export WINOGROUND_IMAGE_PATH=/nlp/data/vision_datasets/winoground/data/images
export COCO_VAL2017=/nlp/data/vision_datasets/coco/val2017_/val2017


cd /nlpgpu/data/artemisp/viunit/visual_unit_testing/

python generate.py --config_path ./viunit_configs/base.yaml --reload 0