#!/bin/bash
#SBATCH --job-name=esrgan-bc-a100
#SBATCH --partition=ais-gpu
#SBATCH --qos=interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

cd ~/Real-ESRGAN
source .venv/bin/activate
export PYTHONPATH=$HOME/Real-ESRGAN:$PYTHONPATH

nvidia-smi
python realesrgan/train.py -opt options/finetune_realesrgan_x4plus_bc_multiset.yml --auto_resume \
    --force_yml name=finetune_RealESRGANx4plus_BC-MultiSet_a100_50k \
                datasets:train:batch_size_per_gpu=12 \
                datasets:train:num_worker_per_gpu=6
