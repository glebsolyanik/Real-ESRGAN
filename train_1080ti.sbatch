#!/bin/bash
#SBATCH --job-name=esrgan-bc-1080ti
#SBATCH --partition=gpu_devel
#SBATCH --qos=interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

cd ~/Real-ESRGAN
source .venv/bin/activate
export PYTHONPATH=$HOME/Real-ESRGAN:$PYTHONPATH

nvidia-smi
# gpu_devel nodes: GTX 1080 Ti (11 GB) and 4 CPU -> batch 4, 3 data workers
python realesrgan/train.py -opt options/finetune_realesrgan_x4plus_bc_multiset.yml --auto_resume \
    --force_yml name=finetune_RealESRGANx4plus_BC-MultiSet_1080ti_50k \
                datasets:train:batch_size_per_gpu=4 \
                datasets:train:num_worker_per_gpu=3
