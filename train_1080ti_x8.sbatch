#!/bin/bash
#SBATCH --job-name=esrgan-bc-1080ti-x8
#SBATCH --partition=gpu_devel
#SBATCH --qos=interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

set -e
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source .venv/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

nvidia-smi
exec python realesrgan/train.py -opt options/finetune_realesrgan_x8plus_bc_multiset.yml --auto_resume
