#!/bin/bash
#SBATCH --job-name=efficientnet_test
#SBATCH --output=/speed-scratch/ae_verma/test_output.log
#SBATCH --error=/speed-scratch/ae_verma/test_error.log
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

source /encs/pkg/anaconda3-2023.03/root/etc/profile.d/conda.sh
conda activate /speed-scratch/ae_verma/Jupyter/jupyter-env

python /speed-scratch/ae_verma/test.py
