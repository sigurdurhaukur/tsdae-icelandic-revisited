#!/bin/bash

#SBATCH --job-name=TSDAE2-ISL
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=1



# activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh

# Load the conda environment
conda activate tsdae2

# Change to the directory where the script is located
cd /home1/s5982960/projects/tsdae-icelandic-revisited
python3 train.py

