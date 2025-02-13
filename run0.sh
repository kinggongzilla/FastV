#!/bin/bash -l
#SBATCH --account=EUHPC_D18_005
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=48:00:00
#SBATCH --chdir=/leonardo_scratch/fast/EUHPC_D18_005/david/FastV/src/LLaVA
#SBATCH --output=/leonardo_scratch/fast/EUHPC_D18_005/david/outputs/fastv.out
#SBATCH --error=/leonardo_scratch/fast/EUHPC_D18_005/david/outputs/fastv.err

# Initialize conda
source /leonardo/home/userexternal/dhauser0/miniconda3/etc/profile.d/conda.sh

# Activate conda environment
conda activate base

# Install accelerate (if not already installed)
pip show accelerate || pip install accelerate

# Deactivate and activate conda so accelerate is correctly found
conda deactivate
conda activate base

# Set HF home directory for offline datasets
export HF_HOME="/leonardo_scratch/fast/EUHPC_D18_005/david/hf-datasets-cache"

# Change directory to make relative model path work
cd /leonardo_scratch/fast/EUHPC_D18_005/david/FastV/src/LLaVA

# Run the script
/leonardo_scratch/fast/EUHPC_D18_005/david/FastV/src/LLaVA/lmms-evals.sh