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

# See running jobs
# watch -n 1 squeue --me

# Interactive mode
# salloc --nodes=1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=8 --account=EUHPC_D18_005 --partition=boost_usr_prod

# Load Java for lmms-evals metric calculation
module load openjdk/11.0.20.1_1

# Initialize conda
source /leonardo/home/userexternal/dhauser0/miniconda3/etc/profile.d/conda.sh

# Activate conda environment
conda activate base

# Install accelerate, local lmms-evals, local transformers (if not already installed)
pip show accelerate || pip install accelerate
cd /leonardo_scratch/fast/EUHPC_D18_005/david/lmms-eval-leonardo
pip install -e .
cd /leonardo_scratch/fast/EUHPC_D18_005/david/FastV/src/transformers
pip install -e .

# Deactivate and activate conda so accelerate is correctly found
conda deactivate
conda activate base

# Set HF home directory for offline datasets
export HF_HOME="/leonardo_scratch/fast/EUHPC_D18_005/david/hf-datasets-cache"

# export HF_DATASETS_OFFLINE=1

# Change directory to make relative model path work
cd /leonardo_scratch/fast/EUHPC_D18_005/david/FastV/src/LLaVA

# Run the script
/leonardo_scratch/fast/EUHPC_D18_005/david/FastV/src/LLaVA/lmms-evals.sh