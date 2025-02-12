#!/bin/bash -l
#SBATCH --account=EUHPC_D18_005
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=01:00:00
#SBATCH --chdir=/leonardo_scratch/fast/EUHPC_D18_005/david/FastV/src/LLaVA
#SBATCH --output=/leonardo_scratch/fast/EUHPC_D18_005/david/outputs
#SBATCH --error=/leonardo_scratch/fast/EUHPC_D18_005/david/outputs/fastv.err

# See running jobs
# watch -n 1 squeue --me

# Interactive mode
# salloc --nodes=1 --gres=gpu:1 --ntasks-per-node=1 --account=EUHPC_D18_005 --partition=boost_usr_prod

# set HF home directory for offline datasets
export HF_HOME="/leonardo_scratch/fast/EUHPC_D18_005/david/hf-datasets-cache"

# activate conda env
conda deactivate
conda activate base

srun ./lmms-eval.sh