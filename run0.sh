#!/bin/bash -l
#SBATCH --account=EUHPC_D18_005
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=01:00:00
#SBATCH --chdir=/leonardo_scratch/fast/EUHPC_D18_005/david/FastV/src/LLaVA
#SBATCH --output=/leonardo_scratch/fast/EUHPC_D18_005/david/outputs

# activate conda env
conda activate base

srun ./lmms-eval.sh