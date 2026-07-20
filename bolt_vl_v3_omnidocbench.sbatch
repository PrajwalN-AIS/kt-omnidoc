#!/bin/bash
#SBATCH --job-name=bolt_v3_omnidocbench
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/project/LNUN5155_1759/personal_workspaces/SHETTYV3468/bolt_v3_omnidocbench_%j.out
#SBATCH --error=/project/LNUN5155_1759/personal_workspaces/SHETTYV3468/bolt_v3_omnidocbench_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=balan.mahesh@cgu.edu

module load gcc/13.3.0
module load conda
source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate /project/LNUN5155_1759/personal_workspaces/SHETTYV3468/envs/py311-bolt

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/project/LNUN5155_1759/personal_workspaces/SHETTYV3468/.cache/huggingface

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"

cd /project/LNUN5155_1759/personal_workspaces/SHETTYV3468/OmniDocBench

python bolt_vl_v3_omnidocbench.py

echo "Job finished: $(date)"
