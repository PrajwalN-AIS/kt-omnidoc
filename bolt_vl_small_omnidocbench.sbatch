#!/bin/bash
#SBATCH --job-name=bolt_vl_small_omnidoc
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/project/LNUN5155_1759/personal_workspaces/SHETTYV3468/bolt_vl_small_omnidoc_%j.out
#SBATCH --error=/project/LNUN5155_1759/personal_workspaces/SHETTYV3468/bolt_vl_small_omnidoc_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=balan.mahesh@cgu.edu

module load gcc/13.3.0
module load conda
source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate /project/LNUN5155_1759/personal_workspaces/SHETTYV3468/envs/py311-bolt

export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/project/LNUN5155_1759/personal_workspaces/SHETTYV3468/.cache/huggingface

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"

cd /project/LNUN5155_1759/personal_workspaces/SHETTYV3468/OmniDocBench

echo "=== Step 1: Inference ==="
python bolt_vl_small_omnidocbench.py

echo "=== Step 2: Evaluation ==="
python pdf_validation.py --config configs/bolt_vl_small_end2end.yaml

echo "Job finished: $(date)"
