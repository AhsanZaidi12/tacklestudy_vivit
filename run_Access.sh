#!/bin/bash
#
# run_vivit_submit.sh
# Usage: bash run_vivit_submit.sh [gpu_type] [partition] [account]
#
# Optional CLI overrides:
#   gpu_type  – e.g. a100:1          (default: h100-80:1)
#   partition – e.g. GPU             (default: GPU-shared)
#   account   – your Slurm account   (default: asc180003p)

GPU_TYPE=${1:-"h100-80:1"}
PARTITION=${2:-"GPU-shared"}
ACCOUNT=${3:-"asc180003p"}

echo "Submitting ViViT training job:"
echo "  GPU      : $GPU_TYPE"
echo "  Partition: $PARTITION"
echo "  Account  : $ACCOUNT"
echo "  Time     : 48h"
echo "  Memory   : 22G"
echo "--------------------------------"

# Ensure a logs directory exists
mkdir -p logs

# Submit one SBATCH job
sbatch <<EOF
#!/bin/bash
#---------------- SLURM DIRECTIVES ----------------#
#SBATCH --job-name=vivit_train
#SBATCH --partition=${PARTITION}
#SBATCH --account=${ACCOUNT}
#SBATCH --gres=gpu:${GPU_TYPE}
#SBATCH --mem=22G
#SBATCH --time=48:00:00
#SBATCH --output=logs/vivit_train_%j.out
#SBATCH --error=logs/vivit_train_%j.err
#--------------------------------------------------#

# Load Conda & activate env

eval "\$(conda shell.bash hook)"
conda activate vivit_pyt

# Go to project directory
cd /ocean/projects/asc180003p/szaidi/TackleStudy/trimmed_dataset/

# Run training once
python train_vivit.py
EOF
