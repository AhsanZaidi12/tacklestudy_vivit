#!/bin/bash
#
# run_vivit_train_taguchi.sh
# Submit multiple SLURM jobs for k-fold cross-validation
# Each job: 3 runs × 5 folds = 15 experiments
#
# Usage: bash run_vivit_train_taguchi.sh [gpu_type] [partition] [account]
#

GPU_TYPE=${1:-"h100-80:1"}
PARTITION=${2:-"GPU-shared"}
ACCOUNT=${3:-"asc180003p"}

BASE_RUNS_DIR="/ocean/projects/asc180003p/szaidi/Tackle_Ablation/taguchi_runs"
RUNS_PER_JOB=3
NUM_FOLDS=5

echo "=========================================="
echo "ViViT Multi-Job Submission System"
echo "=========================================="
echo "  GPU       : $GPU_TYPE"
echo "  Partition : $PARTITION"
echo "  Account   : $ACCOUNT"
echo "  Base Dir  : $BASE_RUNS_DIR"
echo "  Runs/Job  : $RUNS_PER_JOB"
echo "  Folds/Run : $NUM_FOLDS"
echo "=========================================="

mkdir -p logs

cd "$BASE_RUNS_DIR" || exit 1

# Get run directories - INCLUDE run_0 and run_0_original, EXCLUDE files
RUN_DIRS=()
for dir in run_*; do
    if [ -d "$dir" ] && [[ ! "$dir" =~ catalog ]]; then
        RUN_DIRS+=("$dir")
    fi
done

IFS=$'\n' RUN_DIRS=($(sort <<<"${RUN_DIRS[*]}"))
unset IFS

if [ ${#RUN_DIRS[@]} -eq 0 ]; then
    echo "ERROR: No run directories found in $BASE_RUNS_DIR"
    exit 1
fi

echo "Found ${#RUN_DIRS[@]} runs: ${RUN_DIRS[@]}"
echo ""

NUM_JOBS=$(( (${#RUN_DIRS[@]} + RUNS_PER_JOB - 1) / RUNS_PER_JOB ))
TOTAL_EXPERIMENTS=$((${#RUN_DIRS[@]} * NUM_FOLDS))

echo "Will submit $NUM_JOBS jobs for $TOTAL_EXPERIMENTS experiments"
echo ""

for (( job_id=0; job_id<$NUM_JOBS; job_id++ )); do
    start_idx=$((job_id * RUNS_PER_JOB))
    end_idx=$((start_idx + RUNS_PER_JOB - 1))
    
    if [ $end_idx -ge ${#RUN_DIRS[@]} ]; then
        end_idx=$((${#RUN_DIRS[@]} - 1))
    fi
    
    job_runs=("${RUN_DIRS[@]:$start_idx:$RUNS_PER_JOB}")
    num_runs_in_job=${#job_runs[@]}
    experiments_in_job=$((num_runs_in_job * NUM_FOLDS))
    
    echo "----------------------------------------"
    echo "Job $((job_id + 1))/$NUM_JOBS: ${job_runs[@]}"
    echo "  ($experiments_in_job experiments)"
    echo "----------------------------------------"
    
    runs_str=$(IFS=,; echo "${job_runs[*]}")
    
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=vivit_j${job_id}
#SBATCH --partition=${PARTITION}
#SBATCH --account=${ACCOUNT}
#SBATCH --gres=gpu:${GPU_TYPE}
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/ocean/projects/asc180003p/szaidi/Tackle_Ablation/logs/vivit_job${job_id}_%j.out
#SBATCH --error=/ocean/projects/asc180003p/szaidi/Tackle_Ablation/logs/vivit_job${job_id}_%j.err

echo "=========================================="
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_NODELIST"
echo "Runs: ${job_runs[@]}"
echo "=========================================="

eval "\$(conda shell.bash hook)"
conda activate vivit_pyt

cd /ocean/projects/asc180003p/szaidi/Tackle_Ablation/

for fold_id in {0..4}; do
    echo ""
    echo "Starting fold_\${fold_id}"
    python vivit_train_taguchi.py --runs ${runs_str} --fold \${fold_id}
    echo "Completed fold_\${fold_id}"
done

echo "Job ${job_id} completed at \$(date)"
EOF

    echo "Submitted job $((job_id + 1))"
    sleep 1
done

echo ""
echo "=========================================="
echo "All $NUM_JOBS jobs submitted!"
echo "Total: $TOTAL_EXPERIMENTS experiments"
echo "=========================================="
echo "Monitor: squeue -u \$USER"
echo "Logs: logs/"
echo ""