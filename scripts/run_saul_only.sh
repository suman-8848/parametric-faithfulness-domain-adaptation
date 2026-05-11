#!/bin/bash
#SBATCH -J saul_legal
#SBATCH -A c02130
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH -o logs/saul_%j.out
#SBATCH -e logs/saul_%j.err

echo "=========================================="
echo "Saul-7B Legal Faithfulness"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Load modules
module load python/3.11.5
module load cuda/12.2.0

# Create directories
mkdir -p logs
mkdir -p results/legalbench/saul_7b_instruct_v1

# Activate virtual environment (must be set up first with setup_env.sh)
source venv/bin/activate

# Add parametric-faithfulness to Python path
export PYTHONPATH="${PYTHONPATH}:${PWD}/parametric-faithfulness"

# Run Saul-7B
echo ""
echo "=========================================="
echo "Running Saul-7B-Instruct-v1..."
echo "=========================================="

python legal_experiment.py \
    --model_name Equall/Saul-7B-Instruct-v1 \
    --dataset legalbench \
    --method npo_KL \
    --strategy sentencize \
    --stepwise \
    --epochs 5 \
    --lr 5e-06 \
    --seed 1001 \
    --pos \
    --ff2 \
    --max_instances 30

echo ""
echo "=========================================="
echo "Saul-7B completed at: $(date)"
echo "=========================================="
