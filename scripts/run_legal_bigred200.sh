#!/bin/bash
#SBATCH -J legal_faithfulness
#SBATCH -A c02130
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH -o logs/legal_%j.out
#SBATCH -e logs/legal_%j.err

# ============================================================
# Legal Domain Faithfulness Experiment on Big Red 200
# Run with: sbatch run_legal_bigred200.sh
# ============================================================

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Load modules
module load python/3.11.5
module load cuda/12.2.0

# Create logs directory
mkdir -p logs

# Set up Python environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate datasets bitsandbytes
pip install nltk spacy scipy
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Clone repository if needed
REPO_DIR="parametric-faithfulness"
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning repository..."
    git clone https://github.com/technion-cs-nlp/parametric-faithfulness.git $REPO_DIR
fi

# Set HuggingFace token (you'll need to set this)
# export HF_TOKEN="your_token_here"
# Or use: huggingface-cli login

# ============================================================
# Run Mistral-7B first
# ============================================================
echo ""
echo "=========================================="
echo "Running Mistral-7B-Instruct-v0.2..."
echo "=========================================="

python legal_experiment.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
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

echo "Mistral-7B completed at: $(date)"

# ============================================================
# Run Saul-7B second
# ============================================================
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

echo "Saul-7B completed at: $(date)"

# ============================================================
# Generate comparison analysis
# ============================================================
echo ""
echo "=========================================="
echo "Generating comparison analysis..."
echo "=========================================="

python verify_legal_results.py

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
