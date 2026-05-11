#!/bin/bash
#SBATCH --job-name=legal_faithfulness
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/legal_%A_%a.out
#SBATCH --error=logs/legal_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@iu.edu
#SBATCH -A r02144

# Array job: 0=Mistral, 1=Saul
#SBATCH --array=0-1

# Create logs directory
mkdir -p logs

# Load modules
module purge
module load python/3.10.5
module load cuda/11.8

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Set up Python environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

source venv/bin/activate

# Install dependencies (only once)
if [ ! -f "venv/.installed" ]; then
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install transformers accelerate datasets bitsandbytes
    pip install nltk spacy scipy matplotlib seaborn
    python -m spacy download en_core_web_sm
    python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
    touch venv/.installed
    echo "Dependencies installed."
fi

# Clone repository if needed
REPO_DIR="parametric-faithfulness"
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning repository..."
    git clone https://github.com/technion-cs-nlp/parametric-faithfulness.git $REPO_DIR
fi

# Set model based on array task ID
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2"
    MODEL_SHORT="mistral_7b_instruct_v0.2"
    echo "Running MISTRAL-7B (baseline)"
elif [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    MODEL_NAME="Equall/Saul-7B-Instruct-v1"
    MODEL_SHORT="saul_7b_instruct_v1"
    echo "Running SAUL-7B (legal domain)"
fi

# Set HuggingFace token (IMPORTANT: Set this as environment variable or in ~/.bashrc)
# export HF_TOKEN="your_token_here"
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set. Please set it as environment variable."
    exit 1
fi

# Run the experiment
echo "Starting experiment for $MODEL_NAME..."
python legal_experiment.py \
    --model_name "$MODEL_NAME" \
    --dataset legalbench \
    --method npo_KL \
    --strategy sentencize \
    --stepwise \
    --epochs 5 \
    --lr 3e-05 \
    --seed 1001 \
    --pos \
    --ff2 \
    --max_instances 30 \
    --output_dir "results/legal/$MODEL_SHORT"

echo "Experiment completed for $MODEL_NAME"
