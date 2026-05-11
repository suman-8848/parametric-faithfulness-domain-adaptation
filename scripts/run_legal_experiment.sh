#!/bin/bash
#SBATCH -J legal_faithfulness      # Job name
#SBATCH -A r02144                  # Your allocation account
#SBATCH -p gpu                     # GPU partition
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --gpus-per-node=v100:1     # 1 V100 GPU (32GB is enough)
#SBATCH --cpus-per-task=8          # CPU cores
#SBATCH --mem=64G                  # Memory
#SBATCH --time=24:00:00            # Max 24 hours (adjust based on instance count)
#SBATCH -o logs/legal_%j.out       # Output log
#SBATCH -e logs/legal_%j.err       # Error log
#SBATCH --mail-type=END,FAIL       # Email notifications
#SBATCH --mail-user=YOUR_EMAIL@iu.edu  # Replace with your email

# ============================================================
# Legal Domain Faithfulness Experiment on Quartz
# Compares Saul-7B (legal) vs Mistral-7B (baseline) on LegalBench
# ============================================================

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Create necessary directories
mkdir -p logs
mkdir -p results/legal

# Load required modules
module load python/3.10.5
module load cuda/11.8

# Create and activate virtual environment (first time only)
if [ ! -d "venv_legal" ]; then
    echo "Creating virtual environment..."
    python -m venv venv_legal
    source venv_legal/bin/activate
    
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install transformers accelerate datasets bitsandbytes
    pip install nltk spacy scipy numpy matplotlib
    python -m spacy download en_core_web_sm
    
    # Download NLTK data
    python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
else
    source venv_legal/bin/activate
fi

# Set environment variables
export TRANSFORMERS_CACHE=/N/scratch/$USER/hf_cache
export HF_HOME=/N/scratch/$USER/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# HuggingFace token (IMPORTANT: Set this!)
export HF_TOKEN="YOUR_HF_TOKEN_HERE"  # Replace with your actual token

# ============================================================
# CONFIGURATION - Change these as needed
# ============================================================

# Which model to run (change between runs)
MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2"  # Run this first
# MODEL_NAME="Equall/Saul-7B-Instruct-v1"        # Then run this

# Number of instances (30 for preliminary, 250 for full)
MAX_INSTANCES=30

# Other settings (keep these the same as medical experiment)
DATASET="legalbench"
METHOD="npo_KL"
STRATEGY="sentencize"
LR="3e-05"
EPOCHS=5
SEED=1001

# ============================================================
# Clone repository if needed
# ============================================================

REPO_DIR="$HOME/parametric-faithfulness"
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning repository..."
    cd $HOME
    git clone https://github.com/technion-cs-nlp/parametric-faithfulness.git
fi

cd $REPO_DIR

# ============================================================
# Setup LegalBench dataset
# ============================================================

echo "Setting up LegalBench dataset..."
python - <<'SETUP_DATASET'
import os
import json
from datasets import load_dataset

LEGAL_DIR = 'data/legalbench'
os.makedirs(LEGAL_DIR, exist_ok=True)
LEGAL_FILE = os.path.join(LEGAL_DIR, 'legalbench_test.json')

if not os.path.exists(LEGAL_FILE):
    print('Downloading LegalBench...')
    ds = load_dataset('nguha/legalbench', 'abercrombie', split='test')
    
    option_keys = ['A', 'B', 'C', 'D']
    label_map = {
        0: 'Generic',
        1: 'Descriptive', 
        2: 'Suggestive',
        3: 'Arbitrary or Fanciful'
    }
    
    converted = []
    for i, item in enumerate(ds):
        options_list = [f"{k}): {label_map[j]}" for j, k in enumerate(option_keys)]
        answer_idx = option_keys[item['label']]
        
        converted.append({
            'qid': f'legal_{i:05d}',
            'question': f"Classify the following trademark term according to the Abercrombie classification: '{item['text']}'",
            'options': options_list,
            'answer': answer_idx,
            'context': item['text']
        })
    
    with open(LEGAL_FILE, 'w') as f:
        json.dump(converted, f, indent=2)
    print(f'✅ LegalBench saved: {len(converted)} instances')
else:
    print('✅ LegalBench already exists')
SETUP_DATASET

# ============================================================
# Register LegalBench handler
# ============================================================

echo "Registering LegalBench dataset handler..."
cat > register_legalbench.py <<'REGISTER'
import os
import json
from dataload import DataHandler, DATASETS
from dataload import BOWMAN_HUMAN_ANSWER_PREFIX, BOWMAN_ASSISTANT_ANSWER_PREFIX

class LegalBenchDatasetHandler(DataHandler):
    id_key = 'qid'
    q_key  = 'question'
    letter_choices = ['A', 'B', 'C', 'D']

    def __init__(self):
        legal_path = 'data/legalbench/legalbench_test.json'
        with open(legal_path) as f:
            self._data = json.load(f)
        super().__init__()

    def get_dataset_splits(self):
        import random
        data = list(self._data)
        random.Random(42).shuffle(data)
        return data[:8], data[8:16], data

    def get_answer_letters(self, instance):
        return [opt[0] for opt in instance['options']]

    def get_answer_choices(self, instance):
        return instance['options']

    def correct_answer_letter(self, instance):
        return instance['answer']

    def make_bowman_demonstration(self, instance):
        choices = '\n'.join(f'({opt[0]}): {opt[3:]}' for opt in instance['options'])
        q = instance['question']
        return (
            f"Human: Question: {q}\n\n"
            f"Choices:\n{choices}\n\n"
            f"{BOWMAN_ASSISTANT_ANSWER_PREFIX}"
        )

    def make_cot_prompt(self, instance):
        choices = '\n'.join(f'({opt[0]}): {opt[3:]}' for opt in instance['options'])
        q = instance['question']
        return (
            f"Human: Question: {q}\n\n"
            f"Choices:\n{choices}\n\n"
            f"Assistant: Let's think step by step:\n"
        )

    def make_answer_prompt(self, prefix):
        return (
            f"{prefix}\n"
            f"{BOWMAN_HUMAN_ANSWER_PREFIX}\n"
            f"{BOWMAN_ASSISTANT_ANSWER_PREFIX}"
        )

DATASETS['legalbench'] = LegalBenchDatasetHandler()
print(f'✅ LegalBench registered: {len(DATASETS["legalbench"]._data)} instances')
REGISTER

# ============================================================
# Run the experiment
# ============================================================

echo "=========================================="
echo "Starting experiment with:"
echo "  Model: $MODEL_NAME"
echo "  Dataset: $DATASET"
echo "  Max instances: $MAX_INSTANCES"
echo "  Method: $METHOD"
echo "  Strategy: $STRATEGY"
echo "=========================================="

python run_experiment.py \
    --model_name "$MODEL_NAME" \
    --dataset "$DATASET" \
    --method "$METHOD" \
    --strategy "$STRATEGY" \
    --stepwise \
    --epochs $EPOCHS \
    --lr $LR \
    --seed $SEED \
    --pos \
    --ff2 \
    --max_instances $MAX_INSTANCES \
    --register_dataset register_legalbench.py

echo "=========================================="
echo "Job completed at: $(date)"
echo "Results saved to: results/legal/"
echo "=========================================="

# Deactivate virtual environment
deactivate
