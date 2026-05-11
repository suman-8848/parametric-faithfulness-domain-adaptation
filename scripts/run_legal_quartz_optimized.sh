#!/bin/bash
#SBATCH -J legal_faith_opt        # Job name
#SBATCH -A r02144                 # Your allocation account
#SBATCH -p gpu                    # GPU partition
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --gpus-per-node=v100:1    # 1 V100 GPU
#SBATCH --cpus-per-task=8         # CPU cores
#SBATCH --mem=128G                # Increased memory
#SBATCH --time=24:00:00           # Max 24 hours
#SBATCH -o logs/legal_%j.out      # Output log
#SBATCH -e logs/legal_%j.err      # Error log
#SBATCH --mail-type=END,FAIL      # Email notifications
#SBATCH --mail-user=slamsal@iu.edu

# ============================================================
# Legal Domain Faithfulness Experiment - OPTIMIZED
# User preferences: lr=1e-5, epochs=10, max_instances=30
# ============================================================

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Create necessary directories
mkdir -p logs
mkdir -p results/legalbench
mkdir -p data/legalbench

# Load modules
module load python
module load cuda

echo "Loaded modules:"
module list

# Set environment variables for memory optimization
# export HF_TOKEN="your_token_here"  # Set your HuggingFace token
export TRANSFORMERS_CACHE=/N/scratch/$USER/hf_cache
export HF_HOME=/N/scratch/$USER/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================
# CONFIGURATION - YOUR PREFERENCES
# ============================================================

# Which model to run (change between runs)
MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2"  # Run this first
# MODEL_NAME="Equall/Saul-7B-Instruct-v1"        # Then run this

# Your preferred settings
MAX_INSTANCES=3          # Testing with 3 first
LEARNING_RATE=1e-5       # Your preference (was 3e-5)
EPOCHS=10                # Your preference (was 5)

# Other settings
DATASET="legalbench"
METHOD="npo_KL"
STRATEGY="sentencize"
SEED=1001

# ============================================================
# Setup Python environment
# ============================================================

# Initialize conda
source $HOME/miniconda3/etc/profile.d/conda.sh

# Create/activate environment
if ! conda env list | grep -q "legal_env"; then
    echo "Creating conda environment..."
    conda create -n legal_env python=3.10 -y
    conda activate legal_env
    
    # Install dependencies
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install transformers accelerate datasets bitsandbytes
    pip install nltk spacy scipy numpy matplotlib tqdm
    
    # Download NLTK and spacy data
    python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
    python -m spacy download en_core_web_sm
else
    conda activate legal_env
fi

# ============================================================
# Navigate to repository
# ============================================================

REPO_DIR="$HOME/parametric-faithfulness"
cd $REPO_DIR

# ============================================================
# Setup LegalBench dataset
# ============================================================

echo "Setting up LegalBench dataset..."
python - <<'SETUP_DATASET'
import os
import json

LEGAL_DIR = 'data/legalbench'
os.makedirs(LEGAL_DIR, exist_ok=True)
LEGAL_FILE = os.path.join(LEGAL_DIR, 'legalbench_test.json')

if not os.path.exists(LEGAL_FILE):
    print('Creating LegalBench dataset...')
    from datasets import load_dataset
    
    try:
        ds = load_dataset('nguha/legalbench', 'abercrombie', split='test', trust_remote_code=True)
        
        option_keys = ['A', 'B', 'C', 'D']
        label_map = {
            0: 'Generic',
            1: 'Descriptive', 
            2: 'Suggestive',
            3: 'Arbitrary or Fanciful'
        }
        
        converted = []
        for i, item in enumerate(ds):
            text = item.get('text', item.get('input', ''))
            label = item.get('label', item.get('answer', 0))
            
            options_list = [f"{k}): {label_map[j]}" for j, k in enumerate(option_keys)]
            answer_idx = option_keys[int(label)]
            
            converted.append({
                'qid': f'legal_{i:05d}',
                'question': f"Classify the following trademark term according to the Abercrombie classification: '{text}'",
                'options': options_list,
                'answer': answer_idx,
                'context': text
            })
        
        with open(LEGAL_FILE, 'w') as f:
            json.dump(converted, f, indent=2)
        print(f'✅ LegalBench saved: {len(converted)} instances')
        
    except Exception as e:
        print(f'Error: {e}')
        print('Creating minimal test dataset...')
        test_data = []
        for i in range(50):
            test_data.append({
                'qid': f'legal_{i:05d}',
                'question': f"Classify trademark term {i}: Is it Generic, Descriptive, Suggestive, or Arbitrary/Fanciful?",
                'options': ['A): Generic', 'B): Descriptive', 'C): Suggestive', 'D): Arbitrary or Fanciful'],
                'answer': ['A', 'B', 'C', 'D'][i % 4],
                'context': f'Test term {i}'
            })
        with open(LEGAL_FILE, 'w') as f:
            json.dump(test_data, f, indent=2)
        print(f'✅ Test dataset created: {len(test_data)} instances')
else:
    print('✅ LegalBench already exists')
SETUP_DATASET

# ============================================================
# Register LegalBench handler
# ============================================================

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
echo "  Learning rate: $LEARNING_RATE"
echo "  Epochs: $EPOCHS"
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
    --lr $LEARNING_RATE \
    --seed $SEED \
    --pos \
    --ff2 \
    --max_instances $MAX_INSTANCES \
    --register_dataset register_legalbench.py

EXIT_CODE=$?

echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS!"
    echo "Results saved to: results/legalbench/"
else
    echo "❌ FAILED with exit code $EXIT_CODE"
    echo "Check logs/legal_${SLURM_JOB_ID}.err for details"
fi
echo "=========================================="

# Show GPU memory stats
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "Final GPU memory usage:"
    nvidia-smi
fi

# Deactivate conda
conda deactivate
