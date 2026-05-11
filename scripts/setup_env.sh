#!/bin/bash
# Setup Python environment for legal faithfulness experiment
# Run this BEFORE submitting jobs: bash setup_env.sh

echo "=========================================="
echo "Setting up Python environment..."
echo "=========================================="

# Load modules
module load python/3.11.5
module load cuda/12.2.0

# Remove old venv if it exists and is broken
if [ -d "venv" ]; then
    echo "Removing old virtual environment..."
    rm -rf venv
fi

# Create fresh virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate it
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install transformers and related packages
echo "Installing transformers..."
pip install transformers accelerate datasets bitsandbytes

# Install NLP packages
echo "Installing NLP packages..."
pip install nltk spacy scipy

# Download spacy model
echo "Downloading spacy model..."
python -m spacy download en_core_web_sm

# Download NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Clone repository if needed
REPO_DIR="parametric-faithfulness"
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning parametric-faithfulness repository..."
    git clone https://github.com/technion-cs-nlp/parametric-faithfulness.git $REPO_DIR
fi

# Test imports
echo ""
echo "Testing imports..."
python -c "import torch; print(f'✅ PyTorch {torch.__version__}')"
python -c "import transformers; print(f'✅ Transformers {transformers.__version__}')"
python -c "import spacy; print('✅ Spacy')"
python -c "import nltk; print('✅ NLTK')"

echo ""
echo "=========================================="
echo "✅ Environment setup complete!"
echo "=========================================="
echo ""
echo "Now you can submit jobs:"
echo "  sbatch run_mistral_only.sh"
echo "  sbatch run_saul_only.sh"
echo ""
