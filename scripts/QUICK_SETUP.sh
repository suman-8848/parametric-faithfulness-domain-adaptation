#!/bin/bash
# Quick setup script for Big Red 200
# Run this on Big Red 200: bash QUICK_SETUP.sh

echo "=========================================="
echo "Quick Setup for Legal Faithfulness Experiment"
echo "=========================================="

# Create necessary directories
mkdir -p logs
mkdir -p results/legalbench/mistral_7b_instruct_v0.2
mkdir -p results/legalbench/saul_7b_instruct_v1

# Make scripts executable
chmod +x run_mistral_only.sh
chmod +x run_saul_only.sh
chmod +x run_legal_bigred200.sh
chmod +x verify_legal_results.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "=========================================="
echo "Choose your option:"
echo "=========================================="
echo ""
echo "OPTION 1: Run in PARALLEL (FASTEST - ~6-8 hours)"
echo "  sbatch run_mistral_only.sh"
echo "  sbatch run_saul_only.sh"
echo ""
echo "OPTION 2: Run SEQUENTIALLY (~12-16 hours)"
echo "  sbatch run_legal_bigred200.sh"
echo ""
echo "=========================================="
echo "Don't forget to set your HuggingFace token:"
echo "  export HF_TOKEN='hf_your_token_here'"
echo "  OR"
echo "  huggingface-cli login"
echo "=========================================="
