#!/bin/bash
# Personalized upload script for slamsal
# Just run: ./RUN_ME_SLAMSAL.sh

USERNAME="slamsal"
QUARTZ_HOST="quartz.uits.iu.edu"
REMOTE_DIR="~/parametric-faithfulness"

echo "=========================================="
echo "🚀 Uploading Legal Experiment to Quartz"
echo "=========================================="
echo "Username: $USERNAME"
echo "Host: $QUARTZ_HOST"
echo ""

# Check if files exist
FILES=(
    "run_legal_experiment.sh"
    "run_experiment.py"
    "run_unlearning.py"
)

echo "Checking files..."
for file in "${FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Error: $file not found!"
        exit 1
    fi
    echo "✅ $file"
done

echo ""
echo "Uploading files to Quartz..."

# Create remote directory
ssh ${USERNAME}@${QUARTZ_HOST} "mkdir -p ${REMOTE_DIR}"

# Upload files
for file in "${FILES[@]}"; do
    echo "Uploading $file..."
    scp "$file" ${USERNAME}@${QUARTZ_HOST}:${REMOTE_DIR}/
    if [ $? -eq 0 ]; then
        echo "✅ $file uploaded"
    else
        echo "❌ Failed to upload $file"
        exit 1
    fi
done

# Make scripts executable
echo ""
echo "Making scripts executable..."
ssh ${USERNAME}@${QUARTZ_HOST} "cd ${REMOTE_DIR} && chmod +x run_legal_experiment.sh run_experiment.py run_unlearning.py"

echo ""
echo "=========================================="
echo "✅ Upload complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. SSH to Quartz:"
echo "   ssh slamsal@quartz.uits.iu.edu"
echo ""
echo "2. Navigate to directory:"
echo "   cd ~/parametric-faithfulness"
echo ""
echo "3. Edit configuration:"
echo "   nano run_legal_experiment.sh"
echo ""
echo "   Change these 2 lines:"
echo "   - Line 13: #SBATCH --mail-user=slamsal@iu.edu"
echo "   - Line 56: export HF_TOKEN=\"your_token_here\""
echo ""
echo "4. Submit first job (Mistral baseline):"
echo "   sbatch run_legal_experiment.sh"
echo ""
echo "5. Monitor progress:"
echo "   squeue -u slamsal"
echo "   tail -f logs/legal_*.out"
echo ""
echo "6. After first job completes, edit line 62:"
echo "   Change to: MODEL_NAME=\"Equall/Saul-7B-Instruct-v1\""
echo "   Then submit again: sbatch run_legal_experiment.sh"
echo ""
echo "7. Download results (on your local machine):"
echo "   scp -r slamsal@quartz.uits.iu.edu:~/parametric-faithfulness/results/legal ./"
echo ""
echo "=========================================="
echo "Total time: ~5-6 hours (mostly waiting)"
echo "=========================================="
