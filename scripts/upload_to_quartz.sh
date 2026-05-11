#!/bin/bash
# Upload legal experiment files to Quartz
# Usage: ./upload_to_quartz.sh YOUR_USERNAME

if [ -z "$1" ]; then
    echo "Usage: ./upload_to_quartz.sh YOUR_USERNAME"
    echo "Example: ./upload_to_quartz.sh jdoe"
    exit 1
fi

USERNAME=$1
QUARTZ_HOST="quartz.uits.iu.edu"
REMOTE_DIR="~/parametric-faithfulness"

echo "=========================================="
echo "Uploading Legal Experiment Files to Quartz"
echo "=========================================="
echo "Username: $USERNAME"
echo "Host: $QUARTZ_HOST"
echo "Remote directory: $REMOTE_DIR"
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
echo "Uploading files..."

# Create remote directory if it doesn't exist
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
echo "1. SSH to Quartz:"
echo "   ssh ${USERNAME}@${QUARTZ_HOST}"
echo ""
echo "2. Navigate to directory:"
echo "   cd ~/parametric-faithfulness"
echo ""
echo "3. Edit configuration:"
echo "   nano run_legal_experiment.sh"
echo "   - Set your email (line 13)"
echo "   - Set your HF token (line 56)"
echo ""
echo "4. Submit job:"
echo "   sbatch run_legal_experiment.sh"
echo ""
echo "5. Monitor progress:"
echo "   squeue -u \$USER"
echo "   tail -f logs/legal_*.out"
echo ""
