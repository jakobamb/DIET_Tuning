#!/bin/bash

# Script to submit all shell scripts in a directory using bsub (LSF) or sbatch (Slurm)
# Usage: ./runall.sh <directory_path>

# Check if directory argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory_path>"
    echo "Example: $0 /path/to/scripts"
    exit 1
fi

SCRIPT_DIR="$1"

# Check if the provided path is a valid directory
if [ ! -d "$SCRIPT_DIR" ]; then
    echo "Error: '$SCRIPT_DIR' is not a valid directory"
    exit 1
fi

# Detect job scheduler (Slurm or LSF)
SCHEDULER=""
if command -v sbatch &> /dev/null; then
    SCHEDULER="slurm"
    SUBMIT_CMD="sbatch"
    echo "Detected Slurm scheduler - using sbatch"
elif command -v bsub &> /dev/null; then
    SCHEDULER="lsf"
    SUBMIT_CMD="bsub"
    echo "Detected LSF scheduler - using bsub"
else
    echo "Error: Neither sbatch (Slurm) nor bsub (LSF) found in PATH"
    echo "Please ensure you have access to a job scheduler"
    exit 1
fi

# Check if there are any .sh files in the directory
shopt -s nullglob
scripts=("$SCRIPT_DIR"/*.sh)

if [ ${#scripts[@]} -eq 0 ]; then
    echo "No shell scripts (.sh files) found in '$SCRIPT_DIR'"
    exit 1
fi

echo "Found ${#scripts[@]} shell script(s) in '$SCRIPT_DIR'"
echo "Submitting jobs using $SUBMIT_CMD ($SCHEDULER)..."

# Submit each shell script using the detected scheduler
for script in "${scripts[@]}"; do
    script_name=$(basename "$script")
    echo "Submitting: $script_name"
    
    if [ "$SCHEDULER" = "slurm" ]; then
        # For Slurm, use sbatch with the script file
        if sbatch "$script"; then
            echo "✓ Successfully submitted: $script_name"
        else
            echo "✗ Failed to submit: $script_name"
        fi
    elif [ "$SCHEDULER" = "lsf" ]; then
        # For LSF, use bsub with input redirection
        if bsub < "$script"; then
            echo "✓ Successfully submitted: $script_name"
        else
            echo "✗ Failed to submit: $script_name"
        fi
    fi
    echo ""
done

echo "All scripts processed."