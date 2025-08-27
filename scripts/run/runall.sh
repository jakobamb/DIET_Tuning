#!/bin/bash

# Script to submit all shell scripts in a directory as a job array using bsub (LSF) or sbatch (Slurm)
# Usage: ./runall.sh <directory_path> [additional_python_args...]

# Cleanup function
cleanup() {
    if [ -n "$SCRIPT_LIST_FILE" ] && [ -f "$SCRIPT_LIST_FILE" ]; then
        echo "Cleaning up temporary files..."
        rm -f "$SCRIPT_LIST_FILE"
    fi
    if [ -n "$ARRAY_SCRIPT" ] && [ -f "$ARRAY_SCRIPT" ]; then
        rm -f "$ARRAY_SCRIPT"
    fi
}

# Set up cleanup on exit
trap cleanup EXIT

# Check if directory argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory_path> [additional_python_args...]"
    echo "Example: $0 /path/to/scripts"
    echo "Example: $0 /path/to/scripts --epochs 200 --lr 1e-3"
    echo "Note: Scripts will be submitted as a job array to reduce queue congestion"
    exit 1
fi

SCRIPT_DIR="$1"
shift  # Remove first argument, rest are python args
PYTHON_ARGS="$@"

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
if [ -n "$PYTHON_ARGS" ]; then
    echo "Additional Python arguments: $PYTHON_ARGS"
fi

# Ensure logs directory exists
mkdir -p logs

# Create a temporary script list file for the job array
SCRIPT_LIST_FILE=$(mktemp)
printf '%s\n' "${scripts[@]}" > "$SCRIPT_LIST_FILE"

echo "Creating job array with ${#scripts[@]} tasks using $SUBMIT_CMD ($SCHEDULER)..."
echo "Script list saved to: $SCRIPT_LIST_FILE"

# Submit job array based on scheduler type
if [ "$SCHEDULER" = "slurm" ]; then
    # Create Slurm job array script
    ARRAY_SCRIPT=$(mktemp --suffix=.sh)
    cat > "$ARRAY_SCRIPT" << 'EOF'
#!/bin/bash
#SBATCH --job-name=runall_array
#SBATCH --output=logs/runall_array_%A_%a.out
#SBATCH --error=logs/runall_array_%A_%a.err
#SBATCH --array=1-ARRAY_SIZE

# Get the script to run from the list file
SCRIPT_LIST_FILE="SCRIPT_LIST_PLACEHOLDER"
SCRIPT_TO_RUN=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$SCRIPT_LIST_FILE")

if [ -z "$SCRIPT_TO_RUN" ]; then
    echo "Error: Could not find script for task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Running script: $SCRIPT_TO_RUN"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Started at: $(date)"

# Extract and export additional Python args if provided
if [ -n "$EXTRA_PYTHON_ARGS" ]; then
    export EXTRA_PYTHON_ARGS
    echo "Extra Python args: $EXTRA_PYTHON_ARGS"
fi

# Execute the individual script
echo "Executing: bash $SCRIPT_TO_RUN"
bash "$SCRIPT_TO_RUN"
SCRIPT_EXIT_CODE=$?

echo "Script completed at: $(date)"
echo "Exit code: $SCRIPT_EXIT_CODE"

# Clean up temp files on the last task
if [ "$SLURM_ARRAY_TASK_ID" -eq "ARRAY_SIZE" ]; then
    echo "Cleaning up temporary files (last task)..."
    rm -f "$SCRIPT_LIST_FILE" 2>/dev/null || true
fi

exit $SCRIPT_EXIT_CODE
EOF
    
    # Replace placeholders in the array script
    sed -i "s/ARRAY_SIZE/${#scripts[@]}/g" "$ARRAY_SCRIPT"
    sed -i "s|SCRIPT_LIST_PLACEHOLDER|$SCRIPT_LIST_FILE|g" "$ARRAY_SCRIPT"
    
    # Submit the job array
    if [ -n "$PYTHON_ARGS" ]; then
        if sbatch --export=ALL,EXTRA_PYTHON_ARGS="$PYTHON_ARGS" "$ARRAY_SCRIPT"; then
            echo "✓ Successfully submitted job array with ${#scripts[@]} tasks (with args: $PYTHON_ARGS)"
        else
            echo "✗ Failed to submit job array"
            exit 1
        fi
    else
        if sbatch "$ARRAY_SCRIPT"; then
            echo "✓ Successfully submitted job array with ${#scripts[@]} tasks"
        else
            echo "✗ Failed to submit job array"
            exit 1
        fi
    fi
    
    echo "Job array script: $ARRAY_SCRIPT"

elif [ "$SCHEDULER" = "lsf" ]; then
    # Create LSF job array script
    ARRAY_SCRIPT=$(mktemp --suffix=.sh)
    cat > "$ARRAY_SCRIPT" << 'EOF'
#!/bin/bash
#BSUB -J runall_array[1-ARRAY_SIZE]
#BSUB -o logs/runall_array_%J_%I.out
#BSUB -e logs/runall_array_%J_%I.err

# Get the script to run from the list file
SCRIPT_LIST_FILE="SCRIPT_LIST_PLACEHOLDER"
SCRIPT_TO_RUN=$(sed -n "${LSB_JOBINDEX}p" "$SCRIPT_LIST_FILE")

if [ -z "$SCRIPT_TO_RUN" ]; then
    echo "Error: Could not find script for task ID $LSB_JOBINDEX"
    exit 1
fi

echo "Running script: $SCRIPT_TO_RUN"
echo "Task ID: $LSB_JOBINDEX"
echo "Started at: $(date)"

# Extract and export additional Python args if provided
if [ -n "$EXTRA_PYTHON_ARGS" ]; then
    export EXTRA_PYTHON_ARGS
    echo "Extra Python args: $EXTRA_PYTHON_ARGS"
fi

# Execute the individual script
echo "Executing: bash $SCRIPT_TO_RUN"
bash "$SCRIPT_TO_RUN"
SCRIPT_EXIT_CODE=$?

echo "Script completed at: $(date)"
echo "Exit code: $SCRIPT_EXIT_CODE"

# Clean up temp files on the last task
if [ "$LSB_JOBINDEX" -eq "ARRAY_SIZE" ]; then
    echo "Cleaning up temporary files (last task)..."
    rm -f "$SCRIPT_LIST_FILE" 2>/dev/null || true
fi

exit $SCRIPT_EXIT_CODE
EOF
    
    # Replace placeholders in the array script
    sed -i "s/ARRAY_SIZE/${#scripts[@]}/g" "$ARRAY_SCRIPT"
    sed -i "s|SCRIPT_LIST_PLACEHOLDER|$SCRIPT_LIST_FILE|g" "$ARRAY_SCRIPT"
    
    # Submit the job array
    if [ -n "$PYTHON_ARGS" ]; then
        if EXTRA_PYTHON_ARGS="$PYTHON_ARGS" bsub < "$ARRAY_SCRIPT"; then
            echo "✓ Successfully submitted job array with ${#scripts[@]} tasks (with args: $PYTHON_ARGS)"
        else
            echo "✗ Failed to submit job array"
            exit 1
        fi
    else
        if bsub < "$ARRAY_SCRIPT"; then
            echo "✓ Successfully submitted job array with ${#scripts[@]} tasks"
        else
            echo "✗ Failed to submit job array"
            exit 1
        fi
    fi
    
    echo "Job array script: $ARRAY_SCRIPT"
fi

echo "Job array submitted successfully!"
echo "Monitor with: squeue -u \$USER (Slurm) or bjobs (LSF)"
echo ""
echo "To cancel the entire array:"
echo "  Slurm: scancel <job_id>"
echo "  LSF: bkill -J runall_array"