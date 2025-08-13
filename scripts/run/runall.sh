#!/bin/bash

# Script to submit all shell scripts in a directory using bsub
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

# Check if there are any .sh files in the directory
shopt -s nullglob
scripts=("$SCRIPT_DIR"/*.sh)

if [ ${#scripts[@]} -eq 0 ]; then
    echo "No shell scripts (.sh files) found in '$SCRIPT_DIR'"
    exit 1
fi

echo "Found ${#scripts[@]} shell script(s) in '$SCRIPT_DIR'"
echo "Submitting jobs using bsub..."

# Submit each shell script using bsub
for script in "${scripts[@]}"; do
    script_name=$(basename "$script")
    echo "Submitting: $script_name"
    
    if bsub < "$script"; then
        echo "✓ Successfully submitted: $script_name"
    else
        echo "✗ Failed to submit: $script_name"
    fi
    echo ""
done

echo "All scripts processed."