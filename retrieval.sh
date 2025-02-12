#!/bin/bash

# Set the path to the directory containing checkpoints
CHECKPOINT_DIR="checkpoints"

LOG_DIR="retrieval_outputs/multiblock_masking"
# Check if the directory exists
if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "Error: Directory $CHECKPOINT_DIR does not exist."
    exit 1
fi

# Find all files containing 'ckpt' in their name
CHECKPOINT_FILES=("$CHECKPOINT_DIR"/*pth*)


# Check if any matching files exist
if [[ ! -e "${CHECKPOINT_FILES[0]}" ]]; then
    echo "No checkpoint files found in $CHECKPOINT_DIR"
    exit 1
fi
echo "Found checkpoint files:"
for checkpoint in "${CHECKPOINT_FILES[@]}"; do
    echo " - $checkpoint"
done
echo ""


# Loop through each checkpoint file and pass it to the Python script
for checkpoint in "${CHECKPOINT_FILES[@]}"; do
    echo "Processing checkpoint: $checkpoint"
    LOG_FILE="$LOG_DIR/$(basename "$checkpoint").log"
    python retrieval.py --path "$checkpoint" &> "$LOG_FILE"
    echo "Finished processing: $checkpoint"
done
