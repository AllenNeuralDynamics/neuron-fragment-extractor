#!/bin/bash

# Subroutines
init_dir() {
    local dir="$1"
    if [ -d "$dir" ]; then
        rm -rf "$dir"
    fi
    mkdir -p "$dir"
}


# Parameters
BRAIN_ID="802449"
SEGMENTATION_ID="jin_masked_mean40_stddev105"

# Paths
OUTPUT_DIR="/home/jupyter/results/merge_datasets/${BRAIN_ID}/${SEGMENTATION_ID}"
TEMP_DIR="/home/jupyter/results/merge_datasets/${BRAIN_ID}/temp"

init_dir "$OUTPUT_DIR"
init_dir "$TEMP_DIR"

# Main
echo "Step 1: Extract Fragments"
python ../extract_via_skeleton_metrics.py \
    --brain_id "$BRAIN_ID" \
    --segmentation_id "$SEGMENTATION_ID" \
    --output_dir "${TEMP_DIR}"

echo "Step 2: Extract Merge Sites"

echo "Step 3: Extract Non-Merge Sites"
