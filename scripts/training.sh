#!/bin/bash

# NeRF Training Script for YCB Datasets
# Trains multiple NeRF models using NeRFStudio with consistent parameters

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Configuration
DEFAULT_DATASETS=(
    "/workspace/datasets/ycb_dataset/tomato_soup_can"
    "/workspace/datasets/ycb_dataset/tuna_fish_can"
    "/workspace/datasets/ycb_dataset/power_drill"
    "/workspace/datasets/ycb_dataset/tennis_ball"
    "/workspace/datasets/ycb_dataset/rubiks_cube"
)

# Training parameters
METHOD="nerfacto"
MAX_ITERS=20000
DATAPARSER="blender-data"
OUTPUT_DIR="/workspace/datasets/nerfstudio/outputs"
METHOD_OPTS="--output-dir $OUTPUT_DIR \
--pipeline.model.background-color white \
--pipeline.model.proposal-initial-sampler uniform \
--pipeline.model.near-plane 2. \
--pipeline.model.far-plane 6. \
--pipeline.model.camera-optimizer.mode off \
--pipeline.model.use-average-appearance-embedding False \
--pipeline.model.distortion-loss-mult 0 \
--pipeline.model.disable-scene-contraction True"

VIEWER_OPTS="--vis viewer+wandb --viewer.quit-on-train-completion True"
# Train each dataset
for DATASET in "${DATASETS[@]}"; do
    echo "Training on dataset: $DATASET"
    ns-train $METHOD $METHOD_OPTS \
        --data="$DATASET" \
        --max_num_iterations=$MAX_ITERS \
        $VIEWER_OPTS \
        $DATAPARSER
done

# Export to point cloud (uncomment and modify paths as needed)
# ns-export pointcloud \
#   --load-config $OUTPUT_DIR/.../config.yml \
#   --output-dir exports/pcd/ \
#   --num-points 1000000 \
#   --remove-outliers True \
#   --normal-method open3d \
#   --obb-center 0.0 0.0 0.0 \
#   --obb-rotation 0.0 0.0 0.0 \
#   --obb-scale 0.8 0.8 1.3