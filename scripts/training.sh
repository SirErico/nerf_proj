#!/bin/bash

# List of dataset directories to train on
DATASETS=(
    "/workspace/datasets/ycb_dataset/tomato_soup_can"
    "/workspace/datasets/ycb_dataset/tuna_fish_can"
    "/workspace/datasets/ycb_dataset/power_drill"
    "/workspace/datasets/ycb_dataset/tennis_ball"
    "/workspace/datasets/ycb_dataset/rubiks_cube"
)

METHOD="nerfacto"
MAX_ITERS=20000
DATAPARSER="blender-data"
METHOD_OPTS="--output-dir /workspace/datasets/nerfstudio/outputs --pipeline.model.background-color white --pipeline.model.proposal-initial-sampler uniform --pipeline.model.near-plane 2. --pipeline.model.far-plane 6. --pipeline.model.camera-optimizer.mode off --pipeline.model.use-average-appearance-embedding False --pipeline.model.distortion-loss-mult 0 --pipeline.model.disable-scene-contraction True"
VIEWER_OPTS="--vis viewer+wandb --viewer.quit-on-train-completion True"

for DATASET in "${DATASETS[@]}"; do
    echo "Training on dataset: $DATASET"
    ns-train $METHOD $METHOD_OPTS --data="$DATASET" --max_num_iterations=$MAX_ITERS $VIEWER_OPTS $DATAPARSER
done

# export to point cloud
# ns-export pointcloud --load-config /workspace/datasets/nerfstudio/outputs/.../config.yml --output-dir exports/pcd/ --num-points 1000000 --remove-outliers True --normal-method open3d --obb-center 0.0 0.0 0.0 --obb-rotation  0.0 0.0 0.0 --obb-scale 0.8 0.8 1.3
