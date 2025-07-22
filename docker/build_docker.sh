#!/bin/bash

# This scripts builds a Docker image for Nerfstudio with COLMAP 3.11 support.
read -p "Enter CUDA architecture (default: 86): " CUDA_ARCH
CUDA_ARCH=${CUDA_ARCH:-86}

read -p "Enter nerfstudio version/branch (default: main): " NERF_VERSION
NERF_VERSION=${NERF_VERSION:-main}

read -p "Enter Docker tag name (default: nerf_colmap311): " TAG_NAME
TAG_NAME=${TAG_NAME:-nerf_colmap311}

echo ""
echo "Building Docker image with:"
echo "  CUDA Architecture: $CUDA_ARCH"
echo "  Nerfstudio Version: $NERF_VERSION"
echo "  Tag Name: $TAG_NAME"
echo ""

docker build \
    --build-arg CUDA_ARCHITECTURES=$CUDA_ARCH \
    --build-arg NERFSTUDIO_VERSION=$NERF_VERSION \
    --tag $TAG_NAME \
    --file Dockerfile .
