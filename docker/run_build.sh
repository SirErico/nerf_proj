#!/bin/bash
docker build \
    --build-arg CUDA_ARCHITECTURES=86 \
    --tag nerf_colmap311 \
    --file Dockerfile .
