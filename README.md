# 🛠 WIP
# Github repository for the "Geometry-Based Evaluation of State-of-the-Art NeRF Methods Using Real-World and Synthetic Datasets" project. 



1. Clone the repository:
```bash
git clone https://github.com/SirErico/nerf_proj
```

2. Run docker compose
```bash
cd docker
docker compose build
```

We mount the /nerf_dataset:/workspace/nerf_dataset, so store your bags there.
3. Inspect if bag is visible:
```bash
docker compose run --rm ros2_collector \
  ros2 bag info /workspace/nerf_dataset/tomato_soup_can/tomato_soup_can_test
```

4. Convert the bag to nerfstudio dataset:
```bash
docker compose run --rm ros2_collector bash -lc \
'run_robot2nerf.sh /workspace/nerf_dataset/tomato_soup_can/tomato_soup_can_test \
  --image_topic /rgb/image_raw \
  --camera_info_topic /rgb/camera_info \
  --source_frame base_link \
  --target_frame azure_rgb \
  --output_dir /workspace/nerf_dataset/dataset_from_bag \
  --collection_rate 10'
```

5. Train the model:
```bash
docker compose run --rm --service-ports nerfstudio_lrm bash -lc \
'ns-train nerfacto \
  --data /workspace/nerf_dataset/dataset_from_bag \
  --viewer.websocket-port 7007 \
  --viewer.websocket-host 0.0.0.0'
```


## Methods for comparsion
- **NeRF**: Neural Radiance Fields ([Paper](https://arxiv.org/pdf/2003.08934))
- **Mip-NeRF**: Mip-NeRF: A Multiscale Representation for Anti-Aliased Neural Radiance Fields ([Paper](https://arxiv.org/pdf/2103.13415))
- **Instant-NGP**: Instant Neural Graphics Primitives with a Multiresolution Hash Encoding ([Paper](https://arxiv.org/pdf/2201.05989))
- **Nerfacto**: Nerfacto: method created by the Nerfstudio team ([Paper](https://arxiv.org/pdf/2302.04264))
- **NeuS**: NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction ([Paper](https://arxiv.org/pdf/2106.10689))
- **PermutoSDF**: Fast Multi-View Neural Surface Reconstruction with Implicit Surfaces using Permutohedral Lattices ([Paper](https://arxiv.org/pdf/2211.12562))

## Datasets
- YCB dataset: [Link](http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/)
- nerf_synthetic dataset: [Link](https://www.kaggle.com/datasets/nguyenhung1903/nerf-synthetic-dataset)

## Other
- Nerf Baselines: [Link](https://nerfbaselines.github.io/)
- Analysis of NeRF-Based 3D reconstruction methods: [Link](https://www.researchgate.net/publication/376484396_A_Critical_Analysis_of_NeRF-Based_3D_Reconstruction)

