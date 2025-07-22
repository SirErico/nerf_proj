import subprocess
import os
import argparse
import open3d as o3d
import numpy as np


def load_pointcloud(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    if len(pcd.points) == 0:
        raise ValueError(f"Pointcloud at {ply_path} is empty or invalid.")
    return pcd

def compute_chamfer_distance(pcd1, pcd2):
    # Compute squared distances from each point in pcd1 to its nearest neighbor in pcd2
    distances_1_to_2 = np.asarray(pcd1.compute_point_cloud_distance(pcd2)) ** 2
    # Compute squared distances from each point in pcd2 to its nearest neighbor in pcd1
    distances_2_to_1 = np.asarray(pcd2.compute_point_cloud_distance(pcd1)) ** 2

    chamfer1 = np.mean(distances_1_to_2)
    chamfer2 = np.mean(distances_2_to_1)
    return chamfer1, chamfer2

def export_pointcloud(model_name, input_dir, output_dir="exports"):
    config_path = f"{input_dir}/config.yml"
    #export_path = os.path.join(output_dir, model_name)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at: {config_path}")

    os.makedirs(output_dir, exist_ok=True)

    command = [
        "ns-export",
        "pointcloud",
        "--load-config", config_path,
        "--output-dir", output_dir,
        "--normal-method", "open3d",
        "--num-points", '1000000',
        "--remove-outliers", 'True',
        "--obb_center", '0.0000000000', '0.0000000000', '0.0000000000',
        "--obb_rotation", '0.0000000000', '0.0000000000', '0.0000000000', 
        "--obb_scale", '2.0000000000', '2.0000000000', '2.0000000000'
    ]

    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print("Error during export:")
        print(result.stderr)
    else:
        print("Export successful!")
        print(result.stdout)

    return os.path.join(output_dir, "point_cloud.ply")

def export_pointclouds_from_folder():

    for files in os.listdir("datasets/nerfstudio/outputs"):
        model_name = files
        print(f"Exporting point cloud for model: {model_name}")
        for methods in os.listdir(f"datasets/nerfstudio/outputs/{files}"):
            method_use = methods
            for experiments in os.listdir(f"datasets/nerfstudio/outputs/{files}/{methods}"):
                experiment_use = experiments
                input_dir = f"datasets/nerfstudio/outputs/{model_name}/{method_use}/{experiment_use}"
                output_dir = f"exports/{model_name}/{method_use}/{experiment_use}"
                print(f"Input directory: {input_dir}")
                print(f"Output directory: {output_dir}")
                export_pointcloud(model_name, input_dir, output_dir)


def main(is_export):

    if is_export:
        export_pointclouds_from_folder()

    for files in os.listdir("exports"):
        model_name = files
        for methods in os.listdir(f"exports/{files}"):
            method_use = methods
            for experiments in os.listdir(f"exports/{files}/{methods}"):
                experiment_use = experiments
                model_path_pcd = f"exports/{model_name}/{method_use}/{experiment_use}/point_cloud.ply"
                model_path_gt = f"ycb_ply_gt/{model_name}/nontextured.ply"
                if not os.path.exists(model_path_gt):
                    print(f"Ground truth file not found for {model_path_gt}, skipping.")
                    continue
                try:

                    model_pcd = load_pointcloud(model_path_pcd)
                    gt_pcd = load_pointcloud(model_path_gt)               
                    
                    dist1, dist2 = compute_chamfer_distance(model_pcd, gt_pcd)

                    cd = dist1 + dist2

                    print(f"Chamfer Distance between {model_path_pcd} and {model_path_gt}: {cd:.6f}")
                except Exception as e:
                    print(f"Error processing {model_name}/{experiment_use}: {e}")
                o3d.visualization.draw_geometries([model_pcd, gt_pcd], window_name=f"{model_name}/{experiment_use}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export and visualize point clouds.")
    parser.add_argument("--export", action="store_true", help="Export point clouds from nerfstudio outputs.")
    args  = parser.parse_args()
    main(is_export=args.export)