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

class PointCloudMetrics:
    """
    A class to compute various metrics between two point clouds.
    """
    def __init__(self, pcd1, pcd2):
        """Initialize with two point clouds and compute distances."""
        self.distances_1_to_2 = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
        self.distances_2_to_1 = np.asarray(pcd2.compute_point_cloud_distance(pcd1))
        
    def compute_chamfer_distance(self):
        """Compute the Chamfer distance between the point clouds."""
        squared_dist_1_to_2 = self.distances_1_to_2 ** 2
        squared_dist_2_to_1 = self.distances_2_to_1 ** 2
        
        chamfer1 = np.mean(squared_dist_1_to_2)
        chamfer2 = np.mean(squared_dist_2_to_1)
        return (chamfer1 + chamfer2) / 2.0
    
    def compute_hausdorff_distance(self):
        """Compute the Hausdorff distance between the point clouds."""
        hausdorff_1_to_2 = np.max(self.distances_1_to_2)
        hausdorff_2_to_1 = np.max(self.distances_2_to_1)
        
        # Symmetric Hausdorff distance is the max of both directed distances
        return max(hausdorff_1_to_2, hausdorff_2_to_1)
    
    def compute_f_score(self, threshold=0.01):
        """Compute the F-score between the point clouds at the given threshold."""
        # Compute precision (percentage of points in pcd1 that have a neighbor in pcd2 within threshold)
        precision = np.mean(self.distances_1_to_2 < threshold)
        
        # Compute recall (percentage of points in pcd2 that have a neighbor in pcd1 within threshold)
        recall = np.mean(self.distances_2_to_1 < threshold)
        
        # Compute F-score
        if precision + recall > 0:
            f_score = 2 * precision * recall / (precision + recall)
        else:
            f_score = 0.0
            
        return f_score

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

# Removed export_pointclouds_from_folder as it's no longer needed


def evaluate_pointclouds(generated_path, gt_path, output_name=None, threshold=0.01, visualize=True):
    """
    Evaluate metrics between generated point cloud and ground truth.
    
    Args:
        generated_path: Path to the generated point cloud
        gt_path: Path to the ground truth point cloud
        output_name: Name to use in output messages (defaults to filename)
        threshold: Threshold for F-score computation
        visualize: Whether to show visualization
    """
    if not os.path.exists(generated_path):
        raise FileNotFoundError(f"Generated point cloud not found at: {generated_path}")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth not found at: {gt_path}")

    try:
        model_pcd = load_pointcloud(generated_path)
        gt_pcd = load_pointcloud(gt_path)
        
        # Use filename as output name if none provided
        if output_name is None:
            output_name = os.path.basename(os.path.dirname(generated_path))
        
        # Initialize metrics class
        metrics = PointCloudMetrics(model_pcd, gt_pcd)
        
        # Compute all metrics
        cd = metrics.compute_chamfer_distance()
        hd = metrics.compute_hausdorff_distance()
        f1 = metrics.compute_f_score(threshold=threshold)
        
        print(f"\nMetrics for {output_name}:")
        print(f"Chamfer Distance: {cd:.6f}")
        print(f"Hausdorff Distance: {hd:.6f}")
        print(f"F-score (threshold={threshold:.3f}): {f1:.6f}")
        print("-" * 50)
        
        if visualize:
            o3d.visualization.draw_geometries([model_pcd, gt_pcd], 
                                           window_name=output_name)
            
        return {"chamfer": cd, "hausdorff": hd, "f_score": f1}
        
    except Exception as e:
        print(f"Error processing {output_name}: {e}")
        return None

def main(args):
    if args.export:
        if not args.input_dir or not args.output_dir:
            print("Error: --input-dir and --output-dir are required when using --export")
            return
        if not os.path.exists(args.input_dir):
            raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
            
        # Run the export function with user-specified paths
        input_dir = args.input_dir
        output_dir = args.output_dir
        model_name = os.path.basename(input_dir)
        print(f"Exporting point cloud from {input_dir} to {output_dir}")
        export_pointcloud(model_name, input_dir, output_dir)
        return

    # If not exporting, require both point cloud paths for evaluation
    if not (args.generated and args.ground_truth):
        print("Error: Both --generated and --ground-truth paths are required for evaluation")
        return
        
    evaluate_pointclouds(
        args.generated,
        args.ground_truth,
        args.name,
        args.threshold,
        not args.no_viz
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export and evaluate point clouds.")
    
    # Export-related arguments
    parser.add_argument("--export", action="store_true", 
                      help="Export point clouds from nerfstudio outputs")
    parser.add_argument("--input-dir", type=str,
                      help="Input directory containing nerfstudio outputs")
    parser.add_argument("--output-dir", type=str,
                      help="Output directory for exported point clouds")
    
    # Evaluation-related arguments
    parser.add_argument("--generated", type=str,
                      help="Path to the generated point cloud .ply file")
    parser.add_argument("--ground-truth", type=str,
                      help="Path to the ground truth point cloud .ply file")
    parser.add_argument("--name", type=str,
                      help="Name to use in output (defaults to directory name)")
    parser.add_argument("--threshold", type=float, default=0.01,
                      help="Threshold for F-score computation (default: 0.01)")
    parser.add_argument("--no-viz", action="store_true",
                      help="Disable visualization")
    
    args = parser.parse_args()
    main(args)