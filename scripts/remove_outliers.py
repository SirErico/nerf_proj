import open3d as o3d
import sys
import numpy as np

def remove_statistical_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd.select_by_index(ind)

def remove_radius_outliers(pcd, nb_points=16, radius=0.02):
    cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    return pcd.select_by_index(ind)

def crop_sphere(pcd, center, radius):
    points = np.asarray(pcd.points)
    mask = np.linalg.norm(points - center, axis=1) < radius
    return pcd.select_by_index(np.where(mask)[0])

def crop_box(pcd, min_bound, max_bound):
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
    return pcd.crop(bbox)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python remove_outliers.py <input.ply> [method] [param1] [param2] [output.ply]")
        print("method: stat (default) or radius")
        print("stat params: [nb_neighbors] [std_ratio]")
        print("radius params: [nb_points] [radius]")
        sys.exit(1)
    ply_path = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else "stat"
    output_path = sys.argv[5] if len(sys.argv) > 5 else ply_path.replace(".ply", "_filtered.ply")

    pcd = o3d.io.read_point_cloud(ply_path)
    print(f"Loaded {ply_path}, points: {len(pcd.points)}")

    if method == "sphere":
        # Usage: python remove_outliers.py input.ply sphere <center_x> <center_y> <center_z> <radius> [output.ply]
        center = np.array([float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])])
        radius = float(sys.argv[6])
        filtered = crop_sphere(pcd, center, radius)
        print(f"Filtered points (sphere): {len(filtered.points)}")
    elif method == "box":
        # Usage: python remove_outliers.py input.ply box <min_x> <min_y> <min_z> <max_x> <max_y> <max_z> [output.ply]
        if len(sys.argv) > 8:
            min_bound = np.array([float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])])
            max_bound = np.array([float(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8])])
            output_path = sys.argv[9] if len(sys.argv) > 9 else ply_path.replace(".ply", "_filtered.ply")
        else:
            min_bound = pcd.get_min_bound()
            max_bound = pcd.get_max_bound()
            print(f"Using default box bounds: min={min_bound}, max={max_bound}")
            output_path = ply_path.replace(".ply", "_filtered.ply")
        filtered = crop_box(pcd, min_bound, max_bound)
        print(f"Filtered points (box): {len(filtered.points)}")
    else:
        if method == "radius":
            nb_points = int(sys.argv[3]) if len(sys.argv) > 3 else 16
            radius = float(sys.argv[4]) if len(sys.argv) > 4 else 0.02
            filtered = remove_radius_outliers(pcd, nb_points, radius)
            print(f"Filtered points (radius): {len(filtered.points)}")
        else:
            nb_neighbors = int(sys.argv[3]) if len(sys.argv) > 3 else 20
            std_ratio = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
            filtered = remove_statistical_outliers(pcd, nb_neighbors, std_ratio)
            print(f"Filtered points (statistical): {len(filtered.points)}")

    o3d.io.write_point_cloud(output_path, filtered)
    print(f"Saved filtered point cloud to {output_path}")

    # Optional: visualize result
    # o3d.visualization.draw_geometries([filtered])
