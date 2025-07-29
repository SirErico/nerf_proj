import os
import cv2
import rclpy
import numpy as np
import json
import networkx as nx

from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from sensor_msgs.msg import Image, CameraInfo
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
from tf_transformations import quaternion_matrix, quaternion_from_matrix
from scipy.spatial.transform import Rotation as R

def normalize_frame(frame_id):
    return frame_id.strip().lstrip('/')

def build_tf_graph(tf_messages):
    """Build a directed graph of available transforms using networkx"""
    graph = nx.DiGraph()
    for tf in tf_messages:
        parent = normalize_frame(tf.header.frame_id)
        child = normalize_frame(tf.child_frame_id)
        graph.add_edge(parent, child, transform=tf)
    return graph

def find_transform_path(graph, source, target):
    """Find a transform path from source to target in the graph"""
    try:
        path = nx.shortest_path(graph, source=normalize_frame(source), target=normalize_frame(target))
        return path
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None

def transform_to_matrix(transform):
    t = transform.transform.translation
    q = transform.transform.rotation
    mat = quaternion_matrix([q.x, q.y, q.z, q.w])
    mat[0, 3] = t.x
    mat[1, 3] = t.y
    mat[2, 3] = t.z
    return mat

def matrix_to_transform(matrix, parent_frame, child_frame):
    trans = matrix[:3, 3]
    rot = quaternion_from_matrix(matrix)
    tf = TransformStamped()
    tf.header.frame_id = parent_frame
    tf.child_frame_id = child_frame
    tf.transform.translation.x = trans[0]
    tf.transform.translation.y = trans[1]
    tf.transform.translation.z = trans[2]
    tf.transform.rotation.x = rot[0]
    tf.transform.rotation.y = rot[1]
    tf.transform.rotation.z = rot[2]
    tf.transform.rotation.w = rot[3]
    return tf

def compose_transform_chain(graph, path):
    composed_matrix = np.identity(4)
    for i in range(len(path) - 1):
        parent = path[i]
        child = path[i + 1]
        tf = graph[parent][child]['transform']
        mat = transform_to_matrix(tf)
        composed_matrix = composed_matrix @ mat
    return matrix_to_transform(composed_matrix, path[0], path[-1])

def transform_to_nerf_matrix(transform_stamped):
    """Convert ROS TransformStamped to NeRF-compatible 4x4 matrix"""
    t = transform_stamped.transform.translation
    r = transform_stamped.transform.rotation
    
    # Create transformation matrix
    rotation_matrix = R.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_matrix
    matrix[:3, 3] = [t.x, t.y, t.z]
    
    # Convert ROS to NeRF coordinate system
    # ROS: +X forward, +Y left, +Z up
    # NeRF: +X right, +Y up, +Z backward
    ros_to_nerf = np.array([
        [0, -1, 0, 0],
        [0, 0, 1, 0], 
        [-1, 0, 0, 0],
        [0, 0, 0, 1]
    ])
    
    nerf_transform = matrix @ ros_to_nerf
    return nerf_transform.tolist()

def read_rosbag(bag_path, image_topic='/rgb/image_raw', camera_info_topic='/rgb/camera_info', tf_src='base_link', tf_target='azure_rgb', output_dir='nerf_dataset/dataset2'):
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    bridge = CvBridge()
    tf_messages = []
    static_tf_messages = []
    camera_info = None
    frames_data = []

    # Set up ROS 2 bag reader
    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader.open(storage_options, converter_options)
    topic_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topic_types}

    print(f"Available topics in bag:")
    for topic in topic_types:
        print(f"  {topic.name}: {topic.type}")

    # First pass: collect all transforms
    print("[Phase 1] Collecting transforms...")
    while reader.has_next():
        topic, data, t = reader.read_next()
        msg_type = type_map.get(topic)

        if topic == '/tf' and msg_type == 'tf2_msgs/msg/TFMessage':
            msg = deserialize_message(data, TFMessage)
            tf_messages.extend(msg.transforms)

        elif topic == '/tf_static' and msg_type == 'tf2_msgs/msg/TFMessage':
            msg = deserialize_message(data, TFMessage)
            static_tf_messages.extend(msg.transforms)

        elif topic == camera_info_topic and msg_type == 'sensor_msgs/msg/CameraInfo':
            if camera_info is None:  # Store the first camera info we see
                msg = deserialize_message(data, CameraInfo)
                camera_info = msg
                print(f'[Camera Info] {msg.width}x{msg.height}, fx={msg.k[0]:.1f}, fy={msg.k[4]:.1f}')

    # Build TF graph and find transform path
    all_transforms = tf_messages + static_tf_messages
    print(f'[Summary] Found {len(tf_messages)} dynamic transforms and {len(static_tf_messages)} static transforms')
    
    tf_graph = build_tf_graph(all_transforms)
    transform_path = find_transform_path(tf_graph, tf_src, tf_target)

    if transform_path:
        print(f'[TF Graph] Found transform path: {" -> ".join(transform_path)}')
    else:
        print(f'[Warning] No transform path found between {tf_src} and {tf_target} in bag.')
        # Print available transforms for debugging
        print('[Debug] Available transforms:')
        for tf in all_transforms[:10]:  # Show first 10
            parent = normalize_frame(tf.header.frame_id)
            child = normalize_frame(tf.child_frame_id)
            print(f"  {parent} -> {child}")
        if len(all_transforms) > 10:
            print(f"  ... and {len(all_transforms) - 10} more")

    # Second pass: process images
    print("[Phase 2] Processing images...")
    reader = SequentialReader()  # Reset reader
    reader.open(storage_options, converter_options)
    
    image_counter = 0

    while reader.has_next():
        topic, data, t = reader.read_next()
        msg_type = type_map.get(topic)

        if topic == image_topic and msg_type == 'sensor_msgs/msg/Image':
            msg = deserialize_message(data, Image)
            try:
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                filename = f'frame_{image_counter:05d}.jpg'
                filepath = os.path.join(images_dir, filename)
                cv2.imwrite(filepath, cv_image)
                print(f'[Image] Saved {filepath}')
                
                # Get image timestamp for temporal matching
                image_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
                
                # Find the closest transform in time to this image
                matching_transform = None
                closest_time_diff = float('inf')
                
                # Look for time-varying transforms first (from /tf topic)
                for tf in tf_messages:  # Only dynamic transforms
                    if (tf.header.frame_id == tf_src and tf.child_frame_id == tf_target) or \
                       (tf.header.frame_id == tf_target and tf.child_frame_id == tf_src):
                        tf_timestamp = tf.header.stamp.sec + tf.header.stamp.nanosec / 1e9
                        time_diff = abs(tf_timestamp - image_timestamp)
                        if time_diff < closest_time_diff:
                            closest_time_diff = time_diff
                            matching_transform = tf
                
                # If no direct dynamic transform found, try to compose using time-varying transforms
                if matching_transform is None and transform_path:
                    # Build a time-specific TF graph using transforms closest to image timestamp
                    time_specific_transforms = []
                    
                    # For each step in the path, find the closest transform in time
                    for i in range(len(transform_path) - 1):
                        parent = transform_path[i]
                        child = transform_path[i + 1]
                        
                        best_tf = None
                        best_time_diff = float('inf')
                        
                        # Look for time-varying transform for this step
                        for tf in tf_messages:
                            if normalize_frame(tf.header.frame_id) == parent and normalize_frame(tf.child_frame_id) == child:
                                tf_timestamp = tf.header.stamp.sec + tf.header.stamp.nanosec / 1e9
                                time_diff = abs(tf_timestamp - image_timestamp)
                                if time_diff < best_time_diff:
                                    best_time_diff = time_diff
                                    best_tf = tf
                        
                        # If no dynamic transform found, use static one
                        if best_tf is None:
                            for tf in static_tf_messages:
                                if normalize_frame(tf.header.frame_id) == parent and normalize_frame(tf.child_frame_id) == child:
                                    best_tf = tf
                                    break
                        
                        if best_tf:
                            time_specific_transforms.append(best_tf)
                    
                    # Compose the time-specific transforms
                    if len(time_specific_transforms) == len(transform_path) - 1:
                        time_specific_graph = build_tf_graph(time_specific_transforms)
                        matching_transform = compose_transform_chain(time_specific_graph, transform_path)
                        print(f'[Transform] Composed time-specific transform for image {image_counter} at time {image_timestamp:.3f}')
                
                # Fallback to any available transform
                if matching_transform is None:
                    for tf in all_transforms:
                        if (tf.header.frame_id == tf_src and tf.child_frame_id == tf_target) or \
                           (tf.header.frame_id == tf_target and tf.child_frame_id == tf_src):
                            tf_timestamp = tf.header.stamp.sec + tf.header.stamp.nanosec / 1e9
                            time_diff = abs(tf_timestamp - image_timestamp)
                            if time_diff < closest_time_diff:
                                closest_time_diff = time_diff
                                matching_transform = tf
                
                if matching_transform:
                    nerf_matrix = transform_to_nerf_matrix(matching_transform)
                    frame_data = {
                        "file_path": f"images/{filename}",
                        "transform_matrix": nerf_matrix
                    }
                    frames_data.append(frame_data)
                    
                    # Print transform info for the first few images
                    if image_counter < 3:
                        print(f'[Transform] Image {image_counter} at time {image_timestamp:.3f}:')
                        print(f'  Transform: {matching_transform.header.frame_id} -> {matching_transform.child_frame_id}')
                        t = matching_transform.transform.translation
                        print(f'  Translation: [{t.x:.3f}, {t.y:.3f}, {t.z:.3f}]')
                        print(f'  Time diff: {closest_time_diff:.3f}s')
                else:
                    print(f'[Warning] No transform found for image {image_counter}')
                
                image_counter += 1
            except Exception as e:
                print(f'Error converting image: {e}')

        elif topic == camera_info_topic and msg_type == 'sensor_msgs/msg/CameraInfo':
            msg = deserialize_message(data, CameraInfo)
            if camera_info is None:  # Store the first camera info we see
                camera_info = msg
                print(f'[Camera Info] {msg.width}x{msg.height}, fx={msg.k[0]:.1f}, fy={msg.k[4]:.1f}')

        elif topic == '/tf' and msg_type == 'tf2_msgs/msg/TFMessage':
            msg = deserialize_message(data, TFMessage)
            tf_messages.extend(msg.transforms)

        elif topic == '/tf_static' and msg_type == 'tf2_msgs/msg/TFMessage':
            msg = deserialize_message(data, TFMessage)
            static_tf_messages.extend(msg.transforms)

    # Save transforms.json file if we have frame data
    if frames_data:
        # Use camera info if available, otherwise use defaults
        if camera_info is not None:
            K = camera_info.k
            D = camera_info.d
            
            fl_x = K[0]  # fx
            fl_y = K[4]  # fy
            cx = K[2]    # cx
            cy = K[5]    # cy
            w = camera_info.width
            h = camera_info.height
            
            # Extract distortion coefficients
            k1 = D[0] if len(D) > 0 else 0.0
            k2 = D[1] if len(D) > 1 else 0.0
            p1 = D[2] if len(D) > 2 else 0.0
            p2 = D[3] if len(D) > 3 else 0.0
        else:
            # Default values
            fl_x, fl_y = 525.0, 525.0
            cx, cy = 320.0, 240.0
            w, h = 640, 480
            k1, k2, p1, p2 = 0.0, 0.0, 0.0, 0.0
        
        transforms_data = {
            "camera_model": "OPENCV",
            "fl_x": fl_x,
            "fl_y": fl_y,
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "k1": k1,
            "k2": k2,
            "p1": p1,
            "p2": p2,
            "frames": frames_data
        }
        
        json_path = os.path.join(output_dir, 'transforms.json')
        with open(json_path, 'w') as f:
            json.dump(transforms_data, f, indent=2)
        
        print(f'[Dataset] Saved {len(frames_data)} frames to {json_path}')

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('bag_path', help='Path to the ROS2 bag directory')
    parser.add_argument('--image_topic', default='/rgb/image_raw')
    parser.add_argument('--camera_info_topic', default='/rgb/camera_info')
    parser.add_argument('--tf_src', default='base_link', help='Source frame')
    parser.add_argument('--tf_target', default='azure_rgb', help='Target frame')
    parser.add_argument('--output_dir', default='nerf_dataset')
    args = parser.parse_args()

    rclpy.init()
    read_rosbag(args.bag_path, args.image_topic, args.camera_info_topic, args.tf_src, args.tf_target, args.output_dir)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
