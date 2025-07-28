"""
ROS2 Bag to NeRF Dataset Converter

This script reads ROS2 bag files and extracts:
1. Camera images from specified image topics -> saves to images folder
2. TF transformations between camera and world frames -> saves to transforms.json

The output format is compatible with NeRF Studio training pipelines.

Usage:
    python read_bag.py --bag-path /path/to/bagfile --image-topic /camera/image_raw 
                       --camera-frame camera_link --world-frame map --output-dir ./dataset

Requirements:
    - rclpy
    - rosbag2_py
    - cv_bridge
    - tf2_ros
    - sensor_msgs
    - geometry_msgs
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
import cv2
from datetime import datetime

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rosidl_runtime_py.utilities import get_message

# Message type imports
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge


class BagToNeRFConverter:
    def __init__(self, bag_path, image_topic, camera_frame, world_frame, output_dir):
        self.bag_path = bag_path
        self.image_topic = image_topic
        self.camera_frame = camera_frame
        self.world_frame = world_frame
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Storage for data
        self.transforms = []
        self.camera_info = None
        self.image_count = 0
        
    def read_bag(self):
        """Read the ROS2 bag file and extract data"""
        print(f"Reading bag: {self.bag_path}")
        
        # Set up bag reader
        storage_options = StorageOptions(uri=str(self.bag_path), storage_id='sqlite3')
        converter_options = ConverterOptions('', '')
        
        reader = SequentialReader()
        reader.open(storage_options, converter_options)
        
        # Get topic types and names
        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}
        
        print(f"Available topics: {list(type_map.keys())}")
        
        # Read messages
        while reader.has_next():
            (topic, data, timestamp) = reader.read_next()
            
            if topic == self.image_topic:
                self._process_image_message(data, timestamp, type_map[topic])
            elif topic == '/tf' or topic == '/tf_static':
                self._process_tf_message(data, timestamp, type_map[topic])
            elif topic == self.image_topic.replace('image_raw', 'camera_info'):
                self._process_camera_info_message(data, timestamp, type_map[topic])
        
        reader.close()
        print(f"Processed {self.image_count} images and {len(self.transforms)} transforms")
        
    def _process_image_message(self, data, timestamp, msg_type):
        """Process image messages and save as files"""
        msg_class = get_message(msg_type)
        msg = deserialize_message(data, msg_class)
        
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Save image
            image_filename = f"frame_{self.image_count:06d}.png"
            image_path = self.images_dir / image_filename
            cv2.imwrite(str(image_path), cv_image)
            
            # Store transform data with image reference
            transform_data = {
                'image_filename': image_filename,
                'timestamp': timestamp,
                'image_count': self.image_count
            }
            
            self.transforms.append(transform_data)
            self.image_count += 1
            
            if self.image_count % 10 == 0:
                print(f"Processed {self.image_count} images...")
                
        except Exception as e:
            print(f"Error processing image: {e}")
    
    def _process_tf_message(self, data, timestamp, msg_type):
        """Process TF messages to extract camera poses"""
        msg_class = get_message(msg_type)
        msg = deserialize_message(data, msg_class)
        
        # Handle both TFMessage and TransformStamped
        transforms_to_check = []
        if hasattr(msg, 'transforms'):  # TFMessage
            transforms_to_check = msg.transforms
        else:  # TransformStamped
            transforms_to_check = [msg]
        
        for transform in transforms_to_check:
            if (transform.header.frame_id == self.world_frame and 
                transform.child_frame_id == self.camera_frame):
                
                # Find corresponding image data
                for tf_data in self.transforms:
                    if abs(tf_data['timestamp'] - timestamp) < 100000000:  # 100ms tolerance
                        tf_data['transform'] = self._transform_to_matrix(transform)
                        break
    
    def _process_camera_info_message(self, data, timestamp, msg_type):
        """Process camera info messages to get intrinsics"""
        if self.camera_info is not None:
            return
            
        msg_class = get_message(msg_type)
        msg = deserialize_message(data, msg_class)
        
        self.camera_info = {
            'width': msg.width,
            'height': msg.height,
            'fx': msg.k[0],  # K[0,0]
            'fy': msg.k[4],  # K[1,1]
            'cx': msg.k[2],  # K[0,2]
            'cy': msg.k[5],  # K[1,2]
        }
        print(f"Camera info: {self.camera_info}")
    
    def _transform_to_matrix(self, transform_msg):
        """Convert ROS TransformStamped to 4x4 transformation matrix"""
        # Extract translation
        t = transform_msg.transform.translation
        translation = np.array([t.x, t.y, t.z])
        
        # Extract rotation (quaternion)
        r = transform_msg.transform.rotation
        quaternion = np.array([r.x, r.y, r.z, r.w])
        
        # Convert quaternion to rotation matrix
        from scipy.spatial.transform import Rotation as R
        rotation_matrix = R.from_quat(quaternion).as_matrix()
        
        # Create 4x4 transformation matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = translation
        
        # Convert from ROS coordinate system to NeRF coordinate system
        # ROS: +X forward, +Y left, +Z up
        # NeRF: +X right, +Y up, +Z backward (camera looks along -Z)
        ros_to_nerf = np.array([
            [0, -1, 0, 0],
            [0, 0, 1, 0], 
            [-1, 0, 0, 0],
            [0, 0, 0, 1]
        ])
        
        nerf_transform = transform_matrix @ ros_to_nerf
        
        return nerf_transform.tolist()
    
    def save_transforms_json(self):
        """Save transforms to NeRF-compatible JSON format"""
        if not self.camera_info:
            print("Warning: No camera info found, using default values")
            self.camera_info = {
                'width': 640,
                'height': 480,
                'fx': 500.0,
                'fy': 500.0,
                'cx': 320.0,
                'cy': 240.0,
            }
        
        # Filter transforms that have pose data
        valid_transforms = [t for t in self.transforms if 'transform' in t]
        
        if not valid_transforms:
            print("Warning: No valid transforms found!")
            return
        
        # Create NeRF transforms.json format
        nerf_data = {
            "camera_model": "OPENCV",
            "fl_x": self.camera_info['fx'],
            "fl_y": self.camera_info['fy'],
            "cx": self.camera_info['cx'],
            "cy": self.camera_info['cy'],
            "w": self.camera_info['width'],
            "h": self.camera_info['height'],
            "frames": []
        }
        
        for tf_data in valid_transforms:
            frame = {
                "file_path": f"./images/{tf_data['image_filename']}",
                "transform_matrix": tf_data['transform']
            }
            nerf_data["frames"].append(frame)
        
        # Save to file
        output_file = self.output_dir / "transforms.json"
        with open(output_file, 'w') as f:
            json.dump(nerf_data, f, indent=2)
        
        print(f"Saved {len(valid_transforms)} transforms to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert ROS2 bag to NeRF dataset")
    parser.add_argument("--bag-path", type=str, required=True,
                       help="Path to ROS2 bag file")
    parser.add_argument("--image-topic", type=str, required=True,
                       help="Image topic name (e.g., /camera/image_raw)")
    parser.add_argument("--camera-frame", type=str, required=True,
                       help="Camera frame ID (e.g., camera_link)")
    parser.add_argument("--world-frame", type=str, required=True,
                       help="World frame ID (e.g., map, odom)")
    parser.add_argument("--output-dir", type=str, default="./nerf_dataset",
                       help="Output directory for dataset")
    
    args = parser.parse_args()
    
    # Initialize ROS2 (required for message deserialization)
    rclpy.init()
    
    try:
        # Create converter and process bag
        converter = BagToNeRFConverter(
            args.bag_path,
            args.image_topic,
            args.camera_frame,
            args.world_frame,
            args.output_dir
        )
        
        converter.read_bag()
        converter.save_transforms_json()
        
        print(f"\nDataset created successfully in: {args.output_dir}")
        print(f"Images: {converter.images_dir}")
        print(f"Transforms: {converter.output_dir}/transforms.json")
        
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
