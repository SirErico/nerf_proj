"""
ROS2 Bag to NeRF Dataset Converter

Reads ROS2 bag files and extracts:
1. Camera images (every 5th frame) -> saves to images folder
2. TF transformations -> matches with images by timestamp -> saves to transforms.json

Usage:
    python read_bag.py --bag-path /path/to/bagfile --image-topic /rgb/image_raw 
                       --camera-frame azure_rgb --world-frame base_link --output-dir ./dataset
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
import cv2

# ROS2 imports
import rclpy
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R


class BagToNeRFConverter:
    def __init__(self, bag_path, image_topic, camera_frame, world_frame, output_dir):
        self.bag_path = bag_path
        self.image_topic = image_topic
        self.camera_frame = camera_frame
        self.world_frame = world_frame
        self.output_dir = Path(output_dir)
        self.frame_skip = 5
        
        # Create output directories
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Storage
        self.images = []          # List of saved images with timestamps
        self.dynamic_transforms = {}  # Dict: timestamp -> dynamic transform matrix
        self.static_transforms = {}   # Dict: parent_frame -> child_frame -> static transform matrix
        self.camera_info = None
        
    def read_bag(self):
        """Read bag file and extract images and transforms"""
        print(f"Reading bag: {self.bag_path}")
        
        storage_options = StorageOptions(uri=str(self.bag_path), storage_id='sqlite3')
        converter_options = ConverterOptions('', '')
        
        reader = SequentialReader()
        reader.open(storage_options, converter_options)
        
        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}
        
        print(f"Available topics: {list(type_map.keys())}")
        
        # First pass: Read all tf_static transforms (only once)
        print("Reading static transforms...")
        while reader.has_next():
            (topic, data, timestamp) = reader.read_next()
            if topic == '/tf_static':
                self._process_tf_static(data, timestamp, type_map[topic])
        
        reader.close()
        
        # Second pass: Read dynamic transforms and images
        print("Reading dynamic transforms and images...")
        reader = SequentialReader()
        reader.open(storage_options, converter_options)
        
        image_count = 0
        message_count = 0
        
        while reader.has_next():
            (topic, data, timestamp) = reader.read_next()
            message_count += 1
            
            # Always process dynamic TF messages
            if topic == '/tf':
                self._process_tf_dynamic(data, timestamp, type_map[topic])
            
            # Process every 5th message for images and camera_info
            elif message_count % self.frame_skip == 0:
                if topic == self.image_topic:
                    self._process_image(data, timestamp, type_map[topic], image_count)
                    image_count += 1
                elif 'camera_info' in topic:
                    self._process_camera_info(data, timestamp, type_map[topic])
        
        reader.close()
        
        print(f"Processed {len(self.images)} images")
        print(f"Found {len(self.static_transforms)} static transform chains")
        print(f"Found {len(self.dynamic_transforms)} dynamic transforms")
        
    def _process_image(self, data, timestamp, msg_type, image_count):
        """Save image and store its info"""
        msg_class = get_message(msg_type)
        msg = deserialize_message(data, msg_class)
        
        try:
            # Convert to OpenCV and save
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            image_filename = f"frame_{image_count:06d}.png"
            image_path = self.images_dir / image_filename
            cv2.imwrite(str(image_path), cv_image)
            
            # Store image info
            self.images.append({
                'filename': image_filename,
                'timestamp': timestamp,
                'index': image_count
            })
            
            if image_count % 10 == 0:
                print(f"Saved {image_count} images...")
                
        except Exception as e:
            print(f"Error processing image {image_count}: {e}")
    
    def _process_tf_static(self, data, msg_type):
        """Process static TF messages (read once)"""
        msg_class = get_message(msg_type)
        msg = deserialize_message(data, msg_class)
        
        # Handle both TFMessage and TransformStamped
        transforms_list = []
        if hasattr(msg, 'transforms'):
            transforms_list = msg.transforms
        else:
            transforms_list = [msg]
        
        for transform in transforms_list:
            parent_frame = transform.header.frame_id
            child_frame = transform.child_frame_id
            matrix = self._transform_to_matrix(transform)
            
            # Store static transforms
            if parent_frame not in self.static_transforms:
                self.static_transforms[parent_frame] = {}
            self.static_transforms[parent_frame][child_frame] = matrix
            print(f"Found static transform: {parent_frame} -> {child_frame}")
    
    def _process_tf_dynamic(self, data, timestamp, msg_type):
        """Process dynamic TF messages"""
        msg_class = get_message(msg_type)
        msg = deserialize_message(data, msg_class)
        
        # Handle both TFMessage and TransformStamped
        transforms_list = []
        if hasattr(msg, 'transforms'):
            transforms_list = msg.transforms
        else:
            transforms_list = [msg]
        
        for transform in transforms_list:
            parent_frame = transform.header.frame_id
            child_frame = transform.child_frame_id
            matrix = self._transform_to_matrix(transform)
            
            # Store dynamic transforms that lead to our camera chain
            if (parent_frame == self.world_frame or 
                child_frame == 'right_arm_wrist_3_link' or
                'right_arm' in child_frame):
                self.dynamic_transforms[timestamp] = {
                    'parent': parent_frame,
                    'child': child_frame,
                    'matrix': matrix
                }
                
    def _process_camera_info(self, data, timestamp, msg_type):
        """Extract camera intrinsics"""
        if self.camera_info is not None:
            return
            
        msg_class = get_message(msg_type)
        msg = deserialize_message(data, msg_class)
        
        self.camera_info = {
            'width': msg.width,
            'height': msg.height,
            'fx': msg.k[0],
            'fy': msg.k[4], 
            'cx': msg.k[2],
            'cy': msg.k[5]
        }
        print(f"Camera info: {self.camera_info}")
    
    def _transform_to_matrix(self, transform_msg):
        """Convert ROS transform to 4x4 matrix"""
        t = transform_msg.transform.translation
        r = transform_msg.transform.rotation
        
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
        
        return (matrix @ ros_to_nerf).tolist()
    
    def _find_transform_chain(self, image_timestamp):
        """Build complete transform chain from world to camera"""
        # Step 1: Find dynamic transform to right_arm_wrist_3_link at image timestamp
        if not self.dynamic_transforms:
            return None
            
        # Find closest dynamic transform by timestamp
        closest_timestamp = min(self.dynamic_transforms.keys(), 
                              key=lambda t: abs(t - image_timestamp))
        
        time_diff = abs(closest_timestamp - image_timestamp) / 1e9
        if time_diff > 0.1:  # 100ms tolerance
            return None
            
        # Get the dynamic transform
        dynamic_tf = self.dynamic_transforms[closest_timestamp]
        
        # Step 2: Find static transform chain: right_arm_wrist_3_link -> azure_rgb
        # Look for the static transform from wrist_3_link to camera
        static_chain = None
        if 'right_arm_wrist_3_link' in self.static_transforms:
            if self.camera_frame in self.static_transforms['right_arm_wrist_3_link']:
                static_chain = self.static_transforms['right_arm_wrist_3_link'][self.camera_frame]
        
        if static_chain is None:
            return None
            
        # Step 3: Combine transforms
        # Dynamic transform goes to right_arm_wrist_3_link, then static to azure_rgb
        dynamic_matrix = np.array(dynamic_tf['matrix'])
        static_matrix = np.array(static_chain)
        
        # Combine: world -> right_arm_wrist_3_link -> azure_rgb
        final_transform = dynamic_matrix @ static_matrix
        
        return final_transform.tolist()
    
    def save_transforms_json(self):
        """Create NeRF transforms.json file"""
        if not self.camera_info:
            print("Warning: No camera info found, using defaults")
            self.camera_info = {
                'width': 640, 'height': 480,
                'fx': 500.0, 'fy': 500.0,
                'cx': 320.0, 'cy': 240.0
            }
        
        # Match images with transforms
        frames = []
        successful_matches = 0
        
        for image_info in self.images:
            transform_matrix = self._find_transform_chain(image_info['timestamp'])
            
            if transform_matrix is not None:
                frame = {
                    "file_path": f"./images/{image_info['filename']}",
                    "transform_matrix": transform_matrix
                }
                frames.append(frame)
                successful_matches += 1
        
        # Create final JSON structure
        transforms_data = {
            "camera_model": "OPENCV",
            "fl_x": self.camera_info['fx'],
            "fl_y": self.camera_info['fy'],
            "cx": self.camera_info['cx'],
            "cy": self.camera_info['cy'],
            "w": self.camera_info['width'],
            "h": self.camera_info['height'],
            "frames": frames
        }
        
        # Save to file
        output_file = self.output_dir / "transforms.json"
        with open(output_file, 'w') as f:
            json.dump(transforms_data, f, indent=2)
        
        print(f"✓ Saved {successful_matches}/{len(self.images)} image-transform pairs to {output_file}")
        
        if successful_matches == 0:
            print("❌ No matching transforms found! Check frame names:")
            print(f"   Looking for chain: {self.world_frame} -> right_arm_wrist_3_link -> {self.camera_frame}")
            print(f"   Static transforms found: {len(self.static_transforms)}")
            print(f"   Dynamic transforms found: {len(self.dynamic_transforms)}")
            
            # Debug: show what static transforms we found
            for parent, children in self.static_transforms.items():
                for child in children:
                    print(f"     Static: {parent} -> {child}")


def main():
    parser = argparse.ArgumentParser(description="Convert ROS2 bag to NeRF dataset")
    parser.add_argument("--bag-path", required=True, help="Path to ROS2 bag file")
    parser.add_argument("--image-topic", required=True, help="Image topic (e.g., /rgb/image_raw)")
    parser.add_argument("--camera-frame", required=True, help="Camera frame ID (e.g., azure_rgb)")
    parser.add_argument("--world-frame", required=True, help="World frame ID (e.g., base_link)")
    parser.add_argument("--output-dir", default="./dataset", help="Output directory")
    
    args = parser.parse_args()
    
    rclpy.init()
    
    try:
        converter = BagToNeRFConverter(
            args.bag_path, args.image_topic, args.camera_frame, 
            args.world_frame, args.output_dir
        )
        
        converter.read_bag()
        converter.save_transforms_json()
        
        print(f"\n✓ Dataset created in: {args.output_dir}")
        
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
