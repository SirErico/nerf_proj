#!/usr/bin/env python3

"""
ROS2 TF Transform Listener for NeRF Dataset Generation

Real-time TF transform listener that monitors transforms from base_link to azure_rgb
and can be used to capture camera poses for NeRF training datasets.
"""

import sys
import math
import json
import os
from datetime import datetime
import numpy as np
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


class TfToNeRFListener(Node):
    
    def __init__(self, world_frame='base_link', camera_frame='azure_rgb', output_dir='nerf_dataset'):
        super().__init__('tf_to_nerf_listener')
        
        self.world_frame = world_frame
        self.camera_frame = camera_frame
        self.output_dir = output_dir
        
        # Create output directories
        self.images_dir = os.path.join(output_dir, 'images')
        os.makedirs(self.images_dir, exist_ok=True)
        
        self.get_logger().info(f"Monitoring transform from {self.world_frame} to {self.camera_frame}")
        self.get_logger().info(f"Saving dataset to: {self.output_dir}")
        
        # Initialize TF buffer and listener
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        
        # Initialize CV bridge for image conversion
        self.bridge = CvBridge()
        
        # QoS profile for image subscription
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribe to camera image topic
        self.image_subscription = self.create_subscription(
            Image,
            '/rgb/image_raw',  # Updated topic name
            self.image_callback,
            qos_profile
        )
        
        # Subscribe to camera info topic
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            '/rgb/camera_info',
            self.camera_info_callback,
            qos_profile
        )
        
        # Storage for transforms and dataset
        self.transform_count = 0
        self.saved_frame_count = 0
        self.last_transform = None
        self.last_image = None
        self.camera_info = None
        self.frames_data = []
        
        # Timer to save frames at a lower rate (e.g., 2 Hz)
        self.save_timer = self.create_timer(0.5, self.save_frame_callback)  # 2 Hz = 0.5s
        
        # Timer to check transforms at 10 Hz for monitoring
        self.monitor_timer = self.create_timer(0.1, self.monitor_callback)  # 10 Hz = 0.1s
        
    def image_callback(self, msg):
        """Callback for image messages"""
        self.last_image = msg
    
    def camera_info_callback(self, msg):
        """Callback for camera info messages"""
        self.camera_info = msg
        if self.saved_frame_count == 0:  # Log only once
            self.get_logger().info(f"Received camera info: {msg.width}x{msg.height}, fx={msg.k[0]:.1f}, fy={msg.k[4]:.1f}")
    
    def monitor_callback(self):
        """Callback to monitor transforms (no logging)"""
        try:
            # Lookup transform from world to camera
            trans = self._tf_buffer.lookup_transform(
                self.world_frame, 
                self.camera_frame, 
                rclpy.time.Time()  # Latest available
            )
            
            self.transform_count += 1
            self.last_transform = trans
            
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            # Only log errors occasionally to avoid spam
            if self.transform_count % 100 == 0:
                self.get_logger().warn(f'Failed to get transform: {repr(e)}')
    
    def save_frame_callback(self):
        """Callback to save image and transform data"""
        if self.last_image is None or self.last_transform is None:
            return
            
        try:
            # Get current transform
            trans = self._tf_buffer.lookup_transform(
                self.world_frame, 
                self.camera_frame, 
                rclpy.time.Time()  # Latest available
            )
            
            # Convert image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(self.last_image, "bgr8")
            
            # Save image
            frame_filename = f"frame_{self.saved_frame_count:05d}.jpg"
            image_path = os.path.join(self.images_dir, frame_filename)
            cv2.imwrite(image_path, cv_image)
            
            # Convert transform to NeRF matrix
            nerf_matrix = self.transform_to_nerf_matrix(trans)
            
            # Create frame data
            frame_data = {
                "file_path": f"images/{frame_filename}",
                "transform_matrix": nerf_matrix
            }
            
            self.frames_data.append(frame_data)
            self.saved_frame_count += 1
            
            # Save transforms.json file
            self.save_transforms_json()
            
            # Only log every 10th saved frame to reduce output
            if self.saved_frame_count % 10 == 0:
                self.get_logger().info(f"Saved {self.saved_frame_count} frames...")
            
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f'Failed to save frame: {repr(e)}')
        except Exception as e:
            self.get_logger().error(f'Error saving frame: {repr(e)}')
    
    def save_transforms_json(self):
        """Save the transforms.json file in nerfstudio format"""
        # Use camera info if available, otherwise use defaults
        if self.camera_info is not None:
            # Extract camera parameters from CameraInfo message
            K = self.camera_info.k  # 3x3 camera matrix as flat array
            D = self.camera_info.d  # distortion coefficients
            
            fl_x = K[0]  # fx
            fl_y = K[4]  # fy
            cx = K[2]    # cx
            cy = K[5]    # cy
            w = self.camera_info.width
            h = self.camera_info.height
            
            # Extract distortion coefficients (pad with zeros if needed)
            k1 = D[0] if len(D) > 0 else 0.0
            k2 = D[1] if len(D) > 1 else 0.0
            p1 = D[2] if len(D) > 2 else 0.0
            p2 = D[3] if len(D) > 3 else 0.0
        else:
            # Default values if no camera info available
            fl_x, fl_y = 525.0, 525.0
            cx, cy = 320.0, 240.0
            w, h = 640, 480
            k1, k2, p1, p2 = 0.0, 0.0, 0.0, 0.0
        
        transforms_data = {
            "camera_model": "OPENCV",  # camera model type [OPENCV, OPENCV_FISHEYE]
            "fl_x": fl_x,  # focal length x
            "fl_y": fl_y,  # focal length y
            "cx": cx,      # principal point x
            "cy": cy,      # principal point y
            "w": w,        # image width
            "h": h,        # image height
            "k1": k1,      # first radial distortion parameter
            "k2": k2,      # second radial distortion parameter
            "p1": p1,      # first tangential distortion parameter
            "p2": p2,      # second tangential distortion parameter
            "frames": self.frames_data
        }
        
        json_path = os.path.join(self.output_dir, 'transforms.json')
        with open(json_path, 'w') as f:
            json.dump(transforms_data, f, indent=2)
        
        # ====
    def timer_callback(self):
        """Callback to lookup and process transforms"""
        try:
            # Lookup transform from world to camera
            trans = self._tf_buffer.lookup_transform(
                self.world_frame, 
                self.camera_frame, 
                rclpy.time.Time()  # Latest available
            )
            
            self.transform_count += 1
            self.last_transform = trans
            
            # Process and display the transform
            self.process_transform(trans)
            
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f'Failed to get transform: {repr(e)}')
    
    
    def process_transform_for_logging(self, transform):
        """Process the transform for logging purposes"""
        t = transform.transform.translation
        r = transform.transform.rotation
        
        # Convert timestamp to seconds and nanoseconds
        timestamp_sec = transform.header.stamp.sec
        timestamp_nanosec = transform.header.stamp.nanosec
        timestamp_float = timestamp_sec + timestamp_nanosec / 1e9
        
        self.get_logger().info(f"Transform #{self.transform_count}:")
        self.get_logger().info(f"  Timestamp: {timestamp_sec}.{timestamp_nanosec:09d} ({timestamp_float:.9f})")
        self.get_logger().info(f"  Translation: x={t.x:.3f}, y={t.y:.3f}, z={t.z:.3f}")
        self.get_logger().info(f"  Rotation: x={r.x:.3f}, y={r.y:.3f}, z={r.z:.3f}, w={r.w:.3f}")
        self.get_logger().info(f"  Saved frames: {self.saved_frame_count}")
        
        # Convert to NeRF format matrix
        nerf_matrix = self.transform_to_nerf_matrix(transform)
        self.get_logger().info("  NeRF Transform Matrix:")
        for i, row in enumerate(nerf_matrix):
            row_str = '[' + ', '.join(f'{x:8.3f}' for x in row) + ']'
            self.get_logger().info(f"    {row_str}")

    def process_transform(self, transform):
        """Process the transform and convert to NeRF format"""
        t = transform.transform.translation
        r = transform.transform.rotation
        
        # Log every 50th transform to avoid spam
        if self.transform_count % 50 == 0:
            # Convert timestamp to seconds and nanoseconds
            timestamp_sec = transform.header.stamp.sec
            timestamp_nanosec = transform.header.stamp.nanosec
            timestamp_float = timestamp_sec + timestamp_nanosec / 1e9
            
            self.get_logger().info(f"Transform #{self.transform_count}:")
            self.get_logger().info(f"  Timestamp: {timestamp_sec}.{timestamp_nanosec:09d} ({timestamp_float:.9f})")
            self.get_logger().info(f"  Translation: x={t.x:.3f}, y={t.y:.3f}, z={t.z:.3f}")
            self.get_logger().info(f"  Rotation: x={r.x:.3f}, y={r.y:.3f}, z={r.z:.3f}, w={r.w:.3f}")
            
            # Convert to NeRF format matrix
            nerf_matrix = self.transform_to_nerf_matrix(transform)
            self.get_logger().info("  NeRF Transform Matrix:")
            for i, row in enumerate(nerf_matrix):
                row_str = '[' + ', '.join(f'{x:8.3f}' for x in row) + ']'
                self.get_logger().info(f"    {row_str}")
    
    # =====

    def transform_to_nerf_matrix(self, transform_stamped):
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
    
    def get_current_transform_matrix(self):
        """Get the latest transform as NeRF matrix (for external use)"""
        if self.last_transform is None:
            return None
        return self.transform_to_nerf_matrix(self.last_transform)
    
    def get_transform_count(self):
        """Get the number of transforms processed"""
        return self.transform_count
    
    def get_saved_frame_count(self):
        """Get the number of frames saved"""
        return self.saved_frame_count
    
    def cleanup_and_save_final(self):
        """Final cleanup and save operations"""
        if self.frames_data:
            self.save_transforms_json()
            self.get_logger().info(f"Final dataset saved with {self.saved_frame_count} frames")
            self.get_logger().info(f"Dataset location: {self.output_dir}")
            self.get_logger().info(f"Images saved to: {self.images_dir}")


def main(argv=sys.argv):
    rclpy.init(args=argv)
    
    # Parse command line arguments
    world_frame = 'base_link'
    camera_frame = 'azure_rgb'
    output_dir = 'nerf_dataset'
    
    if len(argv) > 1:
        world_frame = argv[1]
    if len(argv) > 2:
        camera_frame = argv[2]
    if len(argv) > 3:
        output_dir = argv[3]
    
    # Create and run the node
    node = TfToNeRFListener(world_frame, camera_frame, output_dir)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
        node.get_logger().info(f"Total transforms processed: {node.get_transform_count()}")
        node.get_logger().info(f"Total frames saved: {node.get_saved_frame_count()}")
        node.cleanup_and_save_final()
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

