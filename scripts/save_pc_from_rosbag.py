#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
import open3d as o3d
import struct

def unpack_pointcloud2(msg: PointCloud2):
    # Determine point size and total number of points
    point_step = msg.point_step       # e.g., 32 bytes per point
    row_step = msg.row_step           # total bytes per row
    data = msg.data                   # binary data (bytes)
    num_points = len(data) // point_step

    # Allocate arrays
    xyz = np.zeros((num_points, 3), dtype=np.float32)
    rgb = np.zeros((num_points, 3), dtype=np.float32)

    for i in range(num_points):
        offset = i * point_step
        x, y, z, rgb_float = struct.unpack_from('fffxxxxf', data, offset)
        xyz[i] = [x, y, z]

        # Unpack float-encoded RGB
        rgb_int = struct.unpack('I', struct.pack('f', rgb_float))[0]
        r = (rgb_int >> 16) & 0x0000FF
        g = (rgb_int >> 8) & 0x0000FF
        b = rgb_int & 0x0000FF
        rgb[i] = [r, g, b]

    # Normalize RGB
    rgb /= 255.0
    return xyz, rgb

class PointCloudSaver(Node):
    def __init__(self):
        super().__init__('pointcloud_saver')
        self.declare_parameter('topic', '/points2')  # Default topic
        topic = self.get_parameter('topic').get_parameter_value().string_value
        self.subscription = self.create_subscription(
            PointCloud2,
            topic,
            self.listener_callback,
            1)  # Changed to 1 to only keep latest message
        self.subscription  # prevent unused var warning

    def listener_callback(self, msg: PointCloud2):
        self.get_logger().info("Received PointCloud2 message")


        points, colors = unpack_pointcloud2(msg)

        # Save with Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud("pointcloud.ply", pcd)

        self.get_logger().info("Saved point cloud to 'pointcloud.ply'")
        rclpy.shutdown()

        # Convert RGB values to float format (0-1 range)
  
        # Shutdown after saving one pointcloud
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudSaver()
    rclpy.spin(node)
    node.destroy_node()


if __name__ == '__main__':
    main()
