#!/usr/bin/python3

import rospy
import geometry_msgs.msg
import sensor_msgs.msg
from sensor_msgs import point_cloud2
import struct
from dataclasses import dataclass
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
import multiprocessing
import colorsys


class GraspDetector:
    def __init__(self):
        # TODO poorly named
        self.pointcloud_topic = rospy.Subscriber("/camera/depth/color/points", sensor_msgs.msg.PointCloud2, self.pointcloud_topic_callback)
        self.publisher = rospy.Publisher("/grasps/pose", geometry_msgs.msg.PoseArray, queue_size=10)

    def pointcloud_topic_callback(self, pointcloud: sensor_msgs.msg.PointCloud2):
        pose_array = geometry_msgs.msg.PoseArray()
        pose_array.header.frame_id = pointcloud.header.frame_id

        red_points = []
        yellow_points = []
        green_points = []
        blue_points = []

        for (x, y, z, bgr) in point_cloud2.read_points(pointcloud):
            [b, g, r, _] = bytearray(struct.pack("f", bgr))

            h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)

            if (h < 0.05 or h > 0.95) and s > 0.5:
                red_points.append([x, y, z])
            elif 0.14 < h < 0.2 and s > 0.5:
                yellow_points.append([x, y, z])
            elif 0.38 < h < 0.5 and s > 0.3:
                green_points.append([x, y, z])
            elif 0.5 < h < 0.75 and s > 0.27 and v > 0.5:
                blue_points.append([x, y, z])
        
        red_points = np.array(red_points)
        yellow_points = np.array(yellow_points)
        green_points = np.array(green_points)
        blue_points = np.array(blue_points)

        if len(red_points) > 3000:
            rospy.logwarn("Found more than 3000 red points. Truncating.")
            red_points = red_points[:3000]
        if len(yellow_points) > 3000:
            rospy.logwarn("Found more than 3000 yellow points. Truncating.")
            yellow_points = yellow_points[:3000]
        if len(green_points) > 3000:
            rospy.logwarn("Found more than 3000 green points. Truncating.")
            green_points = green_points[:3000]
        if len(blue_points) > 3000:
            rospy.logwarn("Found more than 3000 blue points. Truncating.")
            blue_points = blue_points[:3000]

        rospy.loginfo("Found (%d, %d, %d, %d) points." % (len(red_points), len(yellow_points), len(green_points), len(blue_points)))

        def get_cluster_centers(points, threshold):
            if len(points) < 2:
                return []
            clusters = fcluster(linkage(points), threshold, criterion="distance")
            cluster_centers = []
            for cluster in np.unique(clusters):
                cluster_points = points[clusters == cluster]
                if len(cluster_points) < 100:
                    continue
                cluster_center = np.mean(cluster_points, axis=0)
                cluster_centers.append(cluster_center)
            return cluster_centers
        
        cluster_centers = get_cluster_centers(red_points, 0.01) # + get_cluster_centers(yellow_points, 0.01) + get_cluster_centers(green_points, 0.01) + get_cluster_centers(blue_points, 0.01)
        
        for x, y, z in cluster_centers:
            pose = geometry_msgs.msg.Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z
            pose_array.poses.append(pose)

        self.publisher.publish(pose_array)
        rospy.loginfo("Found %d objects." % len(pose_array.poses))

def main():
    rospy.init_node("grasp_detector")
    GraspDetector()
    rospy.loginfo("Grasp detection node is running.")
    rospy.spin()

if __name__ == "__main__":
    main()