#!/usr/bin/env python3

import rospy
import sensor_msgs.msg
import cv_bridge
import cv2

if __name__ == "__main__":
    rospy.init_node("record_images")
    rospy.loginfo("Recording images")
    while not rospy.is_shutdown():
        input("Press Enter to record an image")
        img = rospy.wait_for_message("/camera/color/image_raw", sensor_msgs.msg.Image, timeout=5)
        timestamp = img.header.stamp
        cv_img = cv_bridge.CvBridge().imgmsg_to_cv2(img, desired_encoding="bgr8")
        rospy.loginfo(f"Received image at {timestamp}")
        with open(f"/home/rover/training_imgs/image_{timestamp}.bmp", "wb") as f:
            f.write(cv2.imencode(".bmp", cv_img)[1].tobytes())