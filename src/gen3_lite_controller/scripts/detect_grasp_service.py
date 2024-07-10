#!/usr/bin/env python3

import rospy
import geometry_msgs.msg
import sensor_msgs.msg
import gen3_lite_controller.srv
import time
import cv2
from cv_bridge import CvBridge, CvBridgeError

class GraspDetector:
    def __init__(self):
        # TODO consider creating these subscribers only when the service is called to save resources ..... benchmarks needed
        self.bridge = CvBridge()
        self.rgb_topic = rospy.Subscriber('/camera/color/image_raw', sensor_msgs.msg.Image, self.rgb_topic_callback)
        self.depth_topic = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', sensor_msgs.msg.Image, self.depth_topic_callback)
        self.rgb_image: sensor_msgs.msg.Image = None
        self.depth_image: sensor_msgs.msg.Image = None
        self.rgb_timestamp = 0.0
        self.depth_timestamp = 0.0

    def rgb_topic_callback(self, img: sensor_msgs.msg.Image):
        self.rgb_image = img
        self.rgb_timestamp = time.time()

    def depth_topic_callback(self, img: sensor_msgs.msg.Image):
        self.depth_image = img
        self.depth_timestamp = time.time()
    
    def grasp_generation_callback(self, req):
        rospy.loginfo("Grasp detection service has been called.")
        # transform = geometry_msgs.msg.TransformStamped()
        # transform.header.frame_id = "camera_depth_frame"
        
        if self.rgb_timestamp + 1.0 < time.time() or self.depth_timestamp + 1.0 < time.time():
            rospy.logerr("Failed to generate transform: image data too old.")
            rospy.logerr("Timestamp: "+str(time.time())+", Depth timestamp: "+str(self.depth_timestamp)+", RGB timestamp: "+str(self.rgb_timestamp))
            return (False, 0, 0)
        
        rgb_image = self.bridge.imgmsg_to_cv2(self.rgb_image)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        depth_image = self.bridge.imgmsg_to_cv2(self.depth_image)

        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        red_lower = (0, 100, 100)
        red_upper = (10, 255, 255)
        mask = cv2.inRange(hsv, red_lower, red_upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            center_x, center_y = x + w // 2, y + h // 2
            target_x, target_y = 385, 319 # offset a bit in both axes to center the target
            return (True, center_x - target_x, center_y - target_y)

        else:
            rospy.logerr("Failed to generate transform: no contour found.")
            return (False, 0, 0)

def main():
    grasp_detector = GraspDetector()
    rospy.init_node("grasp_detector")
    s = rospy.Service("generate_grasp", gen3_lite_controller.srv.GraspGeneration, grasp_detector.grasp_generation_callback)
    rospy.loginfo("Grasp detection service is running.")
    rospy.spin()

if __name__ == "__main__":
    main()