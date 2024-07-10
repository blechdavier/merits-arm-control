#!/usr/bin/env python3

import rospy
from std_srvs.srv import Empty
import sensor_msgs.msg
import cv2
from cv_bridge import CvBridge

class BlockDetector:
    def __init__(self):
        self.cv_bridge = CvBridge()
    def grasp_generation_callback(self, _):
        rospy.loginfo("Grasp detection service has been called.")
        rgb_image = rospy.wait_for_message('/camera/color/image_raw', sensor_msgs.msg.Image)
        depth_image = rospy.wait_for_message('/camera/aligned_depth_to_color/image_raw', sensor_msgs.msg.Image)
        # detect lines
        detector = cv2.ximgproc.createFastLineDetector()
        rgb_img = self.cv_bridge.imgmsg_to_cv2(rgb_image)
        gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        lines = detector.detect(gray_img)
        # show lines
        line_img = detector.drawSegments(rgb_img, lines)
        cv2.imwrite('contours.png', line_img)


def main():
    rospy.init_node('contour_detect_grasp_service')
    block_detector = BlockDetector()
    s = rospy.Service('contour_detect_grasp', Empty, block_detector.grasp_generation_callback)
    rospy.loginfo("Contour detect grasp service is running.")
    rospy.spin()

if __name__ == '__main__':
    main()