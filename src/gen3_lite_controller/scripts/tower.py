#!/usr/bin/env python3

import cv2
import cv_bridge

import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import numpy as np
import rospy
import sensor_msgs.msg
from vgn.utils import ros_utils
import vgn.utils.transform
import image_geometry

import tf2_ros
import tf2_geometry_msgs
import tf.transformations

def predict_grasp_candidates(image, bw_publisher, image_publisher, visualize=True) -> "list[tuple[tuple[int, int], tuple[int, int]]]":
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.addWeighted(image, 2.0, blurred, -1.0, 0)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # detect red pixels
    lower_red = np.array([0, 50, 128])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    lower_red = np.array([170, 50, 128])
    upper_red = np.array([180, 255, 255])
    mask += cv2.inRange(hsv, lower_red, upper_red)
    # isolate the value channel
    value = hsv[:, :, 2]
    # mask the value channel to only include red pixels
    value = cv2.bitwise_and(value, mask)

    # blur and sharpen to simplify edges
    value = cv2.GaussianBlur(value, (7, 7), 0)
    value = cv2.addWeighted(value, 2.5, cv2.GaussianBlur(value, (0, 0), 2), -1.5, 0)

    # generate histogram of image brightness
    hist = cv2.calcHist([value], [0], None, [256], [0, 256])
    max_hist_idx = np.argmax(hist[1:])
    max_hist = hist[max_hist_idx]
    if max_hist == 0:
        return []

    thresh = 1
    while thresh <= max_hist_idx:
        moving_squared_avg = 0
        for i in range(10):
            moving_squared_avg += hist[min(255, thresh + i)] ** 2
        moving_squared_avg /= 10
        if moving_squared_avg > 0.1 * max_hist**2:
            break
        thresh += 1

    _, mask = cv2.threshold(value, thresh, 255, cv2.THRESH_BINARY)
    value = cv2.bitwise_and(value, mask)

    # erode the image
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    value = cv2.erode(value, kernel, iterations=1)

    contours, _ = cv2.findContours(value, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if visualize:
        rgb = cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)

        cv2.line(rgb, (255, 0), (255, 480), (0, 0, 255), 1)
        cv2.line(rgb, (thresh, 0), (thresh, 480), (255, 0, 0), 1)
        cv2.line(rgb, (thresh + 10, 0), (thresh + 10, 480), (255, 0, 0), 1)

        for x, y in enumerate(hist):
            cv2.line(rgb, (x, 480), (x, 480 - int(y[0] * 480 / max_hist)), (0, 255, 0), 1)

        cv2.drawContours(rgb, contours, -1, (255, 0, 0), 1)
        cv2.drawContours(image, contours, -1, (255, 0, 0), 1)

    grasps = []

    for contour in contours:
        area = cv2.contourArea(contour)
        AREA_OF_BLOCK = 2150
        area_blocks = area / AREA_OF_BLOCK

        if area_blocks < 0.5:
            continue

        # fit a polygon to the contour
        epsilon = 0.015 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        line_count = len(approx)

        for _retry in range(10):
            if area_blocks < 1.5 and line_count > 4:
                # this is meant to be a rectangle
                epsilon *= 1.1
            elif line_count < 4:
                epsilon *= 0.9
            elif line_count == 4:
                break
            elif line_count < 8:
                # this is meant to be an octagon
                epsilon *= 0.9
            elif line_count == 8:
                break
            elif line_count > 8:
                # maybe meant to be an octagon???
                epsilon *= 1.1
            approx = cv2.approxPolyDP(contour, epsilon, True)
            line_count = len(approx)

        

        if line_count == 4:
            side_vectors = []
            side_lengths = []
            for i in range(4):
                side_vectors.append(approx[(i + 1) % 4] - approx[i])
                side_lengths.append(np.linalg.norm(side_vectors[i]))
            # check if sides are parallel
            for i in range(2):
                j = i + 2
                parallelness = abs(
                    np.dot(side_vectors[i], side_vectors[j].transpose())
                    / (side_lengths[i] * side_lengths[j])
                )
                if parallelness < 0.95:
                    print("WARNING: sides are not parallel")

            # valid rectangle!
            if area_blocks > 1.75 and area_blocks < 2.25:
                longest_side_idx = np.argmax(side_lengths)
                opposite_side_idx = (longest_side_idx + 2) % 4
                grasp1_1 = approx[longest_side_idx] + side_vectors[longest_side_idx] // 4
                grasp1_2 = (
                    approx[opposite_side_idx] + 3 * side_vectors[opposite_side_idx] // 4
                )
                grasp2_1 = (
                    approx[longest_side_idx] + 3 * side_vectors[longest_side_idx] // 4
                )
                grasp2_2 = approx[opposite_side_idx] + side_vectors[opposite_side_idx] // 4
                grasps.append((grasp1_1, grasp1_2))
                grasps.append((grasp2_1, grasp2_2))
            elif area_blocks > 0.75 and area_blocks < 1.25:
                for i in range(2):
                    grasp_1 = approx[i] + side_vectors[i] // 2
                    grasp_2 = approx[i + 2] + side_vectors[i + 2] // 2
                    grasps.append((grasp_1, grasp_2))
            # draw the polygon
            if visualize:
                cv2.drawContours(rgb, [approx], -1, (0, 255, 0), 1)
                cv2.drawContours(image, [approx], -1, (0, 255, 0), 1)
        elif line_count == 8:
            # two blocks have been found. Find the closest pair of non-neighboring vertices and split the contour into two
            min_length = float("inf")
            min_idx = None
            for i in range(8):
                for j in range(i + 2, 8):
                    if i == 0 and j == 7:
                        continue
                    length = np.linalg.norm(approx[i] - approx[j])
                    if length < min_length:
                        min_length = length
                        min_idx = (i, j)
            len_upwards = np.linalg.norm(approx[min_idx[0]] - approx[(min_idx[0] + 1) % 8])
            len_downwards = np.linalg.norm(
                approx[min_idx[0]] - approx[(min_idx[0] - 1) % 8]
            )
            if visualize:
                for i in range(8):
                    x, y = approx[i][0]
                    # text
                    cv2.putText(
                        rgb, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                    )

            contour_1_indices = []
            contour_2_indices = []
            if len_upwards > len_downwards:
                # expand upwards
                for i in range(min_idx[0], min_idx[0] + 4):
                    contour_1_indices.append(i % 8)
                for i in range(min_idx[1], min_idx[1] + 4):
                    contour_2_indices.append(i % 8)
            else:
                # expand downwards
                for i in range(min_idx[0], min_idx[0] - 4, -1):
                    contour_1_indices.append(i % 8)
                for i in range(min_idx[1], min_idx[1] - 4, -1):
                    contour_2_indices.append(i % 8)

            contour1 = np.array([approx[i] for i in contour_1_indices])
            contour2 = np.array([approx[i] for i in contour_2_indices])
            if visualize:
                cv2.drawContours(rgb, [contour1, contour2], -1, (0, 255, 0), 1)
                cv2.drawContours(image, [contour1, contour2], -1, (0, 255, 0), 1)
            grasps.append(
                ((contour1[0] + contour1[1]) // 2, (contour1[2] + contour1[3]) // 2)
            )
            grasps.append(
                ((contour2[0] + contour2[1]) // 2, (contour2[2] + contour2[3]) // 2)
            )
            grasps.append(
                ((contour1[0] + contour1[3]) // 2, (contour1[1] + contour1[2]) // 2)
            )
            grasps.append(
                ((contour2[0] + contour2[3]) // 2, (contour2[1] + contour2[2]) // 2)
            )
        else:
            print("WARNING: invalid polygon detected")
            if visualize:
                cv2.drawContours(rgb, [approx], -1, (0, 0, 255), 1)
                cv2.drawContours(image, [approx], -1, (0, 0, 255), 1)

        if visualize:
            # draw text
            text = f"  area: {area_blocks:.2f}"
            x, y = approx[0][0]
            cv2.putText(rgb, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if visualize:
        # draw all the grasps
        for grasp in grasps:
            cv2.line(image, tuple(grasp[0][0]), tuple(grasp[1][0]), (0, 0, 255), 2)
            cv2.line(rgb, tuple(grasp[0][0]), tuple(grasp[1][0]), (0, 0, 255), 2)
        bw_publisher.publish(cv_bridge.CvBridge().cv2_to_imgmsg(rgb, encoding="bgr8"))
        image_publisher.publish(cv_bridge.CvBridge().cv2_to_imgmsg(image, encoding="bgr8"))
        
    return grasps


class Gen3LiteCommander(object):
    def __init__(self):
        rospy.logerr("Initializing node in namespace " + rospy.get_namespace())
        self._initialize_moveit()
        self._initialize_gripper()
        rospy.logerr("Successfully initialized Gen3LiteCommander")

    def _initialize_moveit(self):
        self.robot = moveit_commander.robot.RobotCommander("robot_description")
        self.scene = moveit_commander.planning_scene_interface.PlanningSceneInterface()
        pose = geometry_msgs.msg.PoseStamped()
        pose.header.frame_id = "base_link"
        pose.pose.position.z = -0.05
        self.scene.add_plane("table", pose, normal=(0, 0, 1))
        pose.pose.position.x = -0.4
        self.scene.add_box("shelf", pose, size=(0.1, 2, 2))
        pose.pose.position.x = 0
        pose.pose.position.y = 0.5
        self.scene.add_box("table_lip", pose, size=(1, 0.05, 0.2))
        self.arm_group = moveit_commander.move_group.MoveGroupCommander("arm")
        self.arm_group.set_max_acceleration_scaling_factor(0.5)
        self.gripper_group = moveit_commander.move_group.MoveGroupCommander("gripper")
        self.display_trajectory_publisher = rospy.Publisher('/my_gen3_lite/move_group/display_planned_path',
                                            moveit_msgs.msg.DisplayTrajectory,
                                            queue_size=20)
            
    def _initialize_gripper(self):
        pass

    def goto_joints(self, joints: "list[float]"):
        # TODO raise an exception if this fails
        if len(joints) != 6:
            raise ValueError("Invalid number of joints. Expected 6 and found "+str(len(joints)))
        arm_group = self.arm_group
        arm_group.set_joint_value_target(joints)
        _, plan, _, _ = arm_group.plan()
        success = arm_group.execute(plan, wait=True)
        return success
    
    def goto_pose(self, pose):
        if not isinstance(pose, geometry_msgs.msg.Pose):
            pose = ros_utils.to_pose_msg(pose)
        for i in range(5):
            self.arm_group.set_pose_target(pose)
            _, plan, _, _ = self.arm_group.plan()
            if self.arm_group.execute(plan, wait=True):
                return
            else:
                rospy.logwarn(f"Failed attempt {i+1}/5")
                rospy.sleep(0.5)
        raise Exception("Failed to reach target pose")



    def move_gripper(self, relative_position: float):
        if relative_position < 0 or relative_position > 1:
            raise ValueError("Relative position must be between 0 and 1")
        gripper_joint = self.robot.get_joint("right_finger_bottom_joint") # no clue why it's called this
        gripper_max_absolute_pos = gripper_joint.max_bound()
        gripper_min_absolute_pos = gripper_joint.min_bound()
        rospy.logerr("Gripper max bound: " + str(gripper_max_absolute_pos))
        rospy.logerr("Gripper min bound: " + str(gripper_min_absolute_pos))
        position = relative_position * (gripper_max_absolute_pos - gripper_min_absolute_pos) + gripper_min_absolute_pos
        rospy.logerr("Gripper position: " + str(position))
        try:
            val = gripper_joint.move(position, True)
            return val
        except Exception as e:
            rospy.logerr("An error occurred while moving the gripper: " + str(e))
            return False 
        
    def open_gripper(self):
        return self.move_gripper(0.9)
    
    def close_gripper(self):
        return self.move_gripper(0)

class TowerController(object):
    def __init__(self):
        self.gen3_lite = Gen3LiteCommander()
        self.bridge = cv_bridge.CvBridge()
        self.T_base_tower = vgn.utils.transform.Transform(vgn.utils.transform.Rotation.from_euler("xz", [np.pi, np.pi]), [-0.15, 0.25, 0.0])
        
        T_task_observation = vgn.utils.transform.Transform(vgn.utils.transform.Rotation.from_euler("xz", [np.pi, np.pi]), [0.15, 0.15, 0.3])
        self.T_base_pickup = vgn.utils.transform.Transform(vgn.utils.transform.Rotation.identity(), [-0.3, -0.4, -0.05])
        self.T_base_pickup_observation = self.T_base_pickup * T_task_observation
        
        self.camera_model = image_geometry.PinholeCameraModel()
        msg = rospy.wait_for_message("/camera/aligned_depth_to_color/camera_info", sensor_msgs.msg.CameraInfo)
        self.camera_model.fromCameraInfo(msg)

        self.tf_buffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tf_buffer)
        
        self.grasp_publisher = rospy.Publisher("/grasps/pose", geometry_msgs.msg.PoseArray, queue_size=10)

        self.bw_publisher = rospy.Publisher("/grasps/bw", sensor_msgs.msg.Image, queue_size=10)
        self.rgb_publisher = rospy.Publisher("/grasps/rgb", sensor_msgs.msg.Image, queue_size=10)

    def pickup_piece(self):
        self.gen3_lite.goto_pose(self.T_base_pickup_observation)
        rgb_image = rospy.wait_for_message("/camera/color/image_raw", sensor_msgs.msg.Image, timeout=5)
        depth_image = rospy.wait_for_message("/camera/aligned_depth_to_color/image_raw", sensor_msgs.msg.Image, timeout=5)
        cv_image = self.bridge.imgmsg_to_cv2(rgb_image, desired_encoding="bgr8")
        cv_depth_image = self.bridge.imgmsg_to_cv2(depth_image, desired_encoding="passthrough")
        pixel_2d_grasps = predict_grasp_candidates(cv_image, self.bw_publisher, self.rgb_publisher, visualize=True)
        # TODO filter based on collision
        pose_array = geometry_msgs.msg.PoseArray()
        pose_array.header.frame_id = "camera_color_optical_frame"
        # translate grasps from camera coordinates to base_link coordinates
        for coord1, coord2 in pixel_2d_grasps:
            coord1 = coord1[0]
            coord2 = coord2[0]
            center = [(coord1[0] + coord2[0]) // 2, (coord1[1] + coord2[1]) // 2]
            rospy.logerr(center)
            depth = cv_depth_image[center[1], center[0]] / 1000.0
            if depth == 0:
                rospy.logwarn("camera depth is 0 -- rejecting grasp candidate. TODO sample redundantly nearby.")
                continue
            unit_projection_vector = np.array(self.camera_model.projectPixelTo3dRay(center)).astype(np.float64)
            center_3d = np.array(unit_projection_vector) * depth / unit_projection_vector[2]
            rospy.logerr(f"center_3d: {center_3d}")
            unit_projection_vector = np.array(self.camera_model.projectPixelTo3dRay(coord1)).astype(np.float64)
            coord1_3d = unit_projection_vector * depth / unit_projection_vector[2]
            rospy.logerr(f"coord1_3d: {coord1_3d}")
            i_hat = np.array(coord1_3d - center_3d)
            i_hat /= np.linalg.norm(i_hat)
            i_hat[1] *= -1 # flip y axis. without this the rotation is mirrored.
            pose: geometry_msgs.msg.Pose = geometry_msgs.msg.Pose()
            pose.position.x = center_3d[0]
            pose.position.y = center_3d[1]
            pose.position.z = center_3d[2]
            rotation_matrix = np.identity(4)
            rotation_matrix[:3, :3] = np.array([
                i_hat,
                np.cross([0, 0, 1], i_hat),
                [0, 0, 1]
            ])
            quat = tf.transformations.quaternion_from_matrix(rotation_matrix)
            pose.orientation.x = quat[0]
            pose.orientation.y = quat[1]
            pose.orientation.z = quat[2]
            pose.orientation.w = quat[3]
            pose_array.poses.append(pose)

        self.grasp_publisher.publish(pose_array)
        if len(pose_array.poses) == 0:
            rospy.logerr("No valid grasps found")
            return
        
        # TODO this transformation is so scuffed
        pose_stamped = geometry_msgs.msg.PoseStamped()
        pose_stamped.header.frame_id = "camera_color_optical_frame"
        pose_stamped.header.stamp = rospy.Time(0)
        pose_stamped.pose = pose_array.poses[0]

        T_base_grasp: geometry_msgs.msg.PoseStamped = None

        while T_base_grasp is None:
            try:
                T_base_grasp = self.tf_buffer.transform(pose_stamped, "base_link", rospy.Duration(1))
            except Exception as e:
                rospy.logerr("Failed to lookup transform: " + str(e))
                return
            
        # offset
        T_base_grasp.pose.position.z += 0.02
        
        approach_pose = geometry_msgs.msg.Pose()
        approach_pose.position.x = T_base_grasp.pose.position.x
        approach_pose.position.y = T_base_grasp.pose.position.y
        approach_pose.position.z = T_base_grasp.pose.position.z + 0.1
        approach_pose.orientation = T_base_grasp.pose.orientation
            
        self.gen3_lite.goto_pose(approach_pose)
        self.gen3_lite.goto_pose(T_base_grasp.pose)
        self.gen3_lite.close_gripper()
        self.gen3_lite.goto_pose(approach_pose)

    def place_piece(self):
        approach_pose = geometry_msgs.msg.Pose()
        approach_pose.position.x = self.T_base_tower.translation[0]
        approach_pose.position.y = self.T_base_tower.translation[1]
        approach_pose.position.z = self.T_base_tower.translation[2] + 0.1
        quat = self.T_base_tower.rotation.as_quat()
        approach_pose.orientation.x = quat[0]
        approach_pose.orientation.y = quat[1]
        approach_pose.orientation.z = quat[2]
        approach_pose.orientation.w = quat[3]
        self.gen3_lite.goto_pose(approach_pose)
        self.gen3_lite.goto_pose(self.T_base_tower)
        self.gen3_lite.open_gripper()
        self.gen3_lite.goto_pose(approach_pose)
        self.T_base_tower.translation[2] += 0.030
            
def main():
    rospy.init_node("tower_controller")
    rospy.sleep(1)
    controller = TowerController()
    controller.gen3_lite.open_gripper()
    
    while not rospy.is_shutdown():
        controller.pickup_piece()
        controller.place_piece()


if __name__ == "__main__":
    main()
