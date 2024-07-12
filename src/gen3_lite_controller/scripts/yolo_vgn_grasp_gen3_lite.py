#!/usr/bin/env python3

"""
Open-loop grasp execution using a Kinova Gen3 Lite arm and wrist-mounted RealSense camera.
"""

from pathlib import Path

import cv2

import cv_bridge
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import numpy as np
import rospy
import sensor_msgs.msg

import roslib.packages

from vgn import vis
from vgn.experiments.clutter_removal import State
from vgn.detection import VGN
from vgn.grasp import Grasp
from vgn.perception import *
from vgn.utils import ros_utils
from vgn.utils.transform import Rotation, Transform

from ultralytics_ros.msg import YoloResult

names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

class Gen3LiteCommander(object):
    def __init__(self):
        rospy.loginfo("Initializing node in namespace " + rospy.get_namespace())
        self._initialize_moveit()
        self._initialize_gripper()
        rospy.loginfo("Successfully initialized Gen3LiteCommander")

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
    
    def goto_pose(self, pose: geometry_msgs.msg.Transform):
        # TODO raise an exception if this fails
        pose_msg = ros_utils.to_pose_msg(pose)
        for i in range(5):
            self.arm_group.set_pose_target(pose_msg)
            _, plan, _, _ = self.arm_group.plan()
            if self.arm_group.execute(plan, wait=True):
                return
            else:
                rospy.logwarn(f"Failed attempt {i+1}/5")
                rospy.sleep(0.5)
        rospy.logerr("Failed to reach pose")



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

class Gen3LiteGraspController(object):
    def __init__(self):
        self.robot_error = False

        # params defined in yaml file and loaded in launch file
        self.base_frame_id = rospy.get_param("/vgn_grasp_gen3_lite/base_frame_id")
        self.T_tool0_tcp = Transform.from_dict(rospy.get_param("/vgn_grasp_gen3_lite/T_tool0_tcp"))  # TODO what even is this
        self.T_tcp_tool0 = self.T_tool0_tcp.inverse()
        self.finger_depth = rospy.get_param("/vgn_grasp_gen3_lite/finger_depth")
        self.size = 6.0 * self.finger_depth
        self.scan_joints_left = rospy.get_param("/vgn_grasp_gen3_lite/scan_joints_left")
        self.scan_joints_right = rospy.get_param("/vgn_grasp_gen3_lite/scan_joints_right")

        self.commander = Gen3LiteCommander()
        self.T_base_task_right = Transform(Rotation.identity(), [-0.3, 0.1, -0.05])
        self.T_base_task_left = Transform(Rotation.identity(), [-0.3, -0.4, -0.05])
        self.tf_tree = ros_utils.TransformTree()
        self.set_side("right")

        path = roslib.packages.get_pkg_dir("vgn")
        self.plan_grasps = VGN(Path(path+"/data/models/vgn_conv.pth"), rviz=True)
        # read objects from yaml
        self.objects = rospy.get_param("/vgn_grasp_gen3_lite/deposit_points")
        self.tsdf_server = TSDFServer(self.objects.keys()) # pass in the names
        rospy.loginfo("resetting gripper")
        self.commander.open_gripper()
        rospy.loginfo("ready")

    def set_side(self, side: str):
        if side == "right":
            self._side = "right"
            self.T_base_task = self.T_base_task_right
            self.tf_tree.broadcast_static(self.T_base_task, self.base_frame_id, "task")
        elif side == "left":
            self._side = "left"
            self.T_base_task = self.T_base_task_left
            self.tf_tree.broadcast_static(self.T_base_task, self.base_frame_id, "task")
        else:
            raise ValueError("Side must be 'right' or 'left")

    def run(self):
        vis.clear()

        tsdf, pc = self.acquire_tsdf()
        # vis.draw_tsdf(tsdf.get_grid().squeeze(), tsdf.voxel_size)
        # vis.draw_tsdf(self.tsdf_server.tsdfs["mouse"].get_grid().squeeze(), self.tsdf_server.tsdfs["mouse"].voxel_size)
        vis.draw_points(np.asarray(pc.points))
        rospy.loginfo("Reconstructed scene")

        state = State(tsdf, pc)
        grasps, scores, planning_time = self.plan_grasps(state)
        vis.draw_grasps(grasps, scores, self.finger_depth)
        rospy.loginfo("Planned grasps")

        if len(grasps) == 0:
            rospy.loginfo("No grasps detected")
            return

        grasp, score, name = self.select_grasp(grasps, scores)
        vis.draw_grasp(grasp, score, self.finger_depth)
        rospy.loginfo("Selected grasp")

        self.execute_grasp(grasp)

        # drop at the correct location
        angles = self.objects[name]
        self.commander.goto_joints(angles)
        self.commander.open_gripper()

    def acquire_tsdf(self):

        self.tsdf_server.reset()
        self.tsdf_server.integrate = True

        if self._side == "right":
            for pose in self.scan_joints_right:
                self.commander.goto_joints(pose)
            # while True:
            #     rospy.sleep(5)
        else:
            for pose in self.scan_joints_left:
                self.commander.goto_joints(pose)

        self.tsdf_server.integrate = False
        tsdf = self.tsdf_server.low_res_tsdf
        pc = self.tsdf_server.high_res_tsdf.get_cloud()

        rospy.loginfo("Pointcloud had %d points", len(pc.points))

        return tsdf, pc

    def select_grasp(self, grasps, scores) -> str:
        tsdfs = self.tsdf_server.tsdfs
        grids = [(classification, tsdf.get_grid().squeeze()) for classification, tsdf in tsdfs.items()]

        # select the highest grasp with a classification TODO this is too much nesting
        max_height = 0
        selected_grasp, selected_score, selected_name = None, None, None
        for grasp in grasps:
            height = grasp.pose.translation[2]
            if height > max_height:
                name = None
                max_score = 0
                for classification, grid in grids:
                    # sample the tsdf at the grasp pose
                    amount_down_finger = 0.75 # world's most poorly named variable: 0.0 is the palm, 1.0 is the fingertip
                    fingertip_pose = grasp.pose * Transform(Rotation.identity(), [0.0, 0.0, self.finger_depth * amount_down_finger])
                    x, y, z = fingertip_pose.translation / tsdfs[classification].voxel_size
                    x, y, z = int(x), int(y), int(z)
                    if grid[x, y, z] > max_score:
                        max_score = grid[x, y, z]
                        name = classification
                if name is not None:
                    max_height = height
                    selected_grasp = grasp
                    selected_score = max_score
                    selected_name = name

        # make sure camera is pointing forward
        rot = grasp.pose.rotation
        axis = rot.as_matrix()[:, 0]
        if axis[0] < 0:
            grasp.pose.rotation = rot * Rotation.from_euler("z", np.pi)

        return selected_grasp, selected_score, selected_name 

    def execute_grasp(self, grasp: Grasp):
        T_task_grasp = grasp.pose
        T_base_grasp = self.T_base_task * T_task_grasp


        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_base_pregrasp = T_base_grasp * T_grasp_pregrasp
        T_base_retreat = T_base_grasp * T_grasp_retreat

        self.commander.goto_pose(T_base_pregrasp * self.T_tcp_tool0)
        self.approach_grasp(T_base_grasp)

        self.commander.close_gripper()

        # lift hand
        T_retreat_lift_base = Transform(Rotation.identity(), [0.0, 0.0, 0.1])
        T_base_lift = T_retreat_lift_base * T_base_retreat
        self.commander.goto_pose(T_base_lift * self.T_tcp_tool0)

    def approach_grasp(self, T_base_grasp):
        target_pose = T_base_grasp * self.T_tcp_tool0
        # move any grips that are below the table up to the table while also preserving alignment
        k_hat = target_pose.rotation.as_matrix()[:, 2]
        rospy.logerr("k_hat: " + str(k_hat))
        z = target_pose.translation[2]
        MIN_Z = -0.01
        rospy.logerr("z: " + str(z))
        if z < MIN_Z:
            k_hat_z = k_hat[2]
            change_vector = (MIN_Z - z) * k_hat / k_hat_z
            rospy.logerr("change_vector: " + str(change_vector))
            target_pose.translation += change_vector
        rospy.logerr("target_pose.translation: " + str(target_pose.translation))
        self.commander.goto_pose(target_pose) 


class TSDFServer(object):
    def __init__(self, classifications: "list[str]"):
        self.aligned_cam_frame_id = rospy.get_param("/vgn_grasp_gen3_lite/aligned_cam/frame_id")
        self.aligned_cam_topic_name = rospy.get_param("/vgn_grasp_gen3_lite/aligned_cam/topic_name")
        self.aligned_cam_intrinsic = CameraIntrinsic.from_dict(rospy.get_param("/vgn_grasp_gen3_lite/aligned_cam/intrinsic"))
        self.size = 6.0 * rospy.get_param("/vgn_grasp_gen3_lite/finger_depth")
        self.classifications = classifications
        self.tsdfs = {}

        self.cv_bridge = cv_bridge.CvBridge()
        self.tf_tree = ros_utils.TransformTree()
        self.integrate = False
        self.most_recent_depth = None
        rospy.Subscriber(self.aligned_cam_topic_name, sensor_msgs.msg.Image, self.sensor_cb)
        rospy.Subscriber("/yolo_result", YoloResult, self.yolo_result_cb)
        self.mouse_depth_mask_pub = rospy.Publisher("/mouse_depth_mask/image_raw", sensor_msgs.msg.Image, queue_size=1)
        # self.camera_info_pub = rospy.Publisher("/mouse_depth_mask/camera_info", sensor_msgs.msg.CameraInfo, queue_size=1)
        # rospy.Subscriber("/camera/aligned_depth_to_color/camera_info", sensor_msgs.msg.CameraInfo, lambda msg: self.camera_info_pub.publish(msg))


    def reset(self):
        self.low_res_tsdf = TSDFVolume(self.size, 40)
        self.high_res_tsdf = TSDFVolume(self.size, 120)
        for classification in self.classifications:
            self.tsdfs[classification] = TSDFVolume(self.size, 40)

    def sensor_cb(self, msg):
        self.most_recent_depth = msg
        if not self.integrate:
            return
        
        rospy.loginfo("Integrating image.")

        img = self.cv_bridge.imgmsg_to_cv2(msg).astype(np.float32) * 0.001
        T_cam_task = self.tf_tree.lookup(
            self.aligned_cam_frame_id, "task", msg.header.stamp, rospy.Duration(0.1)
        )

        self.low_res_tsdf.integrate(img, self.aligned_cam_intrinsic, T_cam_task)
        self.high_res_tsdf.integrate(img, self.aligned_cam_intrinsic, T_cam_task)

        vis.draw_tsdf(self.low_res_tsdf.get_grid().squeeze(), self.low_res_tsdf.voxel_size)

    def yolo_result_cb(self, msg: YoloResult):
        if not self.integrate:
            return
        
        if self.most_recent_depth is None:
            rospy.logwarn("No depth image available. Skipping integration of YOLO result.")
            return
        
        detections = []
        for detection in msg.detections.detections:
            for result in detection.results:
                class_id = result.id
                detections.append(names[class_id])
        

        # merge masks
        masks = {}
        for i, mask in enumerate(msg.masks):
            name = detections[i]
            if name not in self.tsdfs:
                continue
            if name in masks:
                masks[name] += self.cv_bridge.imgmsg_to_cv2(mask, desired_encoding="mono8")
            else:
                masks[name] = self.cv_bridge.imgmsg_to_cv2(mask, desired_encoding="mono8")
        
        # apply masks and integrate
        cv_depth_image = self.cv_bridge.imgmsg_to_cv2(self.most_recent_depth).astype(np.float32) * 0.001
        for name, mask in masks.items():
            mask = cv2.erode(mask, np.ones((20, 20), np.uint8), iterations=1)
            masked_depth_image = cv2.bitwise_and(cv_depth_image, cv_depth_image, mask=mask)
            T_cam_task = self.tf_tree.lookup(
                self.aligned_cam_frame_id, "task", msg.header.stamp, rospy.Duration(0.1)
            )
            tsdf = self.tsdfs[name]
            tsdf.integrate(masked_depth_image, self.aligned_cam_intrinsic, T_cam_task)
            if name == "mouse":
                imgmsg = self.cv_bridge.cv2_to_imgmsg(masked_depth_image)
                imgmsg.header = self.most_recent_depth.header
                self.mouse_depth_mask_pub.publish(imgmsg)
            
        # vis.draw_tsdf(self.tsdfs["mouse"].get_grid().squeeze(), self.tsdfs["mouse"].voxel_size)




def main():
    rospy.init_node("vgn_grasp_gen3_lite")
    controller = Gen3LiteGraspController()

    while not rospy.is_shutdown():
        # controller.set_side("right")
        controller.run()
        # controller.set_side("left")
        # controller.run()


if __name__ == "__main__":
    main()
