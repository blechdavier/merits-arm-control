#!/usr/bin/env python3

"""
Open-loop grasp execution using a Kinova Gen3 Lite arm and wrist-mounted RealSense camera.
"""

import argparse
from pathlib import Path

import cv_bridge
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import numpy as np
import rospy
import sensor_msgs.msg

from vgn import vis
from vgn.experiments.clutter_removal import State
from vgn.detection import VGN
from vgn.perception import *
from vgn.utils import ros_utils
from vgn.utils.transform import Rotation, Transform

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
        self.gripper_group = moveit_commander.move_group.MoveGroupCommander("gripper")
        self.display_trajectory_publisher = rospy.Publisher('/my_gen3_lite/move_group/display_planned_path',
                                                    moveit_msgs.msg.DisplayTrajectory,
                                                    queue_size=20)
    
    def _initialize_gripper(self):
        pass

    def home(self):
        # TODO joint angles
        self.goto_joints([2.1556243716474253, -0.42662136239971726, 0.9317636326954402, -1.5728107418434325, -1.7804819131189396, 2.1533431078576633])

    def goto_joints(self, joints: "list[float]"):
        if len(joints) != 6:
            raise ValueError("Invalid number of joints. Expected 6 and found "+str(len(joints)))
        arm_group = self.arm_group
        arm_group.set_joint_value_target(joints)
        _, plan, _, _ = arm_group.plan()
        success = arm_group.execute(plan, wait=True)
        return success
    
    def goto_pose(self, pose: geometry_msgs.msg.Transform):
        pose_msg = ros_utils.to_pose_msg(pose)
        self.arm_group.set_pose_target(pose_msg)
        _, plan, _, _ = self.arm_group.plan()
        success = self.arm_group.execute(plan, wait=True)
        return success

    def move_gripper(self, relative_position: float):
        if relative_position < 0 or relative_position > 1:
            raise ValueError("Relative position must be between 0 and 1")
        gripper_joint = self.robot.get_joint("right_finger_bottom_joint") # lol idk why it's called this
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
        self.tool0_frame_id = rospy.get_param("/vgn_grasp_gen3_lite/tool0_frame_id")
        self.T_tool0_tcp = Transform.from_dict(rospy.get_param("/vgn_grasp_gen3_lite/T_tool0_tcp"))  # TODO what even is this
        self.T_tcp_tool0 = self.T_tool0_tcp.inverse()
        self.finger_depth = rospy.get_param("/vgn_grasp_gen3_lite/finger_depth")
        self.size = 6.0 * self.finger_depth
        self.scan_joints = rospy.get_param("/vgn_grasp_gen3_lite/scan_joints")

        self.setup_panda_control()
        self.T_base_task = Transform(Rotation.identity(), [-0.3, 0.1, -0.05])
        self.tf_tree = ros_utils.TransformTree()
        self.tf_tree.broadcast_static(self.T_base_task, "base_link", "task")

        self.tsdf_server = TSDFServer()
        self.plan_grasps = VGN(Path("/home/rover/ros_ws/src/vgn/data/models/vgn_conv.pth"), rviz=True) #FIXME hard coded path

        rospy.loginfo("Ready to take action")

    def setup_panda_control(self):
        # TODO error handling
        # rospy.Subscriber(
        #     "/franka_state_controller/franka_states",
        #     franka_msgs.msg.FrankaState,
        #     self.robot_state_cb,
        #     queue_size=1,
        # )
        rospy.Subscriber(
            "/joint_states", sensor_msgs.msg.JointState, self.joints_cb, queue_size=1
        )
        self.commander = Gen3LiteCommander()
        # self.commander.move_group.set_end_effector_link(self.tool0_frame_id)
    
    def robot_state_cb(self, msg):
        # TODO error handling
        # detected_error = False
        # if np.any(msg.cartesian_collision):
        #     detected_error = True
        # for s in franka_msgs.msg.Errors.__slots__:
        #     if getattr(msg.current_errors, s):
        #         detected_error = True
        # if not self.robot_error and detected_error:
        #     self.robot_error = True
        #     rospy.logwarn("Detected robot error")
        pass

    def joints_cb(self, msg):
        self.gripper_width = msg.position[7] + msg.position[8]

    def recover_robot(self):
        # FIXME this is not defined
        self.commander.recover()
        self.robot_error = False
        rospy.loginfo("Recovered from robot error")

    def run(self):
        vis.clear()
        vis.draw_workspace(self.size)
        self.commander.home()
        self.commander.open_gripper()

        tsdf, pc = self.acquire_tsdf()
        vis.draw_tsdf(tsdf.get_grid().squeeze(), tsdf.voxel_size)
        vis.draw_points(np.asarray(pc.points))
        rospy.loginfo("Reconstructed scene")

        state = State(tsdf, pc)
        grasps, scores, planning_time = self.plan_grasps(state)
        vis.draw_grasps(grasps, scores, self.finger_depth)
        rospy.loginfo("Planned grasps")

        if len(grasps) == 0:
            rospy.loginfo("No grasps detected")
            return

        grasp, score = self.select_grasp(grasps, scores)
        vis.draw_grasp(grasp, score, self.finger_depth)
        rospy.loginfo("Selected grasp")

        self.commander.home()
        label = self.execute_grasp(grasp)
        rospy.loginfo("Grasp execution")

        if self.robot_error:
            self.recover_robot()
            return

        if label:
            self.drop()
        self.commander.home()

    def acquire_tsdf(self):
        self.commander.goto_joints(self.scan_joints[0])

        self.tsdf_server.reset()
        self.tsdf_server.integrate = True

        for joint_target in self.scan_joints[1:]:
            if rospy.is_shutdown():
                break
            self.commander.goto_joints(joint_target)

        self.tsdf_server.integrate = False
        tsdf = self.tsdf_server.low_res_tsdf
        pc = self.tsdf_server.high_res_tsdf.get_cloud()

        rospy.loginfo("Pointcloud had %d points", len(pc.points))

        return tsdf, pc

    def select_grasp(self, grasps, scores):
        # select the highest grasp
        heights = np.empty(len(grasps))
        for i, grasp in enumerate(grasps):
            heights[i] = grasp.pose.translation[2]
        idx = np.argmax(heights)
        grasp, score = grasps[idx], scores[idx]

        # make sure camera is pointing forward
        rot = grasp.pose.rotation
        axis = rot.as_matrix()[:, 0]
        if axis[0] < 0:
            grasp.pose.rotation = rot * Rotation.from_euler("z", np.pi)

        return grasp, score

    def execute_grasp(self, grasp):
        T_task_grasp = grasp.pose
        T_base_grasp = self.T_base_task * T_task_grasp


        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_base_pregrasp = T_base_grasp * T_grasp_pregrasp
        T_base_retreat = T_base_grasp * T_grasp_retreat

        self.commander.goto_pose(T_base_pregrasp* self.T_tcp_tool0)
        self.approach_grasp(T_base_grasp)

        # if self.robot_error:
        #     return False

        self.commander.close_gripper()

        # if self.robot_error:
        #     return False

        self.commander.goto_pose(T_base_retreat * self.T_tcp_tool0)

        # # lift hand
        T_retreat_lift_base = Transform(Rotation.identity(), [0.0, 0.0, 0.1])
        T_base_lift = T_retreat_lift_base * T_base_retreat
        self.commander.goto_pose(T_base_lift * self.T_tcp_tool0)

        # if self.gripper_width > 0.004:
        #     return True
        # else:
        #     return False

    def approach_grasp(self, T_base_grasp):
        self.commander.goto_pose(T_base_grasp * self.T_tcp_tool0) 

    def drop(self):
        self.commander.goto_joints(
            [0.678, 0.097, 0.237, -1.63, -0.031, 1.756, 0.931], 0.2, 0.2
        )
        self.commander.open_gripper()


class TSDFServer(object):
    def __init__(self):
        self.cam_frame_id = rospy.get_param("/vgn_grasp_gen3_lite/cam/frame_id")
        self.cam_topic_name = rospy.get_param("/vgn_grasp_gen3_lite/cam/topic_name")
        self.intrinsic = CameraIntrinsic.from_dict(rospy.get_param("/vgn_grasp_gen3_lite/cam/intrinsic"))
        self.size = 6.0 * rospy.get_param("/vgn_grasp_gen3_lite/finger_depth")

        self.cv_bridge = cv_bridge.CvBridge()
        self.tf_tree = ros_utils.TransformTree()
        self.integrate = False
        rospy.Subscriber(self.cam_topic_name, sensor_msgs.msg.Image, self.sensor_cb)

    def reset(self):
        self.low_res_tsdf = TSDFVolume(self.size, 40)
        self.high_res_tsdf = TSDFVolume(self.size, 120)

    def sensor_cb(self, msg):
        if not self.integrate:
            return
        
        rospy.loginfo("Integrating image.")

        img = self.cv_bridge.imgmsg_to_cv2(msg).astype(np.float32) * 0.001
        T_cam_task = self.tf_tree.lookup(
            self.cam_frame_id, "task", msg.header.stamp, rospy.Duration(0.1)
        )

        self.low_res_tsdf.integrate(img, self.intrinsic, T_cam_task)
        self.high_res_tsdf.integrate(img, self.intrinsic, T_cam_task)


def main():
    rospy.init_node("vgn_grasp_gen3_lite")
    controller = Gen3LiteGraspController()

    while not rospy.is_shutdown():
        controller.run()


if __name__ == "__main__":
    main()
