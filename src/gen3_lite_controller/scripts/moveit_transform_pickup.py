#!/usr/bin/env python3

# Software License Agreement (BSD License)
#
# Copyright (c) 2013, SRI International
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of SRI International nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Acorn Pooley, Mike Lautman

# Inspired from http://docs.ros.org/kinetic/api/moveit_tutorials/html/doc/move_group_python_interface/move_group_python_interface_tutorial.html
# Modified by Alexandre Vannobel to test the FollowJointTrajectory Action Server for the Kinova Gen3 robot

# To run this node in a given namespace with rosrun (for example 'my_gen3'), start a Kortex driver and then run : 
# rosrun kortex_examples example_move_it_trajectories.py __ns:=my_gen3

import random
import sys
import rospy
import moveit_commander
import moveit_msgs.msg
from math import pi
import tf2_ros
import tf2_geometry_msgs
import geometry_msgs.msg

class ExampleMoveItTrajectories(object):
  """ExampleMoveItTrajectories"""
  def __init__(self):

    # Initialize the node
    super(ExampleMoveItTrajectories, self).__init__()
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('example_move_it_trajectories')

    try:
      self.is_gripper_present = rospy.get_param(rospy.get_namespace() + "is_gripper_present", False)
      if self.is_gripper_present:
        gripper_joint_names = rospy.get_param(rospy.get_namespace() + "gripper_joint_names", [])
        self.gripper_joint_name = gripper_joint_names[0]
      else:
        self.gripper_joint_name = ""
      self.degrees_of_freedom = rospy.get_param(rospy.get_namespace() + "degrees_of_freedom", 7)

      # Create the MoveItInterface necessary objects
      arm_group_name = "arm"
      self.robot = moveit_commander.RobotCommander("robot_description")
      self.scene = moveit_commander.PlanningSceneInterface(ns=rospy.get_namespace())
      pose = geometry_msgs.msg.PoseStamped()
      pose.header.frame_id = "base_link"
      pose.pose.position.z = -0.03
      self.scene.add_plane("table", pose, normal=(0, 0, 1))
      pose.pose.position.x = -0.4
      self.scene.add_box("shelf", pose, size=(0.1, 2, 2))
      pose.pose.position.x = 0
      pose.pose.position.y = 0.5
      self.scene.add_box("table_lip", pose, size=(1, 0.05, 0.2))
      self.arm_group = moveit_commander.MoveGroupCommander(arm_group_name, ns=rospy.get_namespace())
      self.display_trajectory_publisher = rospy.Publisher(rospy.get_namespace() + 'move_group/display_planned_path',
                                                    moveit_msgs.msg.DisplayTrajectory,
                                                    queue_size=20)

      if self.is_gripper_present:
        gripper_group_name = "gripper"
        self.gripper_group = moveit_commander.MoveGroupCommander(gripper_group_name, ns=rospy.get_namespace())

      rospy.loginfo("Initializing node in namespace " + rospy.get_namespace())
    except Exception as e:
      print (e)
      self.is_init_success = False
    else:
      self.is_init_success = True


  def reach_named_position(self, target):
    arm_group = self.arm_group
    
    # Going to one of those targets
    rospy.loginfo("Going to named target " + target)
    # Set the target
    arm_group.set_named_target(target)
    # Plan the trajectory
    (success_flag, trajectory_message, planning_time, error_code) = arm_group.plan()
    # Execute the trajectory and block while it's not finished
    return arm_group.execute(trajectory_message, wait=True)

  def reach_joint_angles(self, tolerance):
    arm_group = self.arm_group
    success = True

    # Get the current joint positions
    joint_positions = arm_group.get_current_joint_values()
    rospy.loginfo("Printing current joint positions before movement :")
    for p in joint_positions: rospy.loginfo(p)

    # Set the goal joint tolerance
    self.arm_group.set_goal_joint_tolerance(tolerance)

    # Set the joint target configuration
    if self.degrees_of_freedom == 7:
      joint_positions[0] = pi/2
      joint_positions[1] = 0
      joint_positions[2] = pi/4
      joint_positions[3] = -pi/4
      joint_positions[4] = 0
      joint_positions[5] = pi/2
      joint_positions[6] = 0.2
    elif self.degrees_of_freedom == 6:
      joint_positions[0] = 0
      joint_positions[1] = 0
      joint_positions[2] = pi/2
      joint_positions[3] = pi/4
      joint_positions[4] = 0
      joint_positions[5] = pi/2
    arm_group.set_joint_value_target(joint_positions)
    
    # Plan and execute in one command
    success &= arm_group.go(wait=True)

    # Show joint positions after movement
    new_joint_positions = arm_group.get_current_joint_values()
    rospy.loginfo("Printing current joint positions after movement :")
    for p in new_joint_positions: rospy.loginfo(p)
    return success

  def get_cartesian_pose(self):
    arm_group = self.arm_group

    # Get the current pose and display it
    pose = arm_group.get_current_pose()
    rospy.loginfo("Actual cartesian pose is : ")
    rospy.loginfo(pose.pose)

    return pose.pose

  def reach_cartesian_pose(self, pose, tolerance, constraints):
    arm_group = self.arm_group
    
    # Set the tolerance
    arm_group.set_goal_position_tolerance(tolerance)

    # Set the trajectory constraint if one is specified
    if constraints is not None:
      arm_group.set_path_constraints(constraints)

    # Get the current Cartesian Position
    arm_group.set_pose_target(pose)

    # Plan and execute
    rospy.loginfo("Planning and going to the Cartesian Pose")
    return arm_group.go(wait=True)

  def reach_gripper_position(self, relative_position):
    gripper_group = self.gripper_group
    
    # We only have to move this joint because all others are mimic!
    gripper_joint = self.robot.get_joint(self.gripper_joint_name)
    gripper_max_absolute_pos = gripper_joint.max_bound()
    gripper_min_absolute_pos = gripper_joint.min_bound()
    rospy.loginfo("Gripper joint name: " + self.gripper_joint_name)
    rospy.loginfo("Gripper max bound: " + str(gripper_max_absolute_pos))
    rospy.loginfo("Gripper min bound: " + str(gripper_min_absolute_pos))
    try:
      val = gripper_joint.move(relative_position * (gripper_max_absolute_pos - gripper_min_absolute_pos) + gripper_min_absolute_pos, True)
      return val
    except Exception as e:
      rospy.logerr("An error occurred while moving the gripper: " + str(e))
      return False 

def main():
    example = ExampleMoveItTrajectories()
    print("Successfully primed gripper." if example.reach_gripper_position(1.0) else "Failure")
    target_pose = example.get_cartesian_pose() # FIXME not sure what this type is so just taking it from this method :((
    target_pose.position.x = 0.0
    target_pose.position.y = 0.35
    target_pose.position.z = 0.3
    target_pose.orientation.x = 0.0
    target_pose.orientation.y = 1.0
    target_pose.orientation.z = 0.0
    target_pose.orientation.w = 0.0
    deposit_pose = geometry_msgs.msg.Pose()
    deposit_pose.position.x = 0.2
    deposit_pose.position.y = 0.3
    deposit_pose.position.z = 0.3
    deposit_pose.orientation.x = 1.0
    deposit_pose.orientation.y = 0.0
    deposit_pose.orientation.z = 0.0
    deposit_pose.orientation.w = 0.0
    while not rospy.is_shutdown():
      print("Successfully reached observation point." if example.reach_cartesian_pose(pose=target_pose, tolerance=0.01, constraints=None) else "Failure")

      rospy.sleep(2)
      
      grasp_poses: geometry_msgs.msg.PoseArray = rospy.wait_for_message('/grasps/pose', geometry_msgs.msg.PoseArray)

      print("Received poses: " + str(grasp_poses.poses))
      if len(grasp_poses.poses) == 0:
        print("No poses received, shutting down...")
        break

      # find what this pose is in relation to the base_link frame
      tf_buffer = tf2_ros.Buffer()
      listener = tf2_ros.TransformListener(tf_buffer)
      transformed_poses = []
      for pose in grasp_poses.poses:
        # FIXME this is dumb but I don't know how to do this properly
        pose = geometry_msgs.msg.PoseStamped(pose=pose)
        pose.header.frame_id = grasp_poses.header.frame_id
        transformed_grasp_pose = None
        while transformed_grasp_pose is None:
          try:
            transform = tf_buffer.lookup_transform('base_link', pose.header.frame_id, rospy.Time(0))
            transformed_grasp_pose = tf2_geometry_msgs.do_transform_pose(pose, transform)
          except Exception as e:
            if not rospy.is_shutdown():
              rospy.logwarn("Failed to transform pose: "+str(e)+", retrying...")
            else:
              return
        rospy.loginfo("Transformed pose: " + str(transformed_grasp_pose))
        transformed_poses.append(transformed_grasp_pose.pose)
      # choose the pose with the highest z
      transformed_grasp_pose = max(transformed_poses, key=lambda pose: pose.position.z)
      # transformed_grasp_pose.pose.position.z = 0.05
      transformed_grasp_pose.orientation = target_pose.orientation
      print("Successfully reached grasp point." if example.reach_cartesian_pose(pose=transformed_grasp_pose, tolerance=0.01, constraints=None) else "Failure")
      print("Successfully actuated gripper." if example.reach_gripper_position(0.15) else "Failure")
      print("Successfully reached deposit point." if example.reach_cartesian_pose(pose=deposit_pose, tolerance=0.01, constraints=None) else "Failure")
      print("Successfully actuated gripper." if example.reach_gripper_position(0.7) else "Failure")




if __name__ == '__main__':
  main()