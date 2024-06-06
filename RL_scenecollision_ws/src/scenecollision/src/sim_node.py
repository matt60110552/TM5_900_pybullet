#!/usr/bin/env python3
import numpy as np
import os
import time
import open3d as o3d
# from replay_buffer import ReplayMemoryWrapper
from actor_scenecollision import ActorWrapper
import argparse
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from std_msgs.msg import Int32
import std_msgs
from scenecollision.msg import GraspPose, motion_planning
from scenecollision.srv import path_planning, path_planningResponse
from sensor_msgs.point_cloud2 import create_cloud_xyz32
import copy


class ros_node(object):
    def __init__(self, renders):
        self.actor = ActorWrapper(renders=renders)
        self.simulation_server = rospy.Service("simulation_data", path_planning, self.create_path)
        self.cart_path = True

    def create_path(self, request):
        self.obs_pc_world = np.array(list(point_cloud2.read_points(request.env_data.obstacle_pointcloud,
                                                                     field_names=("x", "y", "z"),
                                                                     skip_nans=True)))
        self.tar_pc_world = np.array(list(point_cloud2.read_points(request.env_data.target_pointcloud,
                                                                     field_names=("x", "y", "z"),
                                                                     skip_nans=True)))
        self.actor.env.reset(save=False, enforce_face_target=False, init_joints=self.actor.init_joint_pose, reset_free=True)
        
        self.scene_pc_world = np.concatenate([self.obs_pc_world[:, :3], self.tar_pc_world[:, :3]], axis=0)
        
        
        dims = []
        for i in request.env_data.grasp_poses.layout.dim:
            dims.append(i.size)

        self.grasp_poses = np.array(request.env_data.grasp_poses.data, dtype=np.float32).reshape(dims)
        self.grasp_poses = self.grasp2pre_grasp(self.grasp_poses, drawback_dis=0.07) # Drawback a little
        self.grasp_scores = np.array(request.env_data.scores).reshape(len(self.grasp_poses))
        self.start_joint = np.array(request.env_data.start_joint)
        
        print(f"sim start joint: {self.start_joint}")


        start_time = time.time()
        if self.actor.sim_furniture_id is not None:
            self.actor.remove_sim_fureniture()
            self.actor.sim_furniture_id = None
        # This function is only for sim_actor_id, the real_actor_id won't enter here
        self.actor.env.place_back_objects()
        self.actor.replace_real_furniture()


        
        if self.actor.sim_furniture_id is None:
            starttime = time.time()
            # self.sim_furniture_id, self.sim_target_id = self.create_obstacle_from_pc(obs_pc, target_pc)
            self.actor.sim_furniture_id = self.actor.create_obstacle_from_pc(self.obs_pc_world,
                                                                             self.tar_pc_world)
            print(f"simulate duration: {time.time()-starttime}!!!!!!!!!!!!!!!!!")
        # Be careful, the list of score will increase due to the rotate of the 6th joint
        (grasp_joint_list, grasp_poses_list,
         elbow_pos_list, first3_list,
         grasp_score_list) = self.actor.grasp_pose2grasp_joint(grasp_poses=self.grasp_poses,
                                                               grasp_scores=self.grasp_scores)
        grasp_joint_list = np.array(grasp_joint_list)
        elbow_pos_list = np.array(elbow_pos_list)
        first3_list = np.array(first3_list)
        grasp_poses_list = np.array(grasp_poses_list)

        motion_planning_start_time = time.time()
        if len(elbow_pos_list) == 0:
            None_path_list = [[[None] * 6 for _ in range(40)]]
            print(f"None_path_list: {np.array(None_path_list).shape}")
            path_list = grasp_poses_list = elbow_path_list = gripper_pos_list = gripper_orn_list = None_path_list
        elif len(elbow_pos_list) == 1:
            (path_list,
             elbow_path_list,
             gripper_pos_list,
             gripper_orn_list) = self.actor.motion_planning(grasp_joint_cfg=grasp_joint_list,
                                                            start_joint=self.start_joint,
                                                            elbow_pos_list=elbow_pos_list,
                                                            grasp_poses_list=grasp_poses_list,
                                                            cart=self.cart_path)
            print(f"motion_planning_time:{time.time() - motion_planning_start_time}")
        else:
            (highest_joint_cfg_list,
             highest_elbow_pos_list,
             highest_grasp_poses_list) = self.actor.dbscan_grouping(elbow_pos_list,
                                                                    first3_list,
                                                                    grasp_joint_list,
                                                                    grasp_score_list,
                                                                    grasp_poses_list,
                                                                    self.scene_pc_world)
            grasp_poses_list = highest_grasp_poses_list
            (path_list, 
             elbow_path_list, 
             gripper_pos_list, 
             gripper_orn_list) = self.actor.motion_planning(grasp_joint_cfg=highest_joint_cfg_list,
                                                            start_joint=self.start_joint,
                                                            elbow_pos_list=highest_elbow_pos_list,
                                                            grasp_poses_list=grasp_poses_list,
                                                            cart=self.cart_path)
            print(f"motion_planning_time:{time.time() - motion_planning_start_time}")


        path_list = np.array(path_list)
        grasp_pose_list = np.array(grasp_poses_list)
        elbow_list = np.array(elbow_path_list)
        gripper_pos_list = np.array(gripper_pos_list)
        gripper_orn_list = np.array(gripper_orn_list)


        # elbow's linear middle way points planning



        path_num = path_list.shape[0]
        path_response = path_planningResponse()
        for _ in range(3):
            path_response.joint_config.layout.dim.append(std_msgs.msg.MultiArrayDimension())
        path_response.joint_config.layout.dim[0].label = "path_num"
        path_response.joint_config.layout.dim[0].size = path_num  # Number of path, default is 30
        path_response.joint_config.layout.dim[0].stride = path_num*40*6  # Size of each path
        path_response.joint_config.layout.data_offset = 0    
        path_response.joint_config.layout.dim[1].label = "path_len"
        path_response.joint_config.layout.dim[1].size = 40  # Length of path, default is 30
        path_response.joint_config.layout.dim[1].stride = 40*6  # Total size of waypoints
        path_response.joint_config.layout.data_offset = 0
        path_response.joint_config.layout.dim[2].label = "joint_num"
        path_response.joint_config.layout.dim[2].size = 6 
        path_response.joint_config.layout.dim[2].stride = 6
        path_response.joint_config.layout.data_offset = 0


        if path_list.all() == None:
            print(f"no path")
            tmp_list = (10.*np.ones((1, 40, 6))).flatten().tolist() 
            path_response.joint_config.data = tmp_list
            grasp_pose_list = (10.*np.ones((1, 4, 4)))
        else:
            # The shape of path is (30, 6)
            
            # Get the smoothness of the path
            gripper_mat_list = np.array(self.actor.pos_orn2matrix(gripper_pos_list, gripper_orn_list))
            score_list = []

            for gripper_mat_path in  gripper_mat_list:
                score_list.append(self.path_quality_decision(gripper_mat_path))
            sorted_indices = np.argsort(score_list)
            score_list.sort()
            score_list.sort(reverse=True)
            print(f"score_list: {score_list}")

            path_list = np.array(path_list)[sorted_indices]


            path_flatten = path_list.flatten()
            path_response.joint_config.data = path_flatten.tolist()

        grasp_num = grasp_pose_list.shape[0]
        for _ in range(3):
            path_response.grasp_poses.layout.dim.append(std_msgs.msg.MultiArrayDimension())
        path_response.grasp_poses.layout.dim[0].label = "grasp_num"
        path_response.grasp_poses.layout.dim[0].size = grasp_num  # Number of path, default is 30
        path_response.grasp_poses.layout.dim[0].stride = grasp_num*16  # Size of each path
        path_response.grasp_poses.layout.data_offset = 0    
        path_response.grasp_poses.layout.dim[1].label = "grasp_matrix_height"
        path_response.grasp_poses.layout.dim[1].size = 4  # Length of path, default is 30
        path_response.grasp_poses.layout.dim[1].stride = 4*4  # Total size of waypoints
        path_response.grasp_poses.layout.data_offset = 0
        path_response.grasp_poses.layout.dim[2].label = "grasp_matrix_weight"
        path_response.grasp_poses.layout.dim[2].size = 4
        path_response.grasp_poses.layout.dim[2].stride = 4
        path_response.grasp_poses.layout.data_offset = 0


        grasp_pose_flatten = grasp_pose_list.flatten()
        path_response.grasp_poses.data = grasp_pose_flatten.tolist()



        self.path_list = path_list
        self.grasp_pose_list = grasp_pose_list
        self.elbow_list = elbow_list
        self.gripper_pos_list = gripper_pos_list
        self.gripper_orn_list = gripper_orn_list

        return path_response
        


    def grasp2pre_grasp(self, grasp_poses, drawback_dis=0.02):
        # This function will make the grasp poses retreat a little
        drawback_matrix = np.identity(4)
        drawback_matrix[2, 3] = -drawback_dis

        result_poses = []
        for i in range(len(grasp_poses)):
            grasp_candidate = np.dot(grasp_poses[i], drawback_matrix)
            result_poses.append(grasp_candidate)
        return np.array(result_poses)
    
    def gripper_change(self, positions_list, orientations_list):
        gripper_change_list = []
        for positions, orientations in zip(positions_list, orientations_list):
            tmp_sum = 0
            for i in range(1, len(positions)):
                # Calculate displacement vector between consecutive positions
                displacement = positions[i] - positions[i-1]
                
                # Check if the gripper is moving forward
                is_moving_forward = np.dot(displacement, orientations[i-1])
                tmp_sum += is_moving_forward
            gripper_change_list.append(tmp_sum/len(positions_list))
        return gripper_change_list

    def curvature_decision(self, waypoint_mat):
        # Calculate the curvature at each point along the path
        num_points = len(waypoint_mat)
        curvatures = np.zeros(num_points)

        for i in range(num_points - 1):
            # Calculate vectors between neighboring points
            v1 = waypoint_mat[i][:3, 3] - waypoint_mat[i-1][:3, 3]
            v2 = waypoint_mat[i+1][:3, 3] - waypoint_mat[i][:3, 3]

            # Calculate cross product to find the perpendicular vector
            cross_product = np.cross(v1, v2)

            # Calculate the length of vectors
            length_v1 = np.linalg.norm(v1)
            length_v2 = np.linalg.norm(v2)

            # Calculate the curvature at the point
            if length_v1 != 0 and length_v2 != 0:
                # curvature = 2 * np.linalg.norm(cross_product) / (length_v1 * length_v2 * (length_v1 + length_v2))
                curvature = 2 * np.linalg.norm(cross_product)
                if i > 20:
                    curvatures[i] = curvature
                else:
                    curvatures[i] = curvature * 0.5

        return curvatures

    def path_quality_decision(self, waypoint_mat):
        curvatures = self.curvature_decision(waypoint_mat)

        # Wheather gripper is approaching along the grasp direction
        goal_mat = waypoint_mat[-1]
        approach_list = []
        moving_list = []
        for idx in range(len(waypoint_mat[:-1])):
            moving_vec = waypoint_mat[idx+1][:3, 3] - waypoint_mat[idx][:3, 3]
            moving_vec = moving_vec / np.linalg.norm(moving_vec)
            # moving_list.append(np.dot(moving_vec, goal_mat[:3, 2]))
            # approach_list.append(np.dot(moving_vec, goal_mat[:3, 2]))

            # matrix_diff = waypoint_mat[idx][:3, :3] - goal_mat[:3, :3]
            matrix_diff = waypoint_mat[idx][:3, :2] - goal_mat[:3, :2]
            approach_list.append(np.linalg.norm(matrix_diff, ord='fro') + np.dot(moving_vec, goal_mat[:3, 2]))
        # Make a decision based on the maximum curvature

        smoothness_weight = lambda idx: 1 / (curvatures[idx] + 1)  # Smaller smoothness scores get higher weight
        direction_weight = lambda idx: approach_list[idx]  # Larger direction scores get higher weight
        # Calculate weighted scores for each list
        weighted_smoothness = [score * smoothness_weight(idx) for idx, score in enumerate(curvatures)]
        weighted_direction = [score * direction_weight(idx) for idx, score in enumerate(approach_list)]
        # Combine the weighted scores using a weighted average
        total_weighted_scores = [(w_smooth + w_dir) / 2 for w_smooth, w_dir in zip(weighted_smoothness, weighted_direction)]
        final_score = sum(total_weighted_scores)

        max_curvature = np.max(curvatures)
        # print(f"max_cur: {max_curvature}\n")
        # print(f"curvatures: {curvatures}\n\napproach_list: {approach_list}\n\ntotal_weighted_scores: {total_weighted_scores}\n")
        # print(f"moving_list: {moving_list}\n")
        # print(f"final_score: {final_score}")
        # print(f"============================")
        # return max_curvature
        return final_score

    # Function to evaluate grip pose consistency
    def evaluate_grip_pose(self, grip_pose, target_pose):
        # Calculate angle between grip pose orientation and target orientation
        grip_orientation = grip_pose[:3, :3]
        target_orientation = target_pose[:3, :3]
        angle = np.arccos(np.clip(np.dot(grip_orientation.flatten(), target_orientation.flatten()), -1.0, 1.0))
        return angle  # Return angle as a measure of consistency (smaller is better)

    # Function to calculate dot product between grip position movement and target orientation Z-axis
    def calculate_dot_product(self, grip_pos, prev_grip_pos, target_orientation):
        grip_movement_vector = grip_pos - prev_grip_pos
        target_z_axis = target_orientation[:3, 2]
        dot_product = np.dot(grip_movement_vector, target_z_axis)
        return dot_product  # Return dot product as a measure of vertical movement (larger is better)



if __name__ == "__main__":
    rospy.init_node("sim")
    real_actor_node = ros_node(renders=1)
    rospy.spin()