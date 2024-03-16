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

    def create_path(self, request):
        self.obs_pc_world = np.array(list(point_cloud2.read_points(request.env_data.obstacle_pointcloud,
                                                                     field_names=("x", "y", "z"),
                                                                     skip_nans=True)))
        self.tar_pc_world = np.array(list(point_cloud2.read_points(request.env_data.target_pointcloud,
                                                                     field_names=("x", "y", "z"),
                                                                     skip_nans=True)))
        if len(self.tar_pc_world) != 0:
            """
            Normal process, create mesh and generate paths
            """
            self.actor.env.reset(save=False, enforce_face_target=False, init_joints=self.actor.init_joint_pose, reset_free=True)
            


            # tar_pcd = o3d.geometry.PointCloud()
            # tar_pcd.points = o3d.utility.Vector3dVector(np.array(self.tar_pc_world[:, :3]))

            # obs_pcd = o3d.geometry.PointCloud()
            # obs_pcd.points = o3d.utility.Vector3dVector(np.array(self.obs_pc_world[:, :3]))

            # # Create coordinate axes
            # axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

            # # Visualize the point cloud with axes
            # o3d.visualization.draw_geometries([tar_pcd, obs_pcd, axes])

            
            dims = []
            for i in request.env_data.grasp_poses.layout.dim:
                dims.append(i.size)

            self.grasp_poses = np.array(request.env_data.grasp_poses.data, dtype=np.float32).reshape(dims)
            self.grasp_poses = self.grasp2pre_grasp(self.grasp_poses, drawback_dis=0.05) # Drawback a little
            self.grasp_scores = np.array(request.env_data.scores).reshape(len(self.grasp_poses))



            start_time = time.time()
            (path_list,
            grasp_pose_list,
            elbow_list,
            gripper_pos_list,
            gripper_orn_list) = self.actor.create_simulation_env(self.obs_pc_world,
                                                                self.tar_pc_world,
                                                                self.grasp_poses,
                                                                self.grasp_scores)
            path_list = np.array(path_list)
            grasp_pose_list = np.array(grasp_pose_list)
            elbow_list = np.array(elbow_list)
            gripper_pos_list = np.array(gripper_pos_list)
            gripper_orn_list = np.array(gripper_orn_list)


            # elbow's linear middle way points planning



            path_num = path_list.shape[0]
            path_response = path_planningResponse()
            for _ in range(3):
                path_response.joint_config.layout.dim.append(std_msgs.msg.MultiArrayDimension())
            path_response.joint_config.layout.dim[0].label = "path_num"
            path_response.joint_config.layout.dim[0].size = path_num  # Number of path, default is 30
            path_response.joint_config.layout.dim[0].stride = path_num*30*6  # Size of each path
            path_response.joint_config.layout.data_offset = 0    
            path_response.joint_config.layout.dim[1].label = "path_len"
            path_response.joint_config.layout.dim[1].size = 30  # Length of path, default is 30
            path_response.joint_config.layout.dim[1].stride = 30*6  # Total size of waypoints
            path_response.joint_config.layout.data_offset = 0
            path_response.joint_config.layout.dim[2].label = "joint_num"
            path_response.joint_config.layout.dim[2].size = 6 
            path_response.joint_config.layout.dim[2].stride = 6
            path_response.joint_config.layout.data_offset = 0


            if path_list.all() == None:
                print(f"no path")
                print(f"grasp_pose_list: {grasp_pose_list}")
                tmp_list = (10.*np.ones((1, 30, 6))).flatten().tolist() 
                path_response.joint_config.data = tmp_list
                grasp_pose_list = (10.*np.ones((1, 4, 4)))
            else:
                # The shape of path is (30, 6)
                
                # Get the smoothness of the path
                max_curvation_list = []
                for gripper_pos_path in  gripper_pos_list:
                    max_curvation_list.append(self.curvature_decision(gripper_pos_path))
                
                for maxcur in max_curvation_list:
                    print(f"maxcur: {maxcur}")

                sorted_indices = np.argsort(max_curvation_list)
                path_list = np.array(path_list)[sorted_indices]

                # wheather the gripper is moving toward its facing direction
                gripper_mat_list = np.array(self.actor.pos_orn2matrix(gripper_pos_list, gripper_orn_list))
                change_sum_list = []
                for gripper_mat_path in gripper_mat_list:
                    goal_position =  gripper_mat_path[-1][:3, 3]
                    goal_vector =  -gripper_mat_path[-1][:3, 2]
                    tmp_sum = 0
                    print(f"along_z_axis: ")
                    for idx in range(-10, -2, 1):
                        along_z_axis = np.dot(goal_vector, (gripper_mat_path[idx][:3, 3] - goal_position))
                        print(f"{along_z_axis}", end=" ")
                        tmp_sum += along_z_axis
                    print(f"\ntmp_sum: {tmp_sum}\n")
                    change_sum_list.append(tmp_sum)
                print(f"change_sum_list: {change_sum_list}")

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



            return path_response
        else:
            """
            Given a predefined path in cartesian space, turn them into joint config and then test
            they are valid or not
            """
            dims = []
            for i in request.env_data.grasp_poses.layout.dim:
                dims.append(i.size)

            middle_waypoint_list = np.array(request.env_data.grasp_poses.data, dtype=np.float32).reshape(dims)
            waypoint_num = len(middle_waypoint_list)
            print(f"waypoint_num: {waypoint_num}")
            dims = []
            for i in request.env_data.grasp_poses.layout.dim:
                dims.append(i.size)
            middle_waypoint_list = np.array(request.env_data.grasp_poses.data, dtype=np.float32).reshape(dims)
            mid_waypoint_list, valid_list = self.actor.create_simulation_env(grasp_poses=middle_waypoint_list)
            mid_path_response = path_planningResponse()
            for _ in range(2):
                mid_path_response.joint_config.layout.dim.append(std_msgs.msg.MultiArrayDimension())
            mid_path_response.joint_config.layout.dim[0].label = "waypoint_num"
            mid_path_response.joint_config.layout.dim[0].size = waypoint_num  
            mid_path_response.joint_config.layout.dim[0].stride = waypoint_num*6
            mid_path_response.joint_config.layout.data_offset = 0    
            mid_path_response.joint_config.layout.dim[1].label = "num of joint"
            mid_path_response.joint_config.layout.dim[1].size = 6  # Length of path, default is 30
            mid_path_response.joint_config.layout.dim[1].stride = 6  # Total size of waypoints
            mid_path_response.joint_config.layout.data_offset = 0
            mid_waypoint_flatten = mid_waypoint_list.flatten()
            mid_path_response.joint_config.data = mid_waypoint_flatten.tolist()


            
            mid_path_response.grasp_poses.layout.dim.append(std_msgs.msg.MultiArrayDimension())
            mid_path_response.grasp_poses.layout.dim[0].label = "grasp_num"
            mid_path_response.grasp_poses.layout.dim[0].size = waypoint_num
            mid_path_response.grasp_poses.layout.dim[0].stride = waypoint_num
            mid_path_response.grasp_poses.layout.data_offset = 0    
            valid_list_flatten = valid_list.flatten()
            mid_path_response.grasp_poses.data = valid_list_flatten.tolist()

            return mid_path_response


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

    def curvature_decision(self, path_points):
        # Calculate the curvature at each point along the path
        num_points = len(path_points)
        curvatures = np.zeros(num_points)

        for i in range(1, num_points - 1):
            # Calculate vectors between neighboring points
            v1 = path_points[i] - path_points[i-1]
            v2 = path_points[i+1] - path_points[i]

            # Calculate cross product to find the perpendicular vector
            cross_product = np.cross(v1, v2)

            # Calculate the length of vectors
            length_v1 = np.linalg.norm(v1)
            length_v2 = np.linalg.norm(v2)

            # Calculate the curvature at the point
            if length_v1 != 0 and length_v2 != 0:
                curvature = 2 * np.linalg.norm(cross_product) / (length_v1 * length_v2 * (length_v1 + length_v2))
                if i > 20:
                    curvatures[i] = curvature
                else:
                    curvatures[i] = curvature * 0.5

        # Make a decision based on the maximum curvature
        max_curvature = np.max(curvatures)
        return max_curvature

if __name__ == "__main__":
    rospy.init_node("sim")
    real_actor_node = ros_node(renders=1)
    rospy.spin()