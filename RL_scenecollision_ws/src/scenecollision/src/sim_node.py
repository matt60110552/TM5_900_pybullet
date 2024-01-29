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
        self.simulation_server = rospy.Service("simulation_data", path_planning, self.create_scene)

    def create_scene(self, request):
        self.actor.env.reset(save=False, enforce_face_target=False, init_joints=self.actor.init_joint_pose, reset_free=True)
        self.obs_pc_world = np.array(list(point_cloud2.read_points(request.env_data.obstacle_pointcloud,
                                                                     field_names=("x", "y", "z"),
                                                                     skip_nans=True)))
        self.tar_pc_world = np.array(list(point_cloud2.read_points(request.env_data.target_pointcloud,
                                                                     field_names=("x", "y", "z"),
                                                                     skip_nans=True)))


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

        grasp_poses_camera = np.array(request.env_data.grasp_poses.data, dtype=np.float32).reshape(dims)

        # Convert the grasp poses' frame from camera to world(base)
        self.grasp_poses = []
        ef_pose = self.actor.env._get_ef_pose('mat')
        for grasp_pose_camera in grasp_poses_camera:
            self.grasp_poses.append(np.dot(ef_pose, np.dot(self.actor.env.cam_offset, grasp_pose_camera)))
        self.grasp_poses = np.array(self.grasp_poses)
        
        self.grasp_poses = self.grasp2pre_grasp(self.grasp_poses, drawback_dis=0.04) # Drawback a little



        start_time = time.time()
        path = np.array(self.actor.create_simulation_env(self.obs_pc_world,
                                                         self.tar_pc_world,
                                                         self.grasp_poses))

        print(f"\n\n\n\n\npath's consuming time: {time.time() - start_time}\n\n\n\n\n")

        path_response = path_planningResponse()
        for _ in range(2):
            path_response.joint_config.layout.dim.append(std_msgs.msg.MultiArrayDimension())
        path_response.joint_config.layout.dim[0].label = "path_num"
        path_response.joint_config.layout.dim[0].size = 30  # Size of path, default is 30
        path_response.joint_config.layout.dim[0].stride = 30*6  # Size of each waypoint
        path_response.joint_config.layout.data_offset = 0
        path_response.joint_config.layout.dim[1].label = "joint_num"
        path_response.joint_config.layout.dim[1].size = 6 
        path_response.joint_config.layout.dim[1].stride = 6
        path_response.joint_config.layout.data_offset = 0


        if path.all() == None:
            print(f"no path")
            tmp_list = (10.*np.ones((30, 6))).flatten().tolist() 
            print(f"tmp_list: {type(tmp_list[0])}")
            path_response.joint_config.data = tmp_list
        else:
            # The shape of path is (30, 6)
            path_flatten = path.flatten()
            path_response.joint_config.data = path_flatten.tolist()

        return path_response


    def grasp2pre_grasp(self, grasp_poses, drawback_dis=0.02):
        # This function will make the grasp poses retreat a little
        drawback_matrix = np.identity(4)
        drawback_matrix[2, 3] = -drawback_dis

        rotation_matrix_180_deg = np.array([[-1, 0, 0],
                                            [0, -1, 0],
                                            [0, 0, 1]])
        result_poses = []
        for i in range(len(grasp_poses)):
            grasp_candidate = np.dot(grasp_poses[i], drawback_matrix)
            rotate_grasp_candidate = copy.deepcopy(grasp_candidate)
            rotate_grasp_candidate[:3, :3] = np.dot(rotation_matrix_180_deg, rotate_grasp_candidate[:3, :3])
            result_poses.append(grasp_candidate)
            result_poses.append(rotate_grasp_candidate)
        return result_poses

if __name__ == "__main__":
    rospy.init_node("sim")
    real_actor_node = ros_node(renders=1)
    rospy.spin()