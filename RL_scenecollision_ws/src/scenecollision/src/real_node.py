#!/usr/bin/env python3
import numpy as np
import os
import sys
import time
import open3d as o3d
sys.path.append("/home/user/RL_TM5_900_pybullet")
from utils.utils import *
# from replay_buffer import ReplayMemoryWrapper
from actor_scenecollision import ActorWrapper
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from std_msgs.msg import Int32
import std_msgs
from cv_bridge import CvBridge
from scenecollision.srv import GraspGroup, GraspGroupRequest
from scenecollision.srv import path_planning, path_planningRequest
from scenecollision.msg import GraspPose, motion_planning
from sensor_msgs.point_cloud2 import create_cloud_xyz32

class ros_node(object):
    def __init__(self, renders):
        self.actor = ActorWrapper(renders=renders)
        self.start_sub = rospy.Subscriber("start_real_cmd", Int32, self.get_env_callback)
        self.contact_client = rospy.ServiceProxy('contact_graspnet/get_grasp_result', GraspGroup)

        self.simulation_client = rospy.ServiceProxy("simulation_data", path_planning)

    def get_env_callback(self, msg):
        for _ in range(msg.data):
            self.actor.env.reset(save=False, enforce_face_target=False, init_joints=self.actor.init_joint_pose, reset_free=True)
            self.actor.freeze_release(option=True) # To make the target object be fixed
            self.actor.obstacle_points = None
            self.actor.target_points = None
            obs, joint_pos, camera_info, pose_info = self.actor.env._get_observation(raw_data=False, vis=False, no_gripper=True)
            color_image = obs[1][:3].T
            depth_image = obs[1][3].T
            mask_image = obs[1][4].T # 0 for target_object, 1 for others, including floor, furniture and gripper
            
            # make the mask image inverse
            mask = np.ones_like(mask_image, dtype=int)
            mask_image = -(mask_image - mask)
            

            
            intrinsic_matrix = self.actor.env.intrinsic_matrix
            
            # obstacle_points, target_points, scene_points = self.actor.get_pc_state(frame="world", vis=True)
            obstacle_points, target_points, scene_points = self.actor.get_pc_state(frame="camera", vis=False)
            obstacle_points = obstacle_points[:, :3]
            target_points = target_points[:, :3]
            
            noise_stddev = 0.015/np.sqrt(3)
            obstacle_points = obstacle_points + np.random.normal(scale=noise_stddev, size=obstacle_points.shape)
            target_points = target_points + np.random.normal(scale=noise_stddev, size=target_points.shape)
            vis_scene_points = np.concatenate((target_points, obstacle_points), axis=1)

            bridge = CvBridge()
            color_msg = bridge.cv2_to_imgmsg(np.uint8(color_image), encoding='bgr8')
            depth_msg = bridge.cv2_to_imgmsg(depth_image)
            mask_msg = bridge.cv2_to_imgmsg(np.uint8(mask_image), encoding='mono8')    # Assuming mask is grayscale
            
            # # Create a service request(depth image part)
            contact_request = GraspGroupRequest()
            contact_request.rgb = color_msg
            contact_request.depth = depth_msg
            contact_request.seg = mask_msg
            contact_request.K = intrinsic_matrix.flatten()  # Flatten the 3x3 matrix into a 1D array
            contact_request.segmap_id = 1

            # # Create a service request(pointcloud part)
            header = rospy.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'base_link'  # Replace with your desired frame ID
            full_pc = np.concatenate((obstacle_points[:, :3], target_points[:, :3]), axis=0)
            contact_request.pc_full = create_cloud_xyz32(header, full_pc)
            contact_request.pc_target = create_cloud_xyz32(header, target_points[:, :3])
            contact_request.mode = 1
            grasp_poses = self.contact_client(contact_request).grasp_poses

            # Get the pointcloud of target and obstacle in world frame
            obstacle_points, target_points, scene_points = self.actor.get_pc_state(frame="world", vis=False)
            obstacle_points = obstacle_points[:, :3]
            target_points = target_points[:, :3]

            obstacle_points = obstacle_points + np.random.normal(scale=noise_stddev, size=obstacle_points.shape)
            target_points = target_points + np.random.normal(scale=noise_stddev, size=target_points.shape)
            grasp_num = len(grasp_poses)
            print(f"grasp_num: {grasp_num}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            if grasp_num != 0:
                grasp_list = []
                score_list = []
                along_x_list = []
                ef_pose = self.actor.env._get_ef_pose('mat')
                for grasp in grasp_poses:
                    # Convert from camera to world
                    grasp_camera = np.array(grasp.pred_grasps_cam)
                    cam_offset_inv = np.linalg.inv(self.actor.env.cam_offset)
                    grasp_world = np.dot(ef_pose, np.dot(cam_offset_inv, grasp_camera.reshape(4,4)))
                    along_x_list.append(grasp_world[0, 2])
                    if grasp_world[0, 2] > -0.6:
                        grasp_list.append(grasp_world.flatten())
                        score_list.append(grasp.score)
                grasp_num = len(grasp_list)
                print(f"after z filter grasp_num: {grasp_num}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                
                
                self.visualize_points_grasppose(scene_points=scene_points, grasp_list=grasp_list)

                # Visualize middle waypoints
                # way_points_list = self.create_middle_waypoints(grasp_list[0], ef_pose)
                # way_points_list.append(ef_pose)
                # way_points_list.insert(0, grasp_list[0])
                # way_points_geometries = [self.create_grasp_geometry(way_point) for way_point in way_points_list]
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(np.array(scene_points[:, :3]))
                # axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                # o3d.visualization.draw_geometries([pcd, axes, *way_points_geometries])
                # self.actor.freeze_release(option=False)

                
                combined_data = list(zip(grasp_list, score_list))
                # Sort based on the scores from score_list
                sorted_combined_data = sorted(combined_data, key=lambda x: x[1], reverse=True)
                # sorted_combined_data = sorted(combined_data, key=lambda x: x[1])

                # Unpack the sorted data
                sorted_grasp_list, sorted_score_list = zip(*sorted_combined_data)
                grasp_poses_data = np.array(sorted_grasp_list)
                grasp_score_data = np.array(sorted_score_list)

                custom_msg = path_planningRequest()
                # Fill in the header
                custom_msg.env_data.header.stamp = rospy.Time.now()
                custom_msg.env_data.header.frame_id = 'base_link'  # Replace with your desired frame ID
                # Fill in the obstacle point cloud
                # Assuming 'obstacle_pointcloud_data' is a NumPy array representing your obstacle point cloud
                header = rospy.Header()
                header.stamp = rospy.Time.now()
                header.frame_id = 'base_link'  # Replace with your desired frame ID
                custom_msg.env_data.obstacle_pointcloud = create_cloud_xyz32(header, obstacle_points)
                custom_msg.env_data.target_pointcloud = create_cloud_xyz32(header, target_points)
                grasp_poses_data_flat = grasp_poses_data.flatten()
                for _ in range(3):
                    custom_msg.env_data.grasp_poses.layout.dim.append(std_msgs.msg.MultiArrayDimension())
                custom_msg.env_data.grasp_poses.layout.dim[0].label = "grasp_num"
                custom_msg.env_data.grasp_poses.layout.dim[0].size = grasp_num  # Size of each pose (assuming 4x4 matrix)
                custom_msg.env_data.grasp_poses.layout.dim[0].stride = grasp_num*16  # Total size of the array (4x4 matrix)
                custom_msg.env_data.grasp_poses.layout.data_offset = 0
                custom_msg.env_data.grasp_poses.layout.dim[1].label = "grasp_matrix_height"
                custom_msg.env_data.grasp_poses.layout.dim[1].size = 4  # Size of each pose (assuming 4x4 matrix)
                custom_msg.env_data.grasp_poses.layout.dim[1].stride = 16  # Total size of the array (4x4 matrix)
                custom_msg.env_data.grasp_poses.layout.data_offset = 0
                custom_msg.env_data.grasp_poses.layout.dim[2].label = "grasp_matrix_weight"
                custom_msg.env_data.grasp_poses.layout.dim[2].size = 4  # Size of each pose (assuming 4x4 matrix)
                custom_msg.env_data.grasp_poses.layout.dim[2].stride = 4  # Total size of the array (4x4 matrix)
                custom_msg.env_data.grasp_poses.layout.data_offset = 0
                custom_msg.env_data.grasp_poses.data = grasp_poses_data_flat.tolist()
                custom_msg.env_data.scores = grasp_score_data
                respond = self.simulation_client(custom_msg)
                path_num = respond.joint_config.layout.dim[0].size
                joint_path_list = np.reshape(np.array(respond.joint_config.data), (path_num, 30,  6))

                if joint_path_list.shape[0] == 0:
                    print(f"no valid path!")
                    self.actor.freeze_release(option=False) # To make the target object be released
                else:
                    print(f"start moving!")
                    for joint_path in joint_path_list:
                        middle_pc_list, camera2base_list = self.actor.move2grasp(joint_path=joint_path) # Remember, move2grasp will release the object itself
                        self.actor.replace_target_object()
                        self.actor.env._panda.reset(self.actor.init_joint_pose)
                        self.actor.freeze_release(option=True)
                        break
                    self.actor.freeze_release(option=False)
                    self.actor.env.place_back_objects()
                # Visualize middle grasp poses
                for idx, middle_pc in enumerate(middle_pc_list):
                    # create grasp poses part
                    header = rospy.Header()
                    header.stamp = rospy.Time.now()
                    header.frame_id = 'base_link'  # Replace with your desired frame ID
                    full_pc = np.concatenate((middle_pc[0][:, :3], middle_pc[1][:, :3]), axis=0)
                    contact_request.pc_full = create_cloud_xyz32(header, full_pc)
                    contact_request.pc_target = create_cloud_xyz32(header, middle_pc[1][:, :3])
                    contact_request.mode = 1
                    grasp_poses = self.contact_client(contact_request).grasp_poses

                    mid_grasp_list = []
                    # ef_pose = self.actor.env._get_ef_pose('mat')
                    for grasp in grasp_poses:
                        # Convert from camera to world
                        grasp_camera = np.array(grasp.pred_grasps_cam)
                        # grasp_world = np.dot(camera2base_list[idx], grasp_camera.reshape(4, 4))
                        grasp_world = grasp_camera.reshape(4, 4)
                        if grasp_world[0, 2] > -0.6:
                            mid_grasp_list.append(grasp_world.flatten())


                    self.visualize_points_grasppose(middle_pc[2][:, :3], mid_grasp_list)
            else:
                self.actor.freeze_release(option=False) # To make the target object be released
                print(f"no grasp pose")

    def create_grasp_geometry(self, grasp_pose, length=0.08, width=0.08):
        """Create a geometry representing a grasp pose as a U shape."""
        # Define the grasp frame
        frame = grasp_pose.reshape(4,4)
        # Define the U shape as a line set
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector([
            [0, 0, 0],
            [width/2, 0, 0],
            [-width/2, 0, 0],
            [width/2, 0, length/2],
            [-width/2, 0, length/2],
            [0, 0, -length/2]
        ])
        line_set.lines = o3d.utility.Vector2iVector([
            [0, 1], [0, 2], [1, 3], [2, 4], [0, 5]
        ])
        line_set.transform(frame)

        return line_set

    def create_middle_waypoints(self, goal_pose, start_pose):
        if len(goal_pose) == 16:
            goal_pose = np.array(goal_pose).reshape(4, 4)
        drawback_matrix = np.identity(4)
        drawback_matrix[2, 3] = -0.02 # drawback 2cm
        pre_pose = goal_pose
        middle_waypoints = []
        for i in range(30):
            # mat = np.eye(4)
            # # propotion = 1 / (1 + np.exp(-0.5 * (i/15 - 1)))
            # propotion = max(0.7, 1 - (0.06 * i))
            # retreat_matrix = np.dot(pre_pose, drawback_matrix)
            # mat[:3, :3] = propotion * retreat_matrix[:3, :3] + (1 - propotion)* start_pose[:3, :3]
            # print(f"np.dot: {np.dot(retreat_matrix[:3, 2], start_pose[:3, 2])}")
            # if np.dot(retreat_matrix[:3, 2], start_pose[:3, 2]) > 0.9:
            #     print(f"translation!!")
            #     mat[:3, 3] = retreat_matrix[:3, 3] + 0.05 * (start_pose[:3, 3] - retreat_matrix[:3, 3])
            # else:
            #     mat[:3, 3] = retreat_matrix[:3, 3]
            # pre_pose = mat
            # middle_waypoints.append(mat)

            mat = np.eye(4)
            retreat_matrix = np.dot(pre_pose, drawback_matrix)
            propotion = max(np.dot(retreat_matrix[:3, 2], start_pose[:3, 2]), 0.75)
            mat[:3, :3] = propotion * retreat_matrix[:3, :3] + (1 - propotion)* start_pose[:3, :3]
            print(f"propotion: {propotion}")
            if propotion > 0.9:
                print(f"translation!!")
                mat[:3, 3] = propotion * retreat_matrix[:3, 3] + (1-propotion) * start_pose[:3, 3]
            else:
                mat[:3, 3] = retreat_matrix[:3, 3]
            pre_pose = mat
            middle_waypoints.append(mat)
        
        return middle_waypoints

    def visualize_points_grasppose(self, scene_points, grasp_list):
        
        # Visualize pointcloud and grasp pose part
        grasp_geometries = [self.create_grasp_geometry(grasp_pose) for grasp_pose in grasp_list]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(scene_points[:, :3]))
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([pcd, axes, *grasp_geometries])
        # End of visualization
        

if __name__ == "__main__":
    rospy.init_node("real")
    real_actor_node = ros_node(renders=True)
    rospy.spin()