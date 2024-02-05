#!/usr/bin/env python3
import numpy as np
import os
import time
import open3d as o3d
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
                for grasp in grasp_poses:
                    grasp_list.append(grasp.pred_grasps_cam)
                    score_list.append(grasp.score)


                # grasp_poses_data = np.array(grasp_list)
                    
                combined_data = list(zip(grasp_list, score_list))
                # Sort based on the scores from score_list
                sorted_combined_data = sorted(combined_data, key=lambda x: x[1], reverse=True)

                # Unpack the sorted data
                sorted_grasp_list, sorted_score_list = zip(*sorted_combined_data)
                grasp_poses_data = np.array(sorted_grasp_list)

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

                respond = self.simulation_client(custom_msg)
                path_num = respond.joint_config.layout.dim[0].size
                joint_path_list = np.reshape(np.array(respond.joint_config.data), (path_num, 30,  6))

                if joint_path_list.shape[0] == 0:
                    print(f"no valid path!")
                    self.actor.freeze_release(option=False) # To make the target object be released
                else:
                    print(f"start moving!")
                    self.actor.move2grasp(joint_path_list=joint_path_list.tolist()) # Remember, move2grasp will release the object itself

            else:
                self.actor.freeze_release(option=False) # To make the target object be released
                print(f"no grasp pose")


if __name__ == "__main__":
    rospy.init_node("real")
    real_actor_node = ros_node(renders=True)
    rospy.spin()