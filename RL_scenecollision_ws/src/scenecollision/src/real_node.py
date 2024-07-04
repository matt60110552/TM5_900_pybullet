#!/usr/bin/env python3
import numpy as np
import os
import sys
import time
import open3d as o3d
import copy
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
        self.simple_sub = rospy.Subscriber("simple_cmd", Int32, self.simple_callback)

        self.simulation_client = rospy.ServiceProxy("simulation_data", path_planning)
        self.object_num = 1
        self.redundent = False
    def get_env_callback(self, msg):
        for _ in range(msg.data):
            # self.actor.env.reset(save=False, enforce_face_target=False,
            #                      init_joints=self.actor.init_joint_pose,
            #                      reset_free=True)
            
            self.actor.env._panda.reset(self.actor.init_joint_pose)
            self.actor.env.place_back_objects()
            # self.actor.env._randomly_place_objects_pack(self.actor.env._get_random_object(2), scale=1, if_stack=True)
            self.actor.env._shelf_place_objects(self.actor.env._get_random_object(self.object_num),
                                                scale=1,
                                                if_stack=True)


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
            

            # Get the pointcloud of target and obstacle in world frame
            noise_stddev = 0.015/np.sqrt(3)
            (obstacle_points_world,
             target_points_world,
             scene_points_world) = self.actor.get_pc_state(frame="world", vis=False, concat=False)
            if target_points_world is None:
                self.actor.freeze_release(option=False) # To make the target object be released
                print(f"no target pointcloud")
                continue
            obstacle_points_world = obstacle_points_world[:, :3]
            target_points_world = target_points_world[:, :3]

            obstacle_points_world = obstacle_points_world + np.random.normal(scale=noise_stddev,
                                                                             size=obstacle_points_world.shape)
            target_points_world = target_points_world + np.random.normal(scale=noise_stddev,
                                                                         size=target_points_world.shape)



            # self.actor.env.move([0.02, 0.03, 0.02, 0.1, -0.1, 0])
            # self.actor.env.move([0.02, 0.03, 0.02, 0.1, -0.1, 0])

            # for _ in range(6):
            #     self.actor.env.move([-0.03, 0.03, 0.0, 0., 0.05, 0])
            
            grasp_list = []
            score_list = []
            for i in range(4):
                intrinsic_matrix = self.actor.env.intrinsic_matrix
                

                obstacle_points, target_points, scene_points = self.actor.get_pc_state(frame="camera", vis=False)
                obstacle_points = obstacle_points[:, :3]
                target_points = target_points[:, :3]


                # noise_stddev = 0.015/np.sqrt(3)
                # obstacle_points = obstacle_points + np.random.normal(scale=noise_stddev, size=obstacle_points.shape)
                # target_points = target_points + np.random.normal(scale=noise_stddev, size=target_points.shape)
                vis_scene_points = np.concatenate((target_points, obstacle_points), axis=1)

                grasp_poses_camera = self.setting_contact_req(color_image, depth_image, mask_image,
                                                              intrinsic_matrix, obstacle_points, target_points)
                
                print(f"length of grasp_poses_camera: {len(grasp_poses_camera)}")
                ef_pose = self.actor.env._get_ef_pose('mat')
                cam_offset_inv = np.linalg.inv(self.actor.env.cam_offset)
                for grasp_pose_cam in grasp_poses_camera:
                    grasp_camera = np.array(grasp_pose_cam.pred_grasps_cam)
                    grasp_world = np.dot(ef_pose, np.dot(cam_offset_inv, grasp_camera.reshape(4,4)))
                    if grasp_world[0, 2] > -0.6:
                        grasp_list.append(grasp_world.flatten())
                        score_list.append(grasp_pose_cam.score)

            grasp_num = len(grasp_list)
            print(f"grasp_num: {grasp_num}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            if grasp_num != 0:
                # self.actor.visualize_points_grasppose(scene_points=scene_points_world, grasp_list=grasp_list)
                # self.actor.visualize_points_grasppose(scene_points=obstacle_points_world, grasp_list=grasp_list)
                
                combined_data = list(zip(grasp_list, score_list))
                # Sort based on the scores from score_list
                sorted_combined_data = sorted(combined_data, key=lambda x: x[1], reverse=True)
                # sorted_combined_data = sorted(combined_data, key=lambda x: x[1])

                # Unpack the sorted data
                sorted_grasp_list, sorted_score_list = zip(*sorted_combined_data)
                grasp_poses_data = np.array(sorted_grasp_list)
                grasp_score_data = np.array(sorted_score_list)


                start_joint = self.actor.get_joint_degree()
                # print(f"start_joint: {start_joint}")
                custom_req = self.setup_path_planning_req(grasp_poses_data,
                                                          grasp_score_data,
                                                          start_joint,
                                                          target_points_world,
                                                          obstacle_points_world)
                respond = self.simulation_client(custom_req)
                path_num = respond.joint_config.layout.dim[0].size
                grasp_num = respond.grasp_poses.layout.dim[0].size

                joint_path_list = np.reshape(np.array(respond.joint_config.data), (path_num, 40,  6))
                grasp_pose_list = np.reshape(np.array(respond.grasp_poses.data), (grasp_num, 4, 4))

                # self.actor.visualize_points_grasppose(scene_points=scene_points_world,
                #                                       grasp_list=grasp_pose_list)


                if joint_path_list.shape[0] == 0 or joint_path_list[0][0][0] == 10.0:
                    print(f"no valid path!")
                    self.actor.freeze_release(option=False) # To make the target object be released
                    self.actor.clear_constraints()
                    self.actor.env.place_back_objects()
                else:
                    print(f"start moving!")
                    # Bitstar path
                    for idx, joint_path in enumerate(joint_path_list):
                        # if idx != 0 and idx != len(joint_path_list) - 1:
                        #     continue
                        joint_path = joint_path

                        # if not self.redundent:
                        #     joint_path = self.remove_redundent(joint_path)
                        retreat_joint_path = copy.deepcopy(joint_path)
                        retreat_joint_path = np.flip(retreat_joint_path, axis=0)
                        # Forward part
                        for idx, waypoint in enumerate(joint_path):
                            waypoint = np.append(waypoint, [0, 0, 0])
                            self.actor.env.move(waypoint, obs=False, config=True, repeat=120)
                        
                        # Pregrasp to grasp
                        for _ in range(5):
                            self.actor.env.move([0, 0, 0.01, 0, 0, 0])
                        
                        # Start to grasp
                        self.actor.freeze_release(option=False)
                        self.actor.env.grasp()
                        
                        # Backward part
                        for waypoint in retreat_joint_path:
                            waypoint = np.append(waypoint, [0, 0, 0.8])
                            self.actor.env.move(waypoint, obs=False, config=True, repeat=120)

                        self.actor.replace_target_object()
                        self.actor.env._panda.reset(self.actor.init_joint_pose)
                        self.actor.freeze_release(option=True)
                        # break # grasp one time
                    
                    
                    # print(f"move by predefined path!!!")
                    # # Pre-defined path
                    # for joint_path, valid_path in zip(pre_joint_path_list, pre_valid_path_list):
                    #     for waypoint, valid in zip(joint_path, valid_path):
                    #         if valid:
                    #             self.actor.move_directly(waypoint)
                    #             time.sleep(0.5)

                    self.actor.freeze_release(option=False)
                    self.actor.clear_constraints()
                    self.actor.env.place_back_objects()
            else:
                self.actor.freeze_release(option=False) # To make the target object be released
                self.actor.clear_constraints()
                print(f"no grasp pose")


    def setup_path_planning_req(self, car_poses, poses_score=None, start_joint=None,
                                target_points=None, obstacle_points=None):
        grasp_mat_num = len(car_poses)
        poses_req = path_planningRequest()
        # Fill in the header
        poses_req.env_data.header.stamp = rospy.Time.now()
        poses_req.env_data.header.frame_id = 'base_link'  # Replace with your desired frame ID
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'base_link'  # Replace with your desired frame ID
        # Fill in the obstacle point cloud
        # Assuming 'obstacle_pointcloud_data' is a NumPy array representing your obstacle point cloud
        if target_points is not None:
            poses_req.env_data.target_pointcloud = create_cloud_xyz32(header, target_points)
        if obstacle_points is not None:
            poses_req.env_data.obstacle_pointcloud = create_cloud_xyz32(header, obstacle_points)


        car_poses_flat = car_poses.flatten()
        for _ in range(3):
            poses_req.env_data.grasp_poses.layout.dim.append(std_msgs.msg.MultiArrayDimension())
        poses_req.env_data.grasp_poses.layout.dim[0].label = "grasp_mat_num"
        poses_req.env_data.grasp_poses.layout.dim[0].size = grasp_mat_num  # Size of each pose (assuming 4x4 matrix)
        poses_req.env_data.grasp_poses.layout.dim[0].stride = grasp_mat_num*16  # Total size of the array (4x4 matrix)
        poses_req.env_data.grasp_poses.layout.data_offset = 0
        poses_req.env_data.grasp_poses.layout.dim[1].label = "grasp_matrix_height"
        poses_req.env_data.grasp_poses.layout.dim[1].size = 4  # Size of each pose (assuming 4x4 matrix)
        poses_req.env_data.grasp_poses.layout.dim[1].stride = 16  # Total size of the array (4x4 matrix)
        poses_req.env_data.grasp_poses.layout.data_offset = 0
        poses_req.env_data.grasp_poses.layout.dim[2].label = "grasp_matrix_weight"
        poses_req.env_data.grasp_poses.layout.dim[2].size = 4  # Size of each pose (assuming 4x4 matrix)
        poses_req.env_data.grasp_poses.layout.dim[2].stride = 4  # Total size of the array (4x4 matrix)
        poses_req.env_data.grasp_poses.layout.data_offset = 0
        poses_req.env_data.grasp_poses.data = car_poses_flat.tolist()
        if poses_score is not None:
            poses_req.env_data.scores = poses_score
        if start_joint is not None:
            poses_req.env_data.start_joint = start_joint
        return poses_req


        
    def setting_contact_req(self, color_image, depth_image,
                            mask_image, intrinsic_matrix,
                            obstacle_points, target_points):
        bridge = CvBridge()
        color_msg = bridge.cv2_to_imgmsg(np.uint8(color_image), encoding='bgr8')
        depth_msg = bridge.cv2_to_imgmsg(depth_image)
        mask_msg = bridge.cv2_to_imgmsg(np.uint8(mask_image), encoding='mono8')    # Assuming mask is grayscale
        
        # Create a service request(depth image part)
        contact_request = GraspGroupRequest()
        contact_request.rgb = color_msg
        contact_request.depth = depth_msg
        contact_request.seg = mask_msg
        contact_request.K = intrinsic_matrix.flatten()  # Flatten the 3x3 matrix into a 1D array
        contact_request.segmap_id = 1

        # Create a service request(pointcloud part)
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'base_link'  # Replace with your desired frame ID
        full_pc = np.concatenate((obstacle_points[:, :3], target_points[:, :3]), axis=0)
        contact_request.pc_full = create_cloud_xyz32(header, full_pc)
        contact_request.pc_target = create_cloud_xyz32(header, target_points[:, :3])
        contact_request.mode = 1
        grasp_poses = self.contact_client(contact_request).grasp_poses

        return grasp_poses
    
    def simple_callback(self, msg):
        self.actor.env.reset(save=False, enforce_face_target=False,
                             init_joints=self.actor.init_joint_pose,
                             num_object = 2,
                             reset_free=True)
        # self.actor.env._panda.reset(self.actor.init_joint_pose)
        # self.actor.env.place_back_objects()
        # self.actor.env._randomly_place_objects_pack(self.actor.env._get_random_object(2), scale=1, if_stack=True)

        point_cloud = self.actor.get_world_pointcloud(raw_data=True)
        print(f"point_cloud: {point_cloud}")

    def remove_redundent(self, joint_path, tolerance=3e-2):
        """
        Remove redundant waypoints that lie approximately in the middle of the previous and next waypoints.

        Parameters:
        - waypoints: An Nx6 numpy array of waypoints.
        - tolerance: The tolerance threshold for determining redundancy.

        Returns:
        - A numpy array of waypoints with redundancies removed.
        """
        def is_redundant(prev, curr, next_):
            # Calculate vectors
            vec_prev_curr = curr - prev
            vec_prev_next = next_ - prev

            # Check if curr lies on the line segment between prev and next_
            # if np.linalg.norm(np.cross(vec_prev_next, vec_prev_curr)) / np.linalg.norm(vec_prev_next) < tolerance:
            #     return True
            # return False
            proj_vec = vec_prev_next * np.dot(vec_prev_curr, vec_prev_next)/(np.linalg.norm(vec_prev_curr)*
                                                                             np.linalg.norm(vec_prev_next))
            offset_vec = vec_prev_curr - proj_vec
            for i in offset_vec:
                if np.abs(i) > tolerance:
                    print(f"offset_vec1: {offset_vec}")
                    return False
            print(f"offset_vec2: {offset_vec}")
            return True
        joint_path = np.array(joint_path)
        if len(joint_path) < 3:
            return joint_path

        cleaned_waypoints = [joint_path[0]]
        for i in range(1, len(joint_path) - 1):
            if not is_redundant(joint_path[i-1], joint_path[i], joint_path[i+1]):
                cleaned_waypoints.append(joint_path[i])
        cleaned_waypoints.append(joint_path[-1])
        
        return np.array(cleaned_waypoints)


if __name__ == "__main__":
    rospy.init_node("real")
    real_actor_node = ros_node(renders=True)
    rospy.spin()