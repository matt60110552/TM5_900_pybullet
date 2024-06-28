#!/usr/bin/env python3
import numpy as np
import os
import sys
import time
import open3d as o3d
import copy
sys.path.append("/home/user/MATT_TM5_900_pybullet")
from utils.utils import *
# from replay_buffer import ReplayMemoryWrapper
from actor_scenecollision import ActorWrapper
import rospy
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo, JointState
from std_msgs.msg import Int32
import std_msgs
import tf
import tf2_ros
import itertools
from sklearn.cluster import DBSCAN
from tf.transformations import quaternion_matrix
from cv_bridge import CvBridge
from scenecollision.srv import GraspGroup, GraspGroupRequest
from scenecollision.srv import path_planning, path_planningRequest
from scenecollision.msg import GraspPose, motion_planning, Robotiq2FGripper_robot_output
from sensor_msgs.point_cloud2 import create_cloud_xyz32
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, TransformStamped

class ros_node(object):
    def __init__(self, renders):
        self.actor = ActorWrapper(renders=renders)
        self.start_sub = rospy.Subscriber("test_realworld_cmd", Int32, self.get_env_callback)
        self.joint_sub = rospy.Subscriber("joint_states", JointState, self.joint_callback)
        self.tm_pub = rospy.Publisher("/target_position", Pose, queue_size=1)
        self.tm_joint_pub = rospy.Publisher("/target_joint", JointState, queue_size=1)
        self.robotiq_pub = rospy.Publisher("/Robotiq2FGripperRobotOutput", Robotiq2FGripper_robot_output, queue_size=10)
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)  # Create a tf listener

        self.points_sub = rospy.Subscriber("/uoais/Pointclouds", PointCloud2, self.points_callback)
        self.obs_points_sub = rospy.Subscriber("/uoais/obs_pc", PointCloud2, self.obs_points_callback)
        self.seg_pub = rospy.Publisher("/uoais/data_init", Int32, queue_size=1)
        self.depth_topic = rospy.get_param("~depth", "/camera/aligned_depth_to_color/image_raw")

        self.contact_client = rospy.ServiceProxy('contact_graspnet/get_grasp_result', GraspGroup)
        # rospy.wait_for_service('contact_graspnet/get_grasp_result', timeout=None)
        self.target_points = None
        self.obs_points = None
        # self.home_joint_point = [-0.04229641181761173, -1.7921697281949702, 2.502655034341253, -0.5894170708848987, 1.5575706693473996, -0.0387850963381803]
        self.home_joint_point = [0.012591255025137305, -1.2207295273003245, 1.5559966079851082, 0.023707389283668924, 1.5601789693190231, -0.04487435591399763]
        self.place_joint_point = [-1.002553783421109, -0.3444243268035077, 2.1679726506737955, -0.7325243623262604, 1.320991283913246, 0.6469062632832484]
        rospy.loginfo("Init finished!!!!!!!!!!!!!!!!!!!!!!!!!!!")


    def joint_callback(self, msg):
        cur_states = np.asarray(msg.position)
        cur_states= np.concatenate((cur_states, [0, 0, 0]))
        self.joint_states = cur_states
        

    def get_env_callback(self, msg):
        if msg.data == 0:
            # Pybullet setup
            self.actor.init_joint_pose = self.joint_states
            self.actor.env._panda.reset(self.actor.init_joint_pose)
            if self.actor.sim_furniture_id is not None:
                self.actor.remove_sim_fureniture()
                self.actor.sim_furniture_id = None
            self.actor.replace_real_furniture()
            

            # Reset the gripper
            self.control_gripper("reset")
            time.sleep(1)
            self.control_gripper("set_pose", 0.)
            time.sleep(1)
            self.control_gripper("set_pose", 0.085)
            print(f"finish grasping")


            # Reset the arm's position
            self.move_along_path([self.home_joint_point])
            time.sleep(1)


            # Set init_value to None
            self.target_points = None
            self.obs_points = None
            
            

            # Segmentation part
            seg_msg = Int32()
            seg_msg.data = 2
            self.seg_pub.publish(seg_msg)    
            time.sleep(5) # Sleep to wait for the segmentation pointcloud arrive
            
            
            self.target_points = self.remove_outlier_points(self.target_points)
            self.visual_pc(self.target_points)
            grasp_poses_camera = self.setting_contact_req(obstacle_points=self.obs_points, target_points=self.target_points)

            self.obs_points_base = self.pc_cam2base(self.obs_points)
            self.target_points_base = self.pc_cam2base(self.target_points)

            
            self.add_plane_2_obs_pc()
            # self.visual_pc(self.obs_points_base)
            self.actor.sim_furniture_id = self.actor.create_obstacle_from_pc(self.obs_points_base, self.target_points_base)
            
            self.grasp_list = []
            self.score_list = []
            for grasp_pose_cam in grasp_poses_camera:
                grasp_camera = np.array(grasp_pose_cam.pred_grasps_cam)
                grasp_world = self.pose_cam2base(grasp_camera.reshape(4,4))
                if grasp_world[0, 2] >= -0.3:
                    self.grasp_list.append(grasp_world)
                    self.score_list.append(grasp_pose_cam.score)

            
            self.actor.visualize_points_grasppose(self.obs_points_base, self.grasp_list)


            self.grasp_list = self.grasp2pre_grasp(self.grasp_list, drawback_dis=0.1) # Drawback a little
            
            (grasp_joint_list, grasp_poses_list,
            elbow_pos_list, grasp_score_list) = self.actor.grasp_pose2grasp_joint(grasp_poses=self.grasp_list,
                                                                                grasp_scores=self.score_list)
            
            grasp_joint_list = np.array(grasp_joint_list)
            elbow_pos_list = np.array(elbow_pos_list)
            grasp_poses_list = np.array(grasp_poses_list)

            if len(elbow_pos_list) == 0:
                print(f"There is no path")
                path_list = grasp_poses_list = elbow_path_list = gripper_pos_list = gripper_orn_list = None
            elif len(elbow_pos_list) == 1:
                grasp_joint_list = self.adjust_joint_values(grasp_joint_list)
                (path_list,
                elbow_path_list,
                gripper_pos_list,
                gripper_orn_list) = self.actor.motion_planning(grasp_joint_cfg=grasp_joint_list,
                                                                start_joint=self.actor.init_joint_pose[:6],
                                                                elbow_pos_list=elbow_pos_list,
                                                                grasp_poses_list=grasp_poses_list)
                print(f"grasp_joint_list: {grasp_joint_list}")
            else:
                (highest_joint_cfg_list,
                highest_elbow_pos_list,
                highest_grasp_poses_list) = self.actor.dbscan_grouping(elbow_pos_list,
                                                                        grasp_joint_list,
                                                                        grasp_score_list,
                                                                        grasp_poses_list,
                                                                        self.obs_points_base)
                print(f"highest_joint_cfg_list: {highest_joint_cfg_list}")

                for idx, joint_cfg in enumerate(highest_joint_cfg_list):
                    highest_joint_cfg_list[idx] = self.adjust_joint_values(joint_cfg)


                grasp_poses_list = highest_grasp_poses_list
                (path_list, 
                elbow_path_list, 
                gripper_pos_list, 
                gripper_orn_list) = self.actor.motion_planning(grasp_joint_cfg=highest_joint_cfg_list,
                                                                start_joint=self.actor.init_joint_pose[:6],
                                                                elbow_pos_list=highest_elbow_pos_list,
                                                                grasp_poses_list=grasp_poses_list)
            

            gripper_mat_list = np.array(self.actor.pos_orn2matrix(gripper_pos_list, gripper_orn_list))
            score_list = []

            for gripper_mat_path in  gripper_mat_list:
                score_list.append(self.path_quality_decision(gripper_mat_path))
            sorted_indices = np.argsort(score_list)
            score_list.sort()
            score_list.sort(reverse=True)
            print(f"score_list: {score_list}")

            path_list = np.array(path_list)[sorted_indices]
            
            exe_path_list = np.asarray(path_list[0])
            print(f"exe_path_list: {exe_path_list}")
            reverse_path_list = np.flip(exe_path_list, axis=0)
            # Moving along the path in joint space
            self.move_along_path_vel(exe_path_list)


            # Moving forward in cartesian space
            ef_pose = self.get_ef_pose()
            forward_mat = np.eye(4)
            forward_mat[2, 3] = 0.05
            ef_pose = ef_pose.dot(forward_mat)
            quat_pose = pack_pose(ef_pose)
            RT_grasp = [quat_pose[:3], ros_quat(quat_pose[3:])]
        
            self.set_pose(RT_grasp[0], RT_grasp[1])

            # Close gripper
            self.control_gripper("set_pose", 0.)
            time.sleep(1)

            # Go back to home position
            self.move_along_path_vel(reverse_path_list[:30])
            
            # self.move_along_path([self.home_joint_point])
            self.move_along_path([self.place_joint_point])
            # Open gripper
            self.control_gripper("set_pose", 0.085)
            
            retreat_path = np.linspace(self.place_joint_point[:6], self.home_joint_point[:6], num=5)
            print(f"retreat_path: {retreat_path}")
            
            self.move_along_path_vel(retreat_path)
            # self.move_along_path([self.home_joint_point])
            print(f"finish grasping")
        
        elif msg.data == 1:
            # Pybullet setup
            self.actor.init_joint_pose = self.joint_states
            self.actor.env._panda.reset(self.actor.init_joint_pose)
            if self.actor.sim_furniture_id is not None:
                self.actor.remove_sim_fureniture()
                self.actor.sim_furniture_id = None
            self.actor.replace_real_furniture()
            

            # Reset the gripper
            self.control_gripper("reset")
            time.sleep(1)
            self.control_gripper("set_pose", 0.)
            time.sleep(1)
            self.control_gripper("set_pose", 0.085)
            print(f"finish grasping")


            # Reset the arm's position
            self.move_along_path([self.home_joint_point])
            time.sleep(1)
            # preset_path = [[-0.1045372703283405, -0.576734232246587, 1.5737202113556945, 0.031900739615946805, 1.6365758730876359, 0.11640376686875875],
            #                [0.5550416533244285, -0.3928794775730634, 1.5781743482788522, -0.139025173155353, 1.6365855936256146, 0.11641346243960232],
            #                [0.5547865890710261, -0.17408155268417816, 1.5933722096716072, -0.4246386056786398, 1.636478934023957, 0.11665586835544839],
            #                [0.5662767310151456, -0.0032597910976694265, 1.6183618486023112, -0.810143057823974, 1.0120194568835776, 0.3339396613289475],
            #                [0.5963862305624998, 0.4135828918873394, 1.4569587070511325, -0.9186928380653313, 0.4095706066785396, 0.6347956713841676],
            #                [0.47400099556304465, 0.8366527625258681, 1.1346312603127713, -1.5057343692578793, -0.06861083191019726, 0.34546851897741415]]
            preset_path = [[-0.1045372703283405, -0.576734232246587, 1.5737202113556945, 0.031900739615946805, 1.6365758730876359, 0.11640376686875875],
                           [0.1153223708892492, -0.5154493146887465, 1.5752049239960803, -0.0257418988984868, 1.6365791139336287, 0.1164069980590066],
                           [0.3351820121800661, -0.45416439713090605, 1.576689636636466, -0.08338453741292741, 1.6365823547796214, 0.11641022924925444],
                           [0.5550416533244285, -0.3928794775730634, 1.5781743482788522, -0.139025173155353, 1.6365855936256146, 0.11641346243960232],
                           [0.5549569658236277, -0.3190135023134347, 1.5839069687391038, -0.23489698482911527, 1.636550386091062, 0.1164949317442337],
                           [0.5548722783238274, -0.24514752705380593, 1.5896395891993555, -0.3307687965028776, 1.6365151785565096, 0.11657640074515307],
                           [0.5547865890710261, -0.17408155268417816, 1.5933722096716072, -0.4246386056786398, 1.636478934023957, 0.11665586835544839],
                           [0.5584466363850659, -0.11713563282934257, 1.601032755315842, -0.5538067563930845, 1.4288257749776972, 0.18941779934661443],
                           [0.5621066836991058, -0.061189712974507, 1.6086933009600768, -0.6829749071075291, 1.221172615931437, 0.2621797303377804],
                           [0.5662767310151456, -0.0032597910976694265, 1.6183618486023112, -0.810143057823974, 1.0120194568835776, 0.3339396613289475],
                           [0.576313231531597, 0.1376881038973332, 1.564560134085585, -0.8469936512370935, 0.9445365066485649, 0.4348916646800209],
                           [0.5863497320480484, 0.2786359988923358, 1.5107584195688598, -0.883844244650213, 0.8770535564135521, 0.5358436680310943],
                           [0.5963862305624998, 0.4135828918873394, 1.4569587070511325, -0.9186928380653313, 0.4095706066785396, 0.6347956713841676],
                           [0.5222578182293487, 0.5546068487661823, 1.3494165588053455, -1.114387348193514, 0.25084331595277034, 0.5382432875819165],
                           [0.4481294058961976, 0.6956308056450253, 1.2418744105595584, -1.3100818583216966, 0.0921150252263734, 0.4416909037796658],
                           [0.47400099556304465, 0.8366527625258681, 1.1346312603127713, -1.5057343692578793, -0.06861083191019726, 0.34546851897741415]]

            preset_path = np.array(preset_path)
            reverse_path_list = np.flip(preset_path, axis=0)
            
            self.move_along_path_vel(preset_path)

            # Moving forward in cartesian space
            ef_pose = self.get_ef_pose()
            forward_mat = np.eye(4)
            forward_mat[2, 3] = 0.05
            ef_pose = ef_pose.dot(forward_mat)
            quat_pose = pack_pose(ef_pose)
            RT_grasp = [quat_pose[:3], ros_quat(quat_pose[3:])]
        
            self.set_pose(RT_grasp[0], RT_grasp[1])

            # Close gripper
            self.control_gripper("set_pose", 0.)
            time.sleep(1)

            self.move_along_path_vel(reverse_path_list)
            self.move_along_path([self.place_joint_point])
            # Open gripper
            self.control_gripper("set_pose", 0.085)

            retreat_path = np.linspace(self.place_joint_point[:6], self.home_joint_point[:6], num=5)
            print(f"retreat_path: {retreat_path}")
            self.move_along_path_vel(retreat_path)

            
        elif msg.data == 2:
            print(f"ruckig cartesian testing")
            ef_pose = self.get_ef_pose()
            forward_mat = np.eye(4)
            forward_mat[2, 3] = 0.02
            ef_pose = ef_pose.dot(forward_mat)
            quat_pose = pack_pose(ef_pose)
            RT_grasp = [quat_pose[:3], ros_quat(quat_pose[3:])]
        
            self.set_pose(RT_grasp[0], RT_grasp[1])
            print(f"move forward a little")
            self.control_gripper("reset")
            time.sleep(2)
            self.control_gripper("set_pose", 0.)
            time.sleep(2)
            self.control_gripper("set_pose", 0.085)
            print(f"finish grasping")

        elif msg.data == 3:
            # Pybullet setup
            self.actor.init_joint_pose = self.joint_states
            self.actor.env._panda.reset(self.actor.init_joint_pose)
            if self.actor.sim_furniture_id is not None:
                self.actor.remove_sim_fureniture()
                self.actor.sim_furniture_id = None
            self.actor.replace_real_furniture()
            

            # Reset the gripper
            self.control_gripper("reset")
            time.sleep(1)
            self.control_gripper("set_pose", 0.)
            time.sleep(1)
            self.control_gripper("set_pose", 0.085)
            print(f"finish grasping")


            # Reset the arm's position
            self.move_along_path([self.home_joint_point])


            # Set init_value to None
            self.target_points = None
            self.obs_points = None
            
            

            # Segmentation part
            seg_msg = Int32()
            seg_msg.data = 2
            self.seg_pub.publish(seg_msg)    
            
            time.sleep(2)
            self.target_points_base = self.pc_cam2base(self.target_points)
            self.obs_points_base = self.pc_cam2base(self.obs_points)
            # self.visual_pc(self.target_points_base)
            # self.visual_pc(self.obs_points_base)
            self.visual_pc(self.remove_outlier_points(self.target_points_base))
            self.add_plane_2_obs_pc()
            self.visual_pc(self.obs_points_base)
        elif msg.data == 4:
            # Pybullet setup
            self.actor.init_joint_pose = self.joint_states
            self.actor.env._panda.reset(self.actor.init_joint_pose)
            if self.actor.sim_furniture_id is not None:
                self.actor.remove_sim_fureniture()
                self.actor.sim_furniture_id = None
            self.actor.replace_real_furniture()
            # Segmentation part
            seg_msg = Int32()
            seg_msg.data = 2
            self.seg_pub.publish(seg_msg)
            time.sleep(1)
            self.target_points_base = self.pc_cam2base(self.target_points)
            self.obs_points_base = self.pc_cam2base(self.obs_points)
            # self.visual_pc(self.target_points_base)
            # self.visual_pc(self.obs_points_base)
            self.visual_pc(self.remove_outlier_points(self.target_points_base))
        elif msg.data == 5:
            # Pybullet setup
            self.actor.init_joint_pose = self.joint_states
            self.actor.env._panda.reset(self.actor.init_joint_pose)
            if self.actor.sim_furniture_id is not None:
                self.actor.remove_sim_fureniture()
                self.actor.sim_furniture_id = None
            self.actor.replace_real_furniture()
            

            # Reset the gripper
            self.control_gripper("reset")
            time.sleep(1)
            self.control_gripper("set_pose", 0.)
            time.sleep(1)
            self.control_gripper("set_pose", 0.085)
            print(f"finish grasping")


            # Reset the arm's position
            self.move_along_path([self.home_joint_point])
            time.sleep(1)


            # Set init_value to None
            self.target_points = None
            self.obs_points = None

            seg_msg = Int32()
            seg_msg.data = 2
            self.seg_pub.publish(seg_msg)    
            time.sleep(5) # Sleep to wait for the segmentation po
            

            self.target_points = self.remove_outlier_points(self.target_points)
            self.obs_points_base = self.pc_cam2base(self.obs_points)
            self.target_points_base = self.pc_cam2base(self.target_points)

            
            self.add_plane_2_obs_pc()
            # self.visual_pc(self.obs_points_base)
            self.actor.sim_furniture_id = self.actor.create_obstacle_from_pc(self.obs_points_base, self.target_points_base)
        elif msg.data == 6:
            # Pybullet setup
            self.actor.init_joint_pose = self.joint_states
            self.actor.env._panda.reset(self.actor.init_joint_pose)
            if self.actor.sim_furniture_id is not None:
                self.actor.remove_sim_fureniture()
                self.actor.sim_furniture_id = None
            self.actor.replace_real_furniture()
            

            # Reset the gripper
            self.control_gripper("reset")
            time.sleep(1)
            self.control_gripper("set_pose", 0.)
            time.sleep(1)
            self.control_gripper("set_pose", 0.085)
            print(f"finish grasping")


            # Reset the arm's position
            self.move_along_path([self.home_joint_point])
            time.sleep(1)
            forward_point = [0.1111004061306238, -0.41889559865642484, 1.5924701970103974, 0.06397601199064548, 1.557803296468618, 0.18963972312766425]

            forward_path = np.linspace( self.home_joint_point[:6], forward_point, num=5)
            self.move_along_path_vel(forward_path)
            time.sleep(2)
            retreat_path = np.linspace(forward_point, self.home_joint_point[:6], num=5)
            self.move_along_path_vel(retreat_path)


    def points_callback(self, msg):
        self.target_points = self.pc2_tranfer(msg)

    def obs_points_callback(self, msg):
        self.obs_points = self.pc2_tranfer(msg)

    def pc2_tranfer(self, ros_msg):
        points = point_cloud2.read_points_list(
                ros_msg, field_names=("x", "y", "z"))
        return np.asarray(points)
        

    def pc_cam2base(self, pc, crop=True):
        transform_stamped = self.tf_buffer.lookup_transform('base', 'camera_color_optical_frame', rospy.Time(0))
        trans = np.array([transform_stamped.transform.translation.x,
                            transform_stamped.transform.translation.y,
                            transform_stamped.transform.translation.z])
        quat = np.array([transform_stamped.transform.rotation.x,
                        transform_stamped.transform.rotation.y,
                        transform_stamped.transform.rotation.z,
                        transform_stamped.transform.rotation.w])
        T = quaternion_matrix(quat)
        T[:3, 3] = trans
        T_inv = np.linalg.inv(T)
        o3d_pc = o3d.geometry.PointCloud()
        o3d_pc.points = o3d.utility.Vector3dVector(pc)
        o3d_pc.transform(T)
        self.bounds = [[-0.05, 1.1], [-0.5, 0.5], [-0.12, 2]]  # set the bounds
        bounding_box_points = list(itertools.product(*self.bounds))  # create limit points
        self.bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(bounding_box_points))  # create bounding box object
        if crop:
            o3d_pc.crop(self.bounding_box)
        return np.asarray(o3d_pc.points)


    def pose_cam2base(self, poses):
        transform_stamped = self.tf_buffer.lookup_transform('base', 'camera_color_optical_frame', rospy.Time(0))
        trans = np.array([transform_stamped.transform.translation.x,
                            transform_stamped.transform.translation.y,
                            transform_stamped.transform.translation.z])
        quat = np.array([transform_stamped.transform.rotation.x,
                        transform_stamped.transform.rotation.y,
                        transform_stamped.transform.rotation.z,
                        transform_stamped.transform.rotation.w])
        T = quaternion_matrix(quat)
        T[:3, 3] = trans

        return np.dot(T, poses)

    def remove_outlier_points(self, pointcloud):
        dbscan = DBSCAN(eps=0.05, min_samples=10)  # You may need to adjust these parameters based on your data

        # Fit DBSCAN to the point cloud data
        dbscan.fit(pointcloud)

        # Get labels assigned to each point by DBSCAN
        labels = dbscan.labels_

        # Find the label with the most points (excluding outliers labeled as -1)
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        major_segment_label = unique_labels[np.argmax(label_counts[label_counts > 0])]
        
        # Append he outlier into the obs_points
        outlier_points = pointcloud[labels == -1]
        self.obs_points = np.concatenate((self.obs_points, outlier_points), axis=0)

        # Extract points belonging to the major segment
        return pointcloud[labels == major_segment_label]
    
    def add_plane_2_obs_pc(self):
        x_min, x_max = 0.3, 1.1
        y_min, y_max = -0.5, 0.5
        z = 0.02

        # Define the number of points along x and y axes
        num_points_x = 50
        num_points_y = 50

        # Generate grid of points on the x-y plane
        x = np.linspace(x_min, x_max, num_points_x)
        y = np.linspace(y_min, y_max, num_points_y)
        x_grid, y_grid = np.meshgrid(x, y)

        # Constant z-coordinate for the plane
        z_points = np.full_like(x_grid, z)

        # Flatten the grid into 1D arrays
        x_points = x_grid.flatten()
        y_points = y_grid.flatten()

        # Combine x, y, and z coordinates to form the point cloud
        plane_pc = np.column_stack((x_points, y_points, z_points.flatten()))
        
        self.obs_points_base = np.concatenate((self.obs_points_base, plane_pc), axis=0)

    def visual_pc(self, pc):
        o3d_pc = o3d.geometry.PointCloud()
        o3d_pc.points = o3d.utility.Vector3dVector(pc)
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([o3d_pc, axes])


    def setting_contact_req(self, obstacle_points, target_points):
        contact_request = GraspGroupRequest()
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
    

    def grasp2pre_grasp(self, grasp_poses, drawback_dis=0.02):
        # This function will make the grasp poses retreat a little
        drawback_matrix = np.identity(4)
        drawback_matrix[2, 3] = -drawback_dis

        result_poses = []
        for i in range(len(grasp_poses)):
            grasp_candidate = np.dot(grasp_poses[i], drawback_matrix)
            result_poses.append(grasp_candidate)
        return np.array(result_poses)


    def adjust_joint_values(self, joint_values):
        # This function adjust the value outside the range into the range
        adjusted_values = []
        for value, min_limit, max_limit in zip(joint_values,
                                               self.actor.env._panda._joint_min_limit[:6],
                                               self.actor.env._panda._joint_max_limit[:6]):
            while value > max_limit:
                value -= 2 * np.pi
            while value < min_limit:
                value += 2 * np.pi
            adjusted_values.append(value)
        return adjusted_values
        

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
        for idx in range(len(waypoint_mat[:-1])):
            moving_vec = waypoint_mat[idx+1][:3, 3] - waypoint_mat[idx][:3, 3]
            moving_vec = moving_vec / np.linalg.norm(moving_vec)
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
        return final_score
    

    def get_ef_pose(self):
        """
        (4, 4) end effector pose matrix from base
        """
        try:
            tf_pose = self.tf_buffer.lookup_transform("base",
                                                      # source frame:
                                                      "flange_link",
                                                      rospy.Time(0),
                                                      rospy.Duration(1.0))
            tf_pose = self.unpack_tf(tf_pose)
            pose = self.make_pose(tf_pose)
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):

            pose = None
            print('cannot find end-effector pose')
            sys.exit(1)
        return pose
    
    def make_pose(self, tf_pose):
        """
        Helper function to get a full matrix out of this pose
        """
        trans, rot = tf_pose
        pose = tf.transformations.quaternion_matrix(rot)
        pose[:3, 3] = trans
        return pose
    
    def unpack_tf(self, transform):
        if isinstance(transform, TransformStamped):
            return np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]), \
                   np.array([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w])
        elif isinstance(transform, Pose):
            return np.array([transform.position.x, transform.position.y, transform.position.z]), \
                   np.array([transform.orientation.x, transform.orientation.y, transform.orientation.z, transform.orientation.w])
    

    def set_joint(self, joint_position):
        """
        Send goal joint value to ruckig to move
        """
        target_joint = JointState()
        target_joint.position = joint_position
        target_joint.velocity = [0, 0, 0, 0, 0, 0]
        self.joint_goal = np.concatenate((joint_position, [0, 0, 0]))
        print("Move tm joints to position: {}".format(target_joint.position))
        self.tm_joint_pub.publish(target_joint)
        return self.loop_confirm(mode="joint")
    
    def set_joint_vel(self, joint_position, velocity):
        """
        Send goal joint and goal velocity to ruckig to move
        """
        target_joint = JointState()
        target_joint.position = joint_position
        target_joint.velocity = velocity
        self.joint_goal = np.concatenate((joint_position, [0, 0, 0]))
        print("Move tm joints to position: {}".format(target_joint.position))
        self.tm_joint_pub.publish(target_joint)
        return self.loop_confirm(mode="joint")


    def set_pose(self, pos, orn):
        """
        Send goal cartesian value to ruckig to move
        """
        target_pose = Pose()
        target_pose.position.x = pos[0]
        target_pose.position.y = pos[1]
        target_pose.position.z = pos[2]
        target_pose.orientation.x = orn[0]
        target_pose.orientation.y = orn[1]
        target_pose.orientation.z = orn[2]
        target_pose.orientation.w = orn[3]
        self.pose_goal = target_pose

        print("Move end effector to position: {}".format(target_pose))
        self.tm_pub.publish(target_pose)

        return self.loop_confirm(mode="cart")
        

    def loop_confirm(self, mode="joint"):
        if mode == "joint":
            threshold=0.01
            while True:
                dis = np.linalg.norm(self.joint_states-self.joint_goal)
                # print(f"dis: {dis}")
                if dis < threshold:
                    break
            return True
        else:
            threshold=0.01
            transform_ef = self.tf_buffer.lookup_transform("base",
                                                            # source frame:
                                                            "flange_link",
                                                            rospy.Time(0),
                                                            rospy.Duration(1.0))
            ef_pos, ef_orn = self.unpack_tf(transform_ef)
            target_pos, target_orn = self.unpack_tf(self.pose_goal)
            dis = np.abs(ef_pos - target_pos)
            while True:
                transform_ef = self.tf_buffer.lookup_transform("base",
                                                            # source frame:
                                                            "flange_link",
                                                            rospy.Time(0),
                                                            rospy.Duration(1.0))
                ef_pos, ef_orn = self.unpack_tf(transform_ef)
                target_pos, target_orn = self.unpack_tf(self.pose_goal)
                dis = np.sum(np.abs(ef_pos - target_pos))
                # print(f"dis: {dis}")
                if dis < threshold:
                    break
            return True


    def move_along_path(self, path):
        for waypoint in path:
            self.set_joint(waypoint)
            self.loop_confirm()

    def move_along_path_vel(self, path):
        # First calculate the velocity
        joint_velocities = []
        delta_t = 10
        # Loop through waypoints to compute velocities
        for i in range(len(path) - 1):
            # Difference between consecutive waypoints
            delta_q = path[i+1] - path[i]
            
            # Compute the time interval needed for the max velocity constraint
            velocities = delta_q / delta_t
            joint_velocities.append(velocities)
        joint_velocities.append([0, 0, 0, 0, 0, 0])
        for waypoint, velocity in zip(path, joint_velocities):
            self.set_joint_vel(waypoint, velocity)
            self.loop_confirm()

    def control_gripper(self, type, value=0):
        gripper_command = Robotiq2FGripper_robot_output()
        if type == "reset":
            gripper_command.rACT = 0
            gripper_command.rGTO = 0
            gripper_command.rATR = 0
            gripper_command.rSP = 0
            gripper_command.rFR = 0
            gripper_command.rPR = 0
        elif type == "set_pose":
            if value > 0.085 or value < 0:
                raise ValueError("Error invalid valur for gripper open length")

            uint_value = int(255 - value / 0.085 * 255)
            gripper_command.rACT = 1
            gripper_command.rGTO = 1
            gripper_command.rSP = 200
            gripper_command.rFR = 170
            gripper_command.rPR = uint_value

        self.robotiq_pub.publish(gripper_command)
    

if __name__ == "__main__":
    rospy.init_node("test_realworld")
    real_actor_node = ros_node(renders=False)
    rospy.spin()