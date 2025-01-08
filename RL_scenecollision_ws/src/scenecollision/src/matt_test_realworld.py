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
from scenecollision.srv import GraspGroup, GraspGroupRequest, SetPositions, SetPositionsRequest
from scenecollision.srv import path_planning, path_planningRequest
from scenecollision.msg import GraspPose, motion_planning, Robotiq2FGripper_robot_output, FeedbackState
from sensor_msgs.point_cloud2 import create_cloud_xyz32
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, TransformStamped

class ros_node(object):
    def __init__(self, renders):
        self.actor = ActorWrapper(renders=renders)
        self.start_sub = rospy.Subscriber("test_realworld_cmd", Int32, self.get_env_callback)
        self.state_sub = rospy.Subscriber("feedback_states", FeedbackState, self.state_callback)
        self.tm_pub = rospy.Publisher("/target_position", Pose, queue_size=1)
        self.tm_joint_pub = rospy.Publisher("/target_joint", JointState, queue_size=1)
        self.position_serivce_client = rospy.ServiceProxy("tm_driver/set_positions", SetPositions)
        self.robotiq_pub = rospy.Publisher("/Robotiq2FGripperRobotOutput", Robotiq2FGripper_robot_output, queue_size=10)
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)  # Create a tf listener

        self.points_sub = rospy.Subscriber("/uoais/Pointclouds", PointCloud2, self.points_callback)
        self.obs_points_sub = rospy.Subscriber("/uoais/obs_pc", PointCloud2, self.obs_points_callback)
        self.seg_pub = rospy.Publisher("/uoais/data_init", Int32, queue_size=1)
        self.depth_topic = rospy.get_param("~depth", "/camera/aligned_depth_to_color/image_raw")

        self.raw_point_sub = rospy.Subscriber("/camera/depth/color/points", PointCloud2, self.raw_points_callback)
        self.contact_client = rospy.ServiceProxy('contact_graspnet/get_grasp_result', GraspGroup)
        # rospy.wait_for_service('contact_graspnet/get_grasp_result', timeout=None)
        self.target_points = None
        self.obs_points = None
        self.raw_point_flag = False
        # self.home_joint_point = [-0.0432979892528877, -1.7933704143724325, 2.502642517484129, -0.5873226944246331, 1.6095232164185018, -0.03901780668318279]
        # self.home_joint_point = [0.012591255025137305, -1.2207295273003245, 1.5559966079851082, 0.023707389283668924, 1.5601789693190231, -0.04487435591399763]
        self.home_joint_point = [-0.030949352364245817, -1.2832006285708581, 1.5177811784488526, 0.0606198176669007, 1.6131496428748628, -0.04860742116607019]
        self.place_joint_point = [-1.002553783421109, -0.3444243268035077, 2.1679726506737955, -0.7325243623262604, 1.320991283913246, 0.6469062632832484]
        rospy.loginfo("Init finished!!!!!!!!!!!!!!!!!!!!!!!!!!!")


    def state_callback(self, msg):
        cur_states = np.asarray(msg.joint_pos)
        self.joint_states = np.concatenate((cur_states, [0, 0, 0]))
        quat_pose = pack_pose(self.get_ef_pose())
        self.ef_state = np.array([*quat_pose[:3], *quat2euler(quat_pose[3:])]) # This pose is a 1d array(x, y, z and euler)

    def get_env_callback(self, msg):
        if msg.data == 1:
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
            # preset_path = [[0.38359157011335754, -0.886856078813979, 1.4877649561186785, 0.05883698909771443, 1.530566482210105, 0.18520852267939503],
            #                [0.48071345980107577, -0.5103999494987844, 1.8878669598575506, -0.8928328780903894, 1.1777869868291782, -0.0009599310833953948],
            #                [0.6668263753419049, -0.30597247514148035, 1.980043226242937, -0.8448847931923847, 0.9140192566683513, 0.16450697257788208],
            #                [0.6191612513097212, 0.044399102333905385, 1.7862784163895513, -1.0815902127251262, 0.9140095361303724, 0.16453606761279121],
            #                [0.5060452483528277, 0.27412106850413265, 1.5344720079487064, -1.0815805587661746, 0.9139416921016027, 0.16453606761279121],
            #                [0.5184872040705416, 0.5214359558980183, 1.3437183011818188, -1.081502994199426, 0.3151385762708287, 0.3792115689370451],
            #                [0.494959474130723, 0.5929709262015549, 1.3843598704711364, -1.0814641786265382, 0.2532472784591189, 0.7049288202689138],]
            preset_path = [[0.21426284451334612, -1.0442716026859549, 1.5188760039729843, 0.06808723363394167, 1.6171250765920786, -0.05538511620605141],
                           [0.3669442500078131, -0.6767612977884455, 1.5816270034740854, 0.06804844718937819, 1.6172317361937363, -0.055375420635207846],
                           [0.3671308710212004, -0.3281377984466417, 1.510558685572558, 0.06754424005480973, 1.6172317361937363, -0.05536572090317507],
                           [0.36721173124979795, 0.09827274978566074, 1.5105773277001884, -0.37321925674728507, 1.6172221488138119, -0.05536572090317507],
                           [0.30468470359637256, 0.5356508778219389, 1.1361243615779242, -0.37322897728526383, 1.6171347971300574, -0.05528815217523731],
                           [0.248036520825513, 0.8120612661781309, 0.802512036036461, -0.37321925674728507, 1.6170378580663785, -0.05528815217523731],
                           [0.24058378102822395, 0.8958329290583636, 0.823066180171638, -0.47384719801009256, 1.4270490111645615, -0.05531724304895722],]
            # preset_paths = [[[-0.030930689430669248, -0.6324430348771916, 1.5178496216888677, 0.06136771618021107, 1.6131690839508204, -0.051138148248566835],
            #                 [-0.04447997504607053, -0.40536497597460014, 1.517843363260306, 0.061057432946081625, 1.1995356918713103, 0.1853151822810527],
            #                 [-0.08447997504607053, -0.32536497597460014, 1.597843363260306, 0.060557432946081625, 1.1995356918713103, 0.1853151822810527],
            #                 [-0.11011126648987635, -0.21128920877443408, 1.6677501710178309, 0.06098955978898743, 1.1995648534852468, 0.18530547838783074],
            #                 [-0.11099464534583403, -0.17436772598806397, 1.9675515359918658, -0.8618629777738579, 1.2927944000817595, 0.18530547838783074],
            #                 [-0.1439471525806318, 0.023677033408431223, 1.9098767878445648, -1.3827080448050149, 1.6198691977792963, 0.0339563504375949],
            #                 [-0.11892638795468746, 0.17923252235131173, 1.742694320411805, -1.4240819833941043, 1.7504197512599615, 0.03397574365987664],
            #                 [-0.09733338629016254, 0.22258650456611834, 1.7426445192994207, -1.4241401734639225, 1.7504197512599615, 0.03397574365987664]],
            #                [[-0.23311239910330434, -0.8234518724765088, 1.731359640496319, -0.5721964720652011, 1.7502549015884856, 0.03415997198947208],
            #                 [-0.23310617396425629, -0.4219127937239165, 1.792312873103562, -0.5723031316668588, 1.7502743426644432, 0.03415997198947208],
            #                 [-0.23326792771096497, -0.12842581698397285, 1.817122082870413, -0.572361321736677, 1.7502549015884856, 0.03415997198947208],
            #                 [-0.23316839206522325, -0.03633671919001596, 1.8975157256449622, -1.003835829169873, 1.6157094733146469, 0.03410179440322146],
            #                 [-0.23307506491377278, 0.1231877619491937, 1.8983182692394576, -1.4205331880835192, 1.4698581272650948, 0.034111489974065025],
            #                 [-0.23307506491377278, 0.2175164117723391, 1.8931361572323269, -1.629895067177063, 1.469664249137737, 0.22577773721084665],
            #                 [-0.2515078181483768, 0.3449530974379429, 1.7882940298605778, -1.704692210078865, 1.3716541952264774, 0.22522504806087115]],
            #                [[0.2355136882344447, -0.7850809143325926, 1.7106936435953586, -0.19987897830777365, 1.8082871791121298, -0.24913605681582265],
            #                 [0.23648416078042642, -0.5223628691154203, 1.761295035887409, -0.19828878817613052, 1.8085004983154451, -0.24913605681582265],
            #                 [0.23650281955281377, -0.3536313407451024, 1.761307419586478, -0.35499028527034254, 2.020558029064367, -0.2491554479575098],
            #                 [0.3176616883770834, -0.16276559812010652, 1.7612514932035863, -0.35511631936893057, 2.0205676164442914, -0.24914574406428783],
            #                 [0.3110612100628901, -0.06242748784649448, 1.7750993982397802, -0.8037822972974016, 1.8073562711530926, -0.24909727453244837],
            #                 [0.2717135542339442, -0.005511784302352675, 1.8766008563401337, -1.1928646064987163, 1.875627737277819, -0.24917483909919694],
            #                 [0.4250917092486032, 0.030762725109074712, 1.827131573827476, -1.1927773879730161, 1.548679173415952, 0.07280931785227479]]]

            # for i in range(3):
            #     preset_path = np.array(preset_paths[i])
            #     reverse_path_list = np.flip(preset_path, axis=0)[:-1]
                
            #     self.move_along_path_vel(preset_path)

            #     # Moving forward in cartesian space
            #     ef_pose = self.get_ef_pose()
            #     forward_mat = np.eye(4)
            #     forward_mat[2, 3] = 0.05
            #     ef_pose = ef_pose.dot(forward_mat)
            #     quat_pose = pack_pose(ef_pose)
            #     RT_grasp = [quat_pose[:3], ros_quat(quat_pose[3:])]
            
            #     self.set_pose(RT_grasp[0], RT_grasp[1])

            #     # Close gripper
            #     self.control_gripper("set_pose", 0.)
            #     time.sleep(1)

            #     self.move_along_path_vel(reverse_path_list)
            #     place_path = np.linspace(reverse_path_list[-1], self.place_joint_point, num=8)
            #     self.move_along_path(place_path)
            #     # Open gripper
            #     self.control_gripper("set_pose", 0.085)
            #     time.sleep(1)

            #     retreat_path = np.linspace(self.place_joint_point[:6], self.home_joint_point[:6], num=8)
            #     print(f"retreat_path: {retreat_path}")
            #     self.move_along_path_vel(retreat_path)
            #     time.sleep(15)

            
            preset_path = np.array(preset_path)
            reverse_path_list = np.flip(preset_path, axis=0)[:-1]
            
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
            place_path = np.linspace(reverse_path_list[-1], self.place_joint_point, num=8)
            self.move_along_path(place_path)
            # Open gripper
            self.control_gripper("set_pose", 0.085)
            time.sleep(1)

            retreat_path = np.linspace(self.place_joint_point[:6], self.home_joint_point[:6], num=8)
            print(f"retreat_path: {retreat_path}")
            self.move_along_path_vel(retreat_path)
            time.sleep(15)
        elif msg.data == 2:
            # Directly joint Position set testing
            print(f"Enter preset path part")
            preset_path = [[-0.030949352364245817, -1.2832006285708581, 1.5177811784488526, 0.0606198176669007, 1.6131496428748628, -0.04860742116607019],
                           [-0.04447997504607053, -0.40536497597460014, 1.517843363260306, 0.061057432946081625, 1.1995356918713103, 0.1853151822810527],
                           [-0.030949352364245817, -1.2832006285708581, 1.5177811784488526, 0.0606198176669007, 1.6131496428748628, -0.04860742116607019]]
            
            self.move_along_path_dir(preset_path)
            
            # Moving forward in cartesian space
            ef_pose = self.get_ef_pose()
            forward_mat = np.eye(4)
            forward_mat[2, 3] = 0.32
            ef_pose = ef_pose.dot(forward_mat)
            quat_pose = pack_pose(ef_pose)
            forward_pose = np.array([*quat_pose[:3], *quat2euler(quat_pose[3:])]) # This pose is a 1d array(x, y, z and euler)
            self.dir_set_position(forward_pose, mode="cart")

                
        elif msg.data == 4:
            place_joint_point_list = [[-0.9896016991965904, -0.17409400296227426, 2.157353428985159, -0.7175921512522406, 1.6132660230144993, -0.048665598752320814],
                                      [-1.2017494445274386, 0.21056136684851542, 1.4655062557277962, -0.3963933521838548, 1.6157869713023683, 0.03171651128635137],
                                      [-1.4774008652624717, 0.26199017000173996, 1.5582050343190565, -0.5514077034379881, 1.6157579428464863, 0.03193952606051029],
                                      [-1.518838719717723, -0.16715761681083796, 1.8664170614897253, -0.33952470960900716, 1.6157773839224439, 0.03160015611385013],]
            for times in range(4):
                # Main operation
                # Pybullet setup
                self.actor.init_joint_pose = self.joint_states
                self.actor.env._panda.reset(self.actor.init_joint_pose)
                if self.actor.sim_furniture_id is not None:
                    self.actor.remove_sim_fureniture()
                    self.actor.sim_furniture_id = None
                self.actor.replace_real_furniture()

                # Set middle waypoint
                middle_point_l = [0.6044797768523323, -0.4638172672916634, 1.9192456554014092, -0.7220135978730173, 1.4285132171318873, 0.16713466369119284]
                middle_point_r = [-0.44463801335875336, -0.34688159225181014, 1.955122030552248, -0.8672056118156931, 1.5888701358488284, -0.077308387296449]
                middle_point_m = [0.0757963692651781, -0.6608915874316905, 1.9553211018437315, -0.5384243936496882, 1.5888411073929465, -0.07726960501307473]

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

                # # Segmentation part
                seg_msg = Int32()
                seg_msg.data = 2
                self.seg_pub.publish(seg_msg)
                while(self.target_points is None):
                    time.sleep(0.05) # Sleep to wait for the segmentation pointcloud arrive
                print(f"finished segmentation")
                # self.visual_pc(self.target_points)

                
                # Seperate the pointcloud of objects and assign the first for grasping task
                object_pc_list = self.seperate_object_pointcloud(self.target_points)
                # for object_pc in object_pc_list:
                #     self.visual_pc(object_pc)
                print(f"object_pc_list: {len(object_pc_list)}")
                # Choose one as target object
                cur_target_idx = self.find_optimal_point_cloud_index([self.pc_cam2base(object_pc) for object_pc in object_pc_list])
                self.target_points = object_pc_list.pop(cur_target_idx)

                # Add other objects' points to obstacle pointcloud
                for object_pc in object_pc_list:
                    down_pc = regularize_pc_point_count(object_pc, 256)
                    
                    self.obs_points = np.concatenate((self.obs_points, down_pc), axis=0)
                
                # # convert frame from camera to world(base)
                self.obs_points_base = self.pc_cam2base(self.obs_points)
                self.target_points_base = self.pc_cam2base(self.target_points)
                _, self.target_points_base = self.collect_plane_points(self.target_points_base)
                # object_pc_list = [self.pc_cam2base(i) for i in object_pc_list]
                # self.visual_pc(self.target_points_base)

                # self.add_plane_2_obs_pc()
                self.add_layer_plane_2_obs_pc()
                # self.visual_pc(self.obs_points_base)

                self.actor.sim_furniture_id = self.actor.create_obstacle_from_pc(self.obs_points_base, self.target_points_base)
                

                # Get grasp poses and filter out those facing negative x-axis
                grasp_poses_camera = self.setting_contact_req(obstacle_points=self.obs_points, target_points=self.target_points)
                self.grasp_list = []
                self.score_list = []
                for grasp_pose_cam in grasp_poses_camera:
                    grasp_camera = np.array(grasp_pose_cam.pred_grasps_cam)
                    grasp_world = self.pose_cam2base(grasp_camera.reshape(4,4))
                    if grasp_world[0, 2] >= -0.3:
                        self.grasp_list.append(grasp_world)
                        self.score_list.append(grasp_pose_cam.score)

                print(f"self.grasp_list: {self.grasp_list}")
                print(f"self.score_list: {self.score_list}")
                self.actor.visualize_points_grasppose(self.obs_points_base, self.grasp_list)

                # Retreat the grasp poses a little and convert them to joint space
                self.grasp_list = self.grasp2pre_grasp(self.grasp_list, drawback_dis=0.1) # Drawback a little
                (grasp_joint_list, grasp_poses_list,
                elbow_pos_list, grasp_score_list) = self.actor.grasp_pose2grasp_joint(grasp_poses=self.grasp_list,
                                                                                    grasp_scores=self.score_list)
                
                grasp_joint_list = np.array(grasp_joint_list)
                elbow_pos_list = np.array(elbow_pos_list)
                grasp_poses_list = np.array(grasp_poses_list)

                if np.mean(self.target_points_base, axis=0)[1] > 0.1:
                    middle_point = middle_point_l
                elif np.mean(self.target_points_base, axis=0)[1] < -0.1:
                    middle_point = middle_point_r
                else:
                    middle_point = middle_point_m

                # Motion planning part
                if len(elbow_pos_list) == 0:
                    print(f"There is no path")
                    path_list = grasp_poses_list = elbow_path_list = gripper_pos_list = gripper_orn_list = None
                elif len(elbow_pos_list) == 1:
                    grasp_joint_list = self.adjust_joint_values(grasp_joint_list)
                    (path_list,
                    elbow_path_list,
                    gripper_pos_list,
                    gripper_orn_list) = self.actor.motion_planning(grasp_joint_cfg=grasp_joint_list,
                                                                    start_joint=middle_point,
                                                                    elbow_pos_list=elbow_pos_list,
                                                                    grasp_poses_list=grasp_poses_list,
                                                                    target_pointcloud=self.target_points_base)
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
                                                                    start_joint=middle_point,
                                                                    elbow_pos_list=highest_elbow_pos_list,
                                                                    grasp_poses_list=grasp_poses_list,
                                                                    target_pointcloud=self.target_points_base)

                # Pro-process of motion planning, select the execution one and remove redundent waypoints
                if path_list is None:
                    print(f"no path")
                    times-=1
                    continue
                gripper_mat_list = np.array(self.actor.pos_orn2matrix(gripper_pos_list, gripper_orn_list))
                score_list = []

                for gripper_mat_path in  gripper_mat_list:
                    score_list.append(self.path_quality_decision(gripper_mat_path))
                sorted_indices = np.argsort(score_list)
                score_list.sort()
                score_list.sort(reverse=True)
                print(f"score_list: {score_list}")

                path_list = np.array(path_list)[sorted_indices]
                if len(path_list) == 0:
                    print(f"no path")
                    times-=1
                    continue
                
                first_waypoint_idx = 5
                exe_gripper_pos_path = np.asarray(gripper_pos_list[0])
                for idx, pos in enumerate(exe_gripper_pos_path):
                    if pos[0] > 0.3:
                        first_waypoint_idx = idx
                        break
                exe_gripper_pos_path = exe_gripper_pos_path[first_waypoint_idx:]

                # Execution part
                exe_path_list = np.asarray(path_list[0])[first_waypoint_idx:]
                print(f"before adjust, exe_path_list: {len(exe_path_list)}")
                exe_path_list = self.adjust_waypoint(exe_gripper_pos_path, exe_path_list)
                print(f"after adjust, exe_path_list: {len(exe_path_list)}")
                print(f"exe_path_list: {exe_path_list}")
                exe_path_list = np.concatenate((np.linspace(self.home_joint_point, exe_path_list[0], num=5), exe_path_list), axis=0)
                reverse_path_list = np.flip(exe_path_list, axis=0)[:-2]
                # Moving along the path in joint space
                self.move_along_path_vel(exe_path_list)
                time.sleep(3)
                print(f"finished the path")

                # Moving forward in cartesian space
                ef_pose = self.get_ef_pose()
                forward_mat = np.eye(4)
                forward_mat[2, 3] = 0.05
                ef_pose = ef_pose.dot(forward_mat)
                quat_pose = pack_pose(ef_pose)
                RT_grasp = [quat_pose[:3], ros_quat(quat_pose[3:])]
            
                self.set_pose(RT_grasp[0], RT_grasp[1])

                time.sleep(1)
                # Close gripper
                self.control_gripper("set_pose", 0.)
                time.sleep(1.5)

                # Moving back to placing config            
                self.move_along_path_vel(reverse_path_list[:-3])
                place_path = np.linspace(reverse_path_list[-3], place_joint_point_list[times], num=8)
                self.move_along_path(place_path)
                time.sleep(3)

                # Open gripper
                self.control_gripper("set_pose", 0.085)
                time.sleep(0.5)

                retreat_path = np.linspace(place_joint_point_list[times][:6], self.home_joint_point[:6], num=8)
                print(f"retreat_path: {retreat_path}")
                
                self.move_along_path_vel(retreat_path)
                print(f"finish grasping")
                time.sleep(2)

        elif msg.data == 5:
            # Main operation
            # Pybullet setup
            self.actor.init_joint_pose = self.joint_states
            self.actor.env._panda.reset(self.actor.init_joint_pose)
            if self.actor.sim_furniture_id is not None:
                self.actor.remove_sim_fureniture()
                self.actor.sim_furniture_id = None
            self.actor.replace_real_furniture()

            # Set middle waypoint
            middle_point_l = [0.6044797768523323, -0.4638172672916634, 1.9192456554014092, -0.7220135978730173, 1.4285132171318873, 0.16713466369119284]
            middle_point_r = [-0.44463801335875336, -0.34688159225181014, 1.955122030552248, -0.8672056118156931, 1.5888701358488284, -0.077308387296449]
            middle_point_m = [0.0757963692651781, -0.6608915874316905, 1.9553211018437315, -0.5384243936496882, 1.5888411073929465, -0.07726960501307473]

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

            # # Segmentation part
            seg_msg = Int32()
            seg_msg.data = 2
            self.seg_pub.publish(seg_msg)
            while(self.target_points is None):
                time.sleep(0.05) # Sleep to wait for the segmentation pointcloud arrive
            print(f"finished segmentation")
            # self.visual_pc(self.target_points)

            
            # Seperate the pointcloud of objects and assign the first for grasping task
            object_pc_list = self.seperate_object_pointcloud(self.target_points)
            # for object_pc in object_pc_list:
            #     self.visual_pc(object_pc)
            print(f"object_pc_list: {len(object_pc_list)}")
            # Choose one as target object
            cur_target_idx = self.find_optimal_point_cloud_index([self.pc_cam2base(object_pc) for object_pc in object_pc_list])
            self.target_points = object_pc_list.pop(cur_target_idx)

            # Add other objects' points to obstacle pointcloud
            for object_pc in object_pc_list:
                down_pc = regularize_pc_point_count(object_pc, 256)
                
                self.obs_points = np.concatenate((self.obs_points, down_pc), axis=0)
            
            # # convert frame from camera to world(base)
            self.obs_points_base = self.pc_cam2base(self.obs_points)
            self.target_points_base = self.pc_cam2base(self.target_points)
            _, self.target_points_base = self.collect_plane_points(self.target_points_base)
            # object_pc_list = [self.pc_cam2base(i) for i in object_pc_list]
            # self.visual_pc(self.target_points_base)

            # self.add_plane_2_obs_pc()
            self.add_layer_plane_2_obs_pc()
            # self.visual_pc(self.obs_points_base)

            self.actor.sim_furniture_id = self.actor.create_obstacle_from_pc(self.obs_points_base, self.target_points_base)
            

            # Get grasp poses and filter out those facing negative x-axis
            grasp_poses_camera = self.setting_contact_req(obstacle_points=self.obs_points, target_points=self.target_points)
            self.grasp_list = []
            self.score_list = []
            for grasp_pose_cam in grasp_poses_camera:
                grasp_camera = np.array(grasp_pose_cam.pred_grasps_cam)
                grasp_world = self.pose_cam2base(grasp_camera.reshape(4,4))
                if grasp_world[0, 2] >= -0.3:
                    self.grasp_list.append(grasp_world)
                    self.score_list.append(grasp_pose_cam.score)

            print(f"self.grasp_list: {self.grasp_list}")
            print(f"self.score_list: {self.score_list}")
            self.actor.visualize_points_grasppose(self.obs_points_base, self.grasp_list)

            # Retreat the grasp poses a little and convert them to joint space
            self.grasp_list = self.grasp2pre_grasp(self.grasp_list, drawback_dis=0.1) # Drawback a little
            (grasp_joint_list, grasp_poses_list,
            elbow_pos_list, grasp_score_list) = self.actor.grasp_pose2grasp_joint(grasp_poses=self.grasp_list,
                                                                                grasp_scores=self.score_list)
            
            grasp_joint_list = np.array(grasp_joint_list)
            elbow_pos_list = np.array(elbow_pos_list)
            grasp_poses_list = np.array(grasp_poses_list)

            if np.mean(self.target_points_base, axis=0)[1] > 0.1:
                middle_point = middle_point_l
            elif np.mean(self.target_points_base, axis=0)[1] < -0.1:
                middle_point = middle_point_r
            else:
                middle_point = middle_point_m

            # Motion planning part
            if len(elbow_pos_list) == 0:
                print(f"There is no path")
                path_list = grasp_poses_list = elbow_path_list = gripper_pos_list = gripper_orn_list = None
            elif len(elbow_pos_list) == 1:
                grasp_joint_list = self.adjust_joint_values(grasp_joint_list)
                (path_list,
                 elbow_path_list,
                 gripper_pos_list,
                 gripper_orn_list) = self.actor.motion_planning(grasp_joint_cfg=grasp_joint_list,
                                                                start_joint=middle_point,
                                                                elbow_pos_list=elbow_pos_list,
                                                                grasp_poses_list=grasp_poses_list,
                                                                target_pointcloud=self.target_points_base)
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
                                                                start_joint=middle_point,
                                                                elbow_pos_list=highest_elbow_pos_list,
                                                                grasp_poses_list=grasp_poses_list,
                                                                target_pointcloud=self.target_points_base)

            # Pro-process of motion planning, select the execution one and remove redundent waypoints
            if path_list is None:
                print(f"no path")
                return
            gripper_mat_list = np.array(self.actor.pos_orn2matrix(gripper_pos_list, gripper_orn_list))
            score_list = []

            for gripper_mat_path in  gripper_mat_list:
                score_list.append(self.path_quality_decision(gripper_mat_path))
            sorted_indices = np.argsort(score_list)
            score_list.sort()
            score_list.sort(reverse=True)
            print(f"score_list: {score_list}")

            path_list = np.array(path_list)[sorted_indices]
            if len(path_list) == 0:
                print(f"no path")
                return
            
            first_waypoint_idx = 5
            exe_gripper_pos_path = np.asarray(gripper_pos_list[0])
            for idx, pos in enumerate(exe_gripper_pos_path):
                if pos[0] > 0.3:
                    first_waypoint_idx = idx
                    break
            exe_gripper_pos_path = exe_gripper_pos_path[first_waypoint_idx:]

            # Execution part
            exe_path_list = np.asarray(path_list[0])[first_waypoint_idx:]
            print(f"before adjust, exe_path_list: {len(exe_path_list)}")
            exe_path_list = self.adjust_waypoint(exe_gripper_pos_path, exe_path_list)
            print(f"after adjust, exe_path_list: {len(exe_path_list)}")
            print(f"exe_path_list: {exe_path_list}")
            exe_path_list = np.concatenate((np.linspace(self.home_joint_point, exe_path_list[0], num=5), exe_path_list), axis=0)
            reverse_path_list = np.flip(exe_path_list, axis=0)[:-2]
            # Moving along the path in joint space
            self.move_along_path_vel(exe_path_list)
            time.sleep(3)
            print(f"finished the path")

            # Moving forward in cartesian space
            ef_pose = self.get_ef_pose()
            forward_mat = np.eye(4)
            forward_mat[2, 3] = 0.05
            ef_pose = ef_pose.dot(forward_mat)
            quat_pose = pack_pose(ef_pose)
            RT_grasp = [quat_pose[:3], ros_quat(quat_pose[3:])]
        
            self.set_pose(RT_grasp[0], RT_grasp[1])

            time.sleep(1)
            # Close gripper
            self.control_gripper("set_pose", 0.)
            time.sleep(1.5)

            # Moving back to placing config            
            self.move_along_path_vel(reverse_path_list[:-3])
            place_path = np.linspace(reverse_path_list[-3], self.place_joint_point, num=8)
            self.move_along_path(place_path)
            time.sleep(3)

            # Open gripper
            self.control_gripper("set_pose", 0.085)
            time.sleep(0.5)

            retreat_path = np.linspace(self.place_joint_point[:6], self.home_joint_point[:6], num=8)
            print(f"retreat_path: {retreat_path}")
            
            self.move_along_path_vel(retreat_path)
            print(f"finish grasping")
        elif msg.data == 6:
            # Main operation without ruckig
            # Pybullet setup
            self.actor.init_joint_pose = self.joint_states
            self.actor.env._panda.reset(self.actor.init_joint_pose)
            if self.actor.sim_furniture_id is not None:
                self.actor.remove_sim_fureniture()
                self.actor.sim_furniture_id = None
            self.actor.replace_real_furniture()

            # Set middle waypoint
            middle_point_l = [0.6044797768523323, -0.4638172672916634, 1.9192456554014092, -0.7220135978730173, 1.4285132171318873, 0.16713466369119284]
            middle_point_r = [-0.44463801335875336, -0.34688159225181014, 1.955122030552248, -0.8672056118156931, 1.5888701358488284, -0.077308387296449]
            middle_point_m = [0.0757963692651781, -0.6608915874316905, 1.9553211018437315, -0.5384243936496882, 1.5888411073929465, -0.07726960501307473]

            # Reset the gripper
            self.control_gripper("reset")
            time.sleep(1)
            self.control_gripper("set_pose", 0.)
            time.sleep(1)
            self.control_gripper("set_pose", 0.085)
            print(f"finish grasping")


            # Reset the arm's position
            self.move_along_path_dir([self.home_joint_point])
            time.sleep(1)

            # Set init_value to None
            self.target_points = None
            self.obs_points = None

            # # Segmentation part
            seg_msg = Int32()
            seg_msg.data = 2
            self.seg_pub.publish(seg_msg)
            while(self.target_points is None):
                time.sleep(0.05) # Sleep to wait for the segmentation pointcloud arrive
            print(f"finished segmentation")
            # self.visual_pc(self.target_points)

            
            # Seperate the pointcloud of objects and assign the first for grasping task
            object_pc_list = self.seperate_object_pointcloud(self.target_points)
            # for object_pc in object_pc_list:
            #     self.visual_pc(object_pc)
            print(f"object_pc_list: {len(object_pc_list)}")
            # Choose one as target object
            cur_target_idx = self.find_optimal_point_cloud_index([self.pc_cam2base(object_pc) for object_pc in object_pc_list])
            self.target_points = object_pc_list.pop(cur_target_idx)

            # Add other objects' points to obstacle pointcloud
            for object_pc in object_pc_list:
                down_pc = regularize_pc_point_count(object_pc, 256)
                
                self.obs_points = np.concatenate((self.obs_points, down_pc), axis=0)
            
            # # convert frame from camera to world(base)
            self.obs_points_base = self.pc_cam2base(self.obs_points)
            self.target_points_base = self.pc_cam2base(self.target_points)
            _, self.target_points_base = self.collect_plane_points(self.target_points_base)
            # object_pc_list = [self.pc_cam2base(i) for i in object_pc_list]
            # self.visual_pc(self.target_points_base)

            self.add_plane_2_obs_pc()
            self.add_layer_plane_2_obs_pc()
            # self.visual_pc(self.obs_points_base)

            self.actor.sim_furniture_id = self.actor.create_obstacle_from_pc(self.obs_points_base, self.target_points_base)
            

            # Get grasp poses and filter out those facing negative x-axis
            grasp_poses_camera = self.setting_contact_req(obstacle_points=self.obs_points, target_points=self.target_points)
            self.grasp_list = []
            self.score_list = []
            for grasp_pose_cam in grasp_poses_camera:
                grasp_camera = np.array(grasp_pose_cam.pred_grasps_cam)
                grasp_world = self.pose_cam2base(grasp_camera.reshape(4,4))
                if grasp_world[0, 2] >= -0.3:
                    self.grasp_list.append(grasp_world)
                    self.score_list.append(grasp_pose_cam.score)

            print(f"self.grasp_list: {self.grasp_list}")
            print(f"self.score_list: {self.score_list}")
            self.actor.visualize_points_grasppose(self.obs_points_base, self.grasp_list)

            # Retreat the grasp poses a little and convert them to joint space
            self.grasp_list = self.grasp2pre_grasp(self.grasp_list, drawback_dis=0.1) # Drawback a little
            (grasp_joint_list, grasp_poses_list,
            elbow_pos_list, grasp_score_list) = self.actor.grasp_pose2grasp_joint(grasp_poses=self.grasp_list,
                                                                                grasp_scores=self.score_list)
            
            grasp_joint_list = np.array(grasp_joint_list)
            elbow_pos_list = np.array(elbow_pos_list)
            grasp_poses_list = np.array(grasp_poses_list)

            if np.mean(self.target_points_base, axis=0)[1] > 0.1:
                middle_point = middle_point_l
            elif np.mean(self.target_points_base, axis=0)[1] < -0.1:
                middle_point = middle_point_r
            else:
                middle_point = middle_point_m

            # Motion planning part
            if len(elbow_pos_list) == 0:
                print(f"There is no path")
                path_list = grasp_poses_list = elbow_path_list = gripper_pos_list = gripper_orn_list = None
            elif len(elbow_pos_list) == 1:
                grasp_joint_list = self.adjust_joint_values(grasp_joint_list)
                (path_list,
                 elbow_path_list,
                 gripper_pos_list,
                 gripper_orn_list) = self.actor.motion_planning(grasp_joint_cfg=grasp_joint_list,
                                                                start_joint=middle_point,
                                                                elbow_pos_list=elbow_pos_list,
                                                                grasp_poses_list=grasp_poses_list,
                                                                target_pointcloud=self.target_points_base)
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
                                                                start_joint=middle_point,
                                                                elbow_pos_list=highest_elbow_pos_list,
                                                                grasp_poses_list=grasp_poses_list,
                                                                target_pointcloud=self.target_points_base)

            # Pro-process of motion planning, select the execution one and remove redundent waypoints
            if path_list is None:
                print(f"no path")
                return
            gripper_mat_list = np.array(self.actor.pos_orn2matrix(gripper_pos_list, gripper_orn_list))
            score_list = []

            for gripper_mat_path in  gripper_mat_list:
                score_list.append(self.path_quality_decision(gripper_mat_path))
            sorted_indices = np.argsort(score_list)
            score_list.sort()
            score_list.sort(reverse=True)
            print(f"score_list: {score_list}")

            path_list = np.array(path_list)[sorted_indices]
            if len(path_list) == 0:
                print(f"no path")
                return
            
            first_waypoint_idx = 5
            exe_gripper_pos_path = np.asarray(gripper_pos_list[0])
            for idx, pos in enumerate(exe_gripper_pos_path):
                if pos[0] > 0.3:
                    first_waypoint_idx = idx
                    break
            exe_gripper_pos_path = exe_gripper_pos_path[first_waypoint_idx:]

            # Execution part
            exe_path_list = np.asarray(path_list[0])[first_waypoint_idx:]
            print(f"before adjust, exe_path_list: {len(exe_path_list)}")
            exe_path_list = self.adjust_waypoint(exe_gripper_pos_path, exe_path_list)
            print(f"after adjust, exe_path_list: {len(exe_path_list)}")
            print(f"exe_path_list: {exe_path_list}")
            exe_path_list = np.concatenate((np.linspace(self.home_joint_point, exe_path_list[0], num=3), exe_path_list), axis=0)
            reverse_path_list = np.flip(exe_path_list, axis=0)[:-2]
            # Moving along the path in joint space
            self.move_along_path_dir(exe_path_list)


            # Moving forward in cartesian space
            ef_pose = self.get_ef_pose()
            forward_mat = np.eye(4)
            forward_mat[2, 3] = 0.3
            ef_pose = ef_pose.dot(forward_mat)
            quat_pose = pack_pose(ef_pose)
            forward_pose = np.array([*quat_pose[:3], *quat2euler(quat_pose[3:])]) # This pose is a 1d array(x, y, z and euler)
            self.dir_set_position(forward_pose, mode="cart")
            print(f"Start grasping")
            # Close gripper
            self.control_gripper("set_pose", 0.)
            time.sleep(1)
            print(f"Finished grasping")

            # Moving back to placing config            
            self.move_along_path_dir(reverse_path_list)
            place_path = np.linspace(reverse_path_list[-1], self.place_joint_point, num=3)
            self.move_along_path_dir(place_path)

            time.sleep(0.5)
            # Open gripper
            self.control_gripper("set_pose", 0.085)
            time.sleep(0.5)

            retreat_path = np.linspace(self.place_joint_point[:6], self.home_joint_point[:6], num=3)
            print(f"retreat_path: {retreat_path}")
            
            self.move_along_path_dir(retreat_path)
            print(f"finish grasping")
        elif msg.data == 7:
            # Reset the gripper
            self.control_gripper("reset")
            time.sleep(5)
            self.control_gripper("set_pose", 0.)
            time.sleep(10)
            self.control_gripper("set_pose", 0.085)
            print(f"finish grasping")

    def raw_points_callback(self, msg):
        if(self.raw_point_flag):
            raw_scene_points_base = self.pc_cam2base(self.pc2_tranfer(msg))
            o3d_pc = o3d.geometry.PointCloud()
            o3d_pc.points = o3d.utility.Vector3dVector(raw_scene_points_base)
            min_bound = np.array([0.7, -0.5, 0.3])
            max_bound = np.array([0.93, 0.5, 0.63])

            # 创建立方体裁剪框
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)

            print(f"{o3d_pc}")
            o3d_pc = o3d_pc.crop(bbox)
            print(f"{o3d_pc}")
            self.raw_point_flag = False
            self.target_points_base = regularize_pc_point_count(np.asarray(o3d_pc.points), 2048)
            print(f"self.target_points_base: {type(self.target_points_base)}")
            


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
        self.bounds = [[-0.05, 1.05], [-0.5, 0.5], [-0.12, 2]]  # set the bounds
        bounding_box_points = list(itertools.product(*self.bounds))  # create limit points
        self.bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(bounding_box_points))  # create bounding box object
        if crop:
            o3d_pc = o3d_pc.crop(self.bounding_box)
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

    def add_layer_plane_2_obs_pc(self):
        x_min, x_max = 0.63, 1.0
        y_min, y_max = -0.5, 0.5
        z = 0.655

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



        x_min, x_max = 0.61, 1.0
        y_min, y_max = -0.5, 0.5
        z = 0.23

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
    

    def dir_set_position(self, position, mode="joint"):
        """
        Send goal joint value to tm_driver directly
        """
        srv = SetPositionsRequest()
        
        if mode == "joint":
            srv.motion_type = 1
            self.goal = np.concatenate((position, [0, 0, 0]))
        else:
            srv.motion_type = 2
            self.goal = position
        srv.positions = position
        srv.velocity = 5.0
        srv.acc_time = 1.4
        srv.blend_percentage = 10
        srv.fine_goal = False
        self.position_serivce_client(srv)
        
        return self.dir_set_loop_confirm()

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
        print(f"Move tm joints to velocity: {target_joint.velocity}\n")
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
        last_state = None
        last_time = None
        if mode == "joint":
            threshold=0.015
            while True:
                dis = np.linalg.norm(self.joint_states-self.joint_goal)
                # print(f"dis: {dis}")
                if last_state is None or np.linalg.norm(self.joint_states - last_state) > 0.001:
                    last_time = time.time()

                if dis < threshold:
                    break
                if time.time() - last_time > 0.1:
                    break
                last_state = self.joint_states
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
                if last_state is None or np.linalg.norm(ef_pos - last_state) > 0.01:
                    last_time = time.time()
                # print(f"dis: {dis}")
                if dis < threshold:
                    break
                if time.time() - last_time > 0.1:
                    break
                last_state = ef_pos
            return True

    def dir_set_loop_confirm(self):
        # This loop_confirm only comfirm the joint part, cartesian will just sleep for a while
        threshold=0.01
        if len(self.goal) == 9:
            while True:
                dis = np.linalg.norm(self.joint_states-self.goal)
                if dis < threshold:
                    break
        else:
            time.sleep(2)
        return True

    def move_along_path_dir(self, path):
        for waypoint in path:
            self.dir_set_position(waypoint)
            self.goal = None
            print("finished one waypoint")

    def move_along_path(self, path):
        for waypoint in path:
            self.set_joint(waypoint)
            self.loop_confirm()

    def move_along_path_vel(self, path):
        # First calculate the velocity
        joint_velocities = []
        max_vel = 0.08
        # Loop through waypoints to compute velocities
        for i in range(len(path) - 1):
            # Difference between consecutive waypoints
            delta_q = path[i+1] - path[i]
            max_value = np.max(np.abs(delta_q))
            # Compute the time interval needed for the max velocity constraint
            velocities = delta_q * (max_vel / max_value)
            # print(f"velocities: {velocities}")
            joint_velocities.append(velocities)
        joint_velocities.append([0, 0, 0, 0, 0, 0])
        for waypoint, velocity in zip(path, joint_velocities):
            # self.set_joint_vel(waypoint, velocity)
            self.set_joint_vel(waypoint, [0, 0, 0, 0, 0, 0])
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
    
    def seperate_object_pointcloud(self, pc):
        eps = 0.03  # Maximum distance between two samples for them to be considered as in the same neighborhood
        min_samples = 20  # Minimum number of samples in a neighborhood for a point to be considered as a core point

        # Apply DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(pc)
        labels = db.labels_
        clusters = [pc[labels == label] for label in set(labels) if label != -1]
        return clusters
    
    def collect_plane_points(self, point_cloud):
        # 使用布林遮罩，過濾出 z < 0.27 的點
        mask = point_cloud[:, 2] < 0.28
        return point_cloud[mask], point_cloud[~mask]
        
        
    
    def adjust_waypoint(self, gripper_pos_path, joint_path):
        filtered_pos_waypoints = [gripper_pos_path[0]]  # 保留第一个点
        filtered_joint_waypoints = [joint_path[0]]
        for i in range(1, len(gripper_pos_path)-1):
            if np.linalg.norm(gripper_pos_path[i] - filtered_pos_waypoints[-1]) >= 0.05:
                filtered_pos_waypoints.append(gripper_pos_path[i])
                filtered_joint_waypoints.append(joint_path[i])
        filtered_pos_waypoints.append(gripper_pos_path[-1])
        filtered_joint_waypoints.append(joint_path[-1])

        print(f"filtered_joint_waypoints: {len(filtered_joint_waypoints)}")

        adjusted_joint_path = [filtered_joint_waypoints[0]]
        for idx in range(1, len(filtered_joint_waypoints)):
            if np.linalg.norm(filtered_pos_waypoints[idx] - filtered_pos_waypoints[idx-1]) > 0.07:
                adjusted_joint_path.append((filtered_joint_waypoints[idx] + filtered_joint_waypoints[idx-1])/2)
            adjusted_joint_path.append(filtered_joint_waypoints[idx])
        return np.array(adjusted_joint_path)


    def find_optimal_point_cloud_index(self, point_clouds):
        best_index = -1
        min_y = 1
        
        for i, cloud in enumerate(point_clouds):
            # 計算每個點雲的質心
            x, y, z = np.mean(cloud, axis=0)
            
            if z < 0.27:
                continue
            # 比較 x 值最小
            if x < 0.73:
                if y < min_y:
                    min_y =  y
                    best_index = i  # 儲存該點雲的索引
                

                
        return best_index

if __name__ == "__main__":
    rospy.init_node("test_realworld")
    real_actor_node = ros_node(renders=False)
    rospy.spin()