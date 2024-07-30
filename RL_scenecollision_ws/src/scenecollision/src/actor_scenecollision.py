import pybullet as p
import numpy as np
import time
import json
import open3d as o3d
import sys
import ray
import os
parent_dir = "/home/user/RL_TM5_900_pybullet"
sys.path.append(parent_dir)
from utils.utils import *
from utils.grasp_checker import ValidGraspChecker
from env.ycb_scene import SimulatedYCBEnv
from pointmlp import farthest_point_sample, index_points
from Helper3D.trimesh_URDF import getURDF
from pybullet_ompl import pb_ompl, pb_ompl_utils
from itertools import product
import copy
import alphashape
from sklearn.cluster import DBSCAN
from scipy.spatial.transform import Rotation as R
from utils.planner_matt import GraspPlanner
class ActorWrapper(object):
    """
    wrapper testing, use ray to create multiple pybullet
    """
    def __init__(self, renders=False, simulation_id=None):
        # from env.ycb_scene import SimulatedYCBEnv
        # file = os.path.join("object_index", 'acronym_90.json')
        # file = os.path.join(parent_dir, "object_index", 'proper_objects.json')
        file = os.path.join(parent_dir, "object_index", 'small_objects.json')
        with open(file) as f: file_dir = json.load(f)
        file_dir = file_dir['train']
        # file_dir = file_dir['test']
        file_dir = [f[:-5] for f in file_dir]
        test_file_dir = list(set(file_dir))
        test_file_dir = random.sample(test_file_dir, 15)
        # self.furniture_name = "carton_box"
        # self.furniture_name = "table"
        # self.furniture_name = "shelf"
        self.furniture_name = "shelf_2"
        # self.furniture_name = "shelf_3"
        # self.furniture_name = "shelf_4"
        # self.furniture_name = "shelf_5"
        self.env = SimulatedYCBEnv(renders=renders)
        self.env._load_index_objs(test_file_dir)
        self.env.reset(save=False, enforce_face_target=True, furniture=self.furniture_name)
        self.grasp_checker = ValidGraspChecker(self.env)
        self.target_points = None   # This is for merging point-cloud from different time
        self.obstacle_points = None
        self.simulation_id = simulation_id
        self.sim_furniture_id = None
        self.joint_bounds = list(zip(self.env._panda._joint_min_limit, self.env._panda._joint_max_limit))[:6]
        self.joint_bounds[0] = (-1.57, 1.57)
        # disable the collision between the basse of TM5 and plane        
        p.setCollisionFilterPair(self.env.plane_id, self.env._panda.pandaUid, -1, 0, enableCollision=False)
        

        # Dictionary of targets' pos, orn and constraint
        self.targets_dict = {}

        # This part is for pybullet_ompl
        self.joint_idx = [1, 2, 3, 4, 5, 6] # The joint idx that is considerated in pynullet_ompl
        self.obstacles = [self.env.plane_id, self.env.furniture_id] # Set up the obstacles
        if self.furniture_name == "shelf":
            self.init_joint_pose = [-0.15, -1.55, 1.8, -0.1, 1.8, 0.0, 0.0, 0.0, 0.0]
        elif self.furniture_name == "shelf_2":
            # self.init_joint_pose = [-0.16361913, -1.5037375, 1.80286642, -0.14974172, 1.81350627, -0.00209805, 0.0, 0.0, 0.0]
            self.init_joint_pose = [-0.163, -1.755, 2.506, -0.508, 1.832, -0.016, 0.0, 0.0, 0.0]
        elif self.furniture_name == "shelf_3":
            self.init_joint_pose = [-0.15, -1.55, 1.8, -0.1, 1.8, 0.0, 0.0, 0.0, 0.0]
        elif self.furniture_name == "shelf_4":
            self.init_joint_pose = [-0.15, -1.55, 1.8, -0.1, 1.8, 0.0, 0.0, 0.0, 0.0]
        elif self.furniture_name == "shelf_5":
            self.init_joint_pose = [-0.163, -1.755, 2.506, -0.508, 1.832, -0.016, 0.0, 0.0, 0.0]
        elif self.furniture_name == "table":
            self.init_joint_pose = [-0., -0.95, 1.9, -0.1, 1.571, 0.0, 0.0, 0.0, 0.0]
        elif self.furniture_name == "carton_box":
            self.init_joint_pose = [0.03, -1., 1.9, -0.1, 1.571, 0.0, 0.0, 0.0, 0.0]


    def rollout_once(self, vis=False):
        start = time.time()
        self.env.reset(save=False, enforce_face_target=False, init_joints=self.init_joint_pose, reset_free=True)
        rewards = self.expert_move(vis=vis)
        duration = time.time() - start
        print(f"actor duration: {duration}")
        return rewards

    def get_grasp_pose(self):
        '''
        Take pre-define grasp dataset of target object as an example.
        Load npy file by object names.
        '''

        scale_str_num = len(f"_{self.env.object_scale[self.env.target_idx]}") * (-1)
        obj_name = self.env.obj_path[self.env.target_idx].split('/')[-2][:scale_str_num]
        data_dir = parent_dir + "/data/grasps/acronym"
        tr = np.load(f'{data_dir}/{obj_name}.npy',
                     allow_pickle=True,
                     fix_imports=True,
                     encoding="bytes")
        grasp = tr.item()[b'transforms']

        # Transforms grasp pose to current position

        obj_pos = p.getBasePositionAndOrientation(self.env._objectUids[self.env.target_idx])
        obj_pos_mat = unpack_pose([*obj_pos[0], *tf_quat(obj_pos[1])])
        grasp_candidate = obj_pos_mat.dot(grasp.T)
        grasp_candidate = np.transpose(grasp_candidate, axes=[2, 0, 1])

        '''
        The extract_grasp() function takes grasp group[N, 4, 4] as input and outputs valid grasps.
        The parameter "drawback_distance" is the distance to draw back the end effector pose along z-axis in validation process.
        The parameter "filter_elbow" denote if checker use estimated elbow point and bounding box of table
            as one of the measurements to prevent collision of other joint.
        Note: The estimated elbow point is NOT calculate by IK, so it's nearly a rough guess.
        '''

        grasp_arrays, grasp_index = self.grasp_checker.extract_grasp(grasp_candidate,
                                                                     drawback_distance=0.03,
                                                                     visual=False,
                                                                     filter_elbow=True)
        # print(f"Valid index in grasp group:\n    {grasp_index}\
        #       \nGrasp Matrix:\n{grasp_arrays[0]}")

        if len(grasp_arrays) == 0:
            return None

        
        # # get the nearest grasp pose
        # cur_ef_pose = self.env._get_ef_pose(mat=True)
        # cur_xyz = cur_ef_pose[:, 3:4].reshape(4, )[:3]
        # min_dist = 100
        # final_pose = None
        # for candidate_pose in grasp_arrays:
        #     can_xyz = candidate_pose[:, 3:4].reshape(4, )[:3]
        #     xyz_dis = np.linalg.norm(cur_xyz - can_xyz)
        #     if min_dist > xyz_dis:
        #         min_dist = xyz_dis
        #         final_pose = candidate_pose
        
        # grasp_arrays = [final_pose]


        # get the nearest grasp pose
        cur_ef_pose = self.env._get_ef_pose(mat=True)
        cur_xyz = cur_ef_pose[:, 3:4].reshape(4, )[:3]
        final_poses = []

        for candidate_pose in grasp_arrays:
            can_xyz = candidate_pose[:, 3:4].reshape(4, )[:3]
            
            # point_unit_vector is a unit vector from gripper to grasp pose 
            point_unit_vector = (can_xyz - cur_xyz)/np.linalg.norm(cur_xyz - can_xyz)
            # Extract the orientation component of the matrix (assuming it's a rotation matrix)
            orientation_vector = candidate_pose[:3, 2:3].reshape(3, )
            

            # Calculate the dot product of the grasp psoe and the unit vector from gripper to grasp pose
            face_foward2arm = np.dot(point_unit_vector, orientation_vector)


            # Check if the pose is not too horizontal or too vertical
            if face_foward2arm > 0.2:
                final_poses.append(candidate_pose)


        if len(final_poses) == 0:
            return
        
        grasp_arrays = final_poses
        grasp_joint_cfg = []
        for grasp_array in grasp_arrays:
            pos_orn = pack_pose(grasp_array)
            grasp_joint_cfg.append(p.calculateInverseKinematics(self.env._panda.pandaUid,
                                        self.env._panda.pandaEndEffectorIndex,
                                        pos_orn[:3],
                                        ros_quat(pos_orn[3:]),
                                        maxNumIterations=500,
                                        residualThreshold=1e-8))

        grasp_joint_cfg = np.array(grasp_joint_cfg)

        return grasp_joint_cfg
        
    def expert_move(self, vis=False):
        self.target_points = None
        self.obstacle_points = None
        pos, ori = p.getBasePositionAndOrientation(self.env._objectUids[self.env.target_idx])
        fixed_joint_constraint = p.createConstraint(
            parentBodyUniqueId=self.env._objectUids[self.env.target_idx],
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=pos,
            childFrameOrientation=ori
            )
        
        grasp_joint_pose = self.get_grasp_pose()
        if grasp_joint_pose is None:
            p.removeConstraint(fixed_joint_constraint)
            self.env.place_back_objects()
            return (0, 0)
        
        obstacle_points, target_points, scene_points = self.get_pc_state(frame="base", vis=vis)
        

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(np.array(obstacle_points[:, :3]))

        # # Create coordinate axes
        # axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        # # Visualize the point cloud with axes
        # o3d.visualization.draw_geometries([pcd, axes])

        start_time = time.time()
        joint_path = ray.get(self.simulation_id.create_simulation_env.remote(obstacle_points[:, :3],
                                                                target_points[:, :3],
                                                                grasp_joint_pose))

        if joint_path is None:
            p.removeConstraint(fixed_joint_constraint)
            self.env.place_back_objects()
            return


        # This part is for the path after grasping
        # Use copy.deepcopy because the list is 2dlist, use deepcopy to copy the whole list(including 
        # the 1d list in it), otherwise the second dimension(which is 1) still point to the same memory
        retreat_joint_path = copy.deepcopy(joint_path)
        retreat_joint_path.reverse()


        for joint_con in joint_path:
            extend_joint_con = joint_con
            extend_joint_con.extend([0, 0, 0])
            p.setJointMotorControlArray(bodyUniqueId=self.env._panda.pandaUid,
                                        # jointIndices=self.joint_idx,
                                        jointIndices=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=extend_joint_con,
                                        forces=[250, 250, 250, 250, 250, 250, 100, 100, 100],
                                        positionGains=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                                        velocityGains=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            for _ in range(200):
                p.stepSimulation()

        # Slowly move to grasp pose
        for _ in range(3):
            self.env.step([0, 0, 0.01, 0, 0, 0])
        
        # Start to grasp
        p.removeConstraint(fixed_joint_constraint)
        self.env.grasp()

        for joint_con in retreat_joint_path:
            extend_joint_con = joint_con
            extend_joint_con.extend([0, 0, 0.8])
            p.setJointMotorControlArray(bodyUniqueId=self.env._panda.pandaUid,
                                        # jointIndices=self.joint_idx,
                                        jointIndices=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=extend_joint_con,
                                        forces=[250, 250, 250, 250, 250, 250, 100, 100, 100],
                                        positionGains=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                                        velocityGains=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            for _ in range(100):
                p.stepSimulation()
        
        self.env.place_back_objects()
        

    def get_gripper_points(self, target_pointcloud=None):
        # skeleton part, the links' center
        inner_point = list(p.getLinkState(self.env._panda.pandaUid, 8)[0])

        # 12 and 17 for finger, 19 for camera
        gripper_points = np.array([p.getLinkState(self.env._panda.pandaUid, 12)[0],
                                   p.getLinkState(self.env._panda.pandaUid, 17)[0],
                                   p.getLinkState(self.env._panda.pandaUid, 19)[0],
                                   inner_point])
        gripper_points = np.hstack((gripper_points, 2 * np.ones((gripper_points.shape[0], 1))))

        link_points = np.array([p.getLinkState(self.env._panda.pandaUid, i)[0] for i in range(7)])
        link_points = np.hstack((link_points, 2 * np.ones((link_points.shape[0], 1))))

        skeleton_points = np.concatenate((link_points, gripper_points), axis=0)

        # Surface part
        surface_points = self.get_surface_points()
        manipulator_points = np.hstack((surface_points, 2*np.ones((surface_points.shape[0], 1))))

        if target_pointcloud.any() is not None:
            final_pointcloud = np.concatenate((target_pointcloud, manipulator_points), axis=0)
        else:
            final_pointcloud = manipulator_points
        return final_pointcloud, manipulator_points, skeleton_points

    def get_world_pointcloud(self, raw_data=False, no_gripper=False):
        obs, joint_pos, camera_info, pose_info = self.env._get_observation(raw_data=raw_data, vis=False, no_gripper=no_gripper)
        pointcloud = obs[0]
        ef_pose = pose_info[1]
        
        # # transform the pointcloud from camera back to world frame
        pointcloud_tar = np.hstack((pointcloud.T, np.ones((len(pointcloud.T), 1)))).T
        cam_offset_inv = np.linalg.inv(self.env.cam_offset)
        points_world = (np.dot(ef_pose, cam_offset_inv.dot(pointcloud_tar)).T)[:, :3]

        if len(points_world) == 0:
            return None
        if raw_data is True or raw_data == "obstacle":
            points_world = regularize_pc_point_count(points_world, 2048)
        else:
            points_world = regularize_pc_point_count(points_world, 2048)
        return points_world

    def get_joint_degree(self):
        con_action = p.getJointStates(self.env._panda.pandaUid, [i for i in range(1, 7)])
        con_action = np.array([i[0] for i in con_action])
        return con_action

    def get_pc_state(self, vis=False, frame="camera", target_only=False, concat=True):
        """
        The output pointcloud should (N, 4)
        """
        if target_only:
            target_points = self.get_world_pointcloud(raw_data=False)
            # target_points = self.get_world_pointcloud(raw_data="multiple_objects")
            if target_points is not None:
                if concat:
                    self.target_points = self.concatenate_pc(self.target_points, target_points)
                    target_points = np.hstack((self.target_points, np.ones((self.target_points.shape[0], 1))))
                else:
                    self.target_points = target_points
                    target_points = np.hstack((target_points, np.ones((target_points.shape[0], 1))))
            else:
                target_points = self.target_points
            obstacle_points = self.obstacle_points
            scene_points = self.obstacle_points
            obstacle_points = np.hstack((self.obstacle_points, np.zeros((self.obstacle_points.shape[0], 1))))
            scene_points = np.hstack((scene_points, 2 * np.ones((scene_points.shape[0], 1))))
        else:
            obstacle_points = self.get_world_pointcloud(raw_data="obstacle")
            target_points = self.get_world_pointcloud(raw_data=False)
            # target_points = self.get_world_pointcloud(raw_data="multiple_objects")
            if target_points is None:
                return None, None, None
            self.obstacle_points = self.concatenate_pc(self.obstacle_points, obstacle_points)
            self.target_points = self.concatenate_pc(self.target_points, target_points)
            scene_points = self.concatenate_pc(self.target_points, self.obstacle_points)
            obstacle_points = np.hstack((self.obstacle_points, np.zeros((self.obstacle_points.shape[0], 1))))
            target_points = np.hstack((self.target_points, np.ones((self.target_points.shape[0], 1))))
            scene_points = np.hstack((scene_points, 2 * np.ones((scene_points.shape[0], 1))))
        
        # # transform back to camera frame
        if frame == "camera":
            obstacle_points = self.base2camera(obstacle_points)
            target_points = self.base2camera(target_points)
            scene_points = self.base2camera(scene_points)

        if vis:
            all_o3d_pc = o3d.geometry.PointCloud()
            all_o3d_pc.points = o3d.utility.Vector3dVector(target_points[:, :3])
            axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            # o3d.visualization.draw_geometries([all_o3d_pc]+[target_o3d_pc]+[axis_pcd])
            o3d.visualization.draw_geometries([all_o3d_pc]+[axis_pcd])
        
        return obstacle_points, target_points, scene_points

    def concatenate_pc(self, pc_old, pc_new):
        if pc_new is not None:
            if pc_old is not None:
                # concatenate two pointcloud
                pc_old_tensor = torch.from_numpy(pc_old)
                pc_new_tensor = torch.from_numpy(pc_new)
                combined_pc_tensor = torch.cat((pc_old_tensor, pc_new_tensor), dim=0).unsqueeze(0)
                index = farthest_point_sample(combined_pc_tensor, 2048)
                combined_pc = index_points(combined_pc_tensor, index).squeeze().detach().numpy()
            else:
                # sample the pc_new, make the number of the pointcloud to 2048
                pc_new_tensor = torch.from_numpy(pc_new).unsqueeze(0)
                index = farthest_point_sample(pc_new_tensor, 2048)
                combined_pc = index_points(pc_new_tensor, index).squeeze().detach().numpy()
        else:
            if pc_old is not None:
                combined_pc = pc_old
            else:
                return None
        
        return combined_pc

    def base2camera(self, pointcloud):
        inverse_camera_matrix = np.linalg.inv(self.env.cam_offset)
        inverse_ef_pose_matrix = np.linalg.inv(self.env._get_ef_pose('mat'))
        original_fourth_column = pointcloud[:, 3].copy()
        pointcloud[:, 3] = 1
        pointcloud_ef_pose = np.dot(inverse_ef_pose_matrix, pointcloud.T).T[:, :3]
        pointcloud_camera = np.dot(inverse_camera_matrix, np.hstack((pointcloud_ef_pose, np.ones((pointcloud_ef_pose.shape[0], 1)))).T).T
        pointcloud_camera[:, 3] = original_fourth_column
        return pointcloud_camera
    

    def setup_collision_detection(self, obstacles, self_collisions = True, allow_collision_links = []):
        self.check_link_pairs = pb_ompl_utils.get_self_link_pairs(self.env._panda.pandaUid, self.joint_idx) if self_collisions else []
        moving_links = frozenset(
            [item for item in pb_ompl_utils.get_moving_links(self.env._panda.pandaUid, self.joint_idx) if not item in allow_collision_links])
        moving_bodies = [(self.env._panda.pandaUid, moving_links)]
        self.check_body_pairs = list(product(moving_bodies, obstacles))    

    def calculate_pose_difference(self, T1, T2):
        # Extract rotation matrices and translation vectors from 4x4 matrices
        R1 = T1[:3, :3]
        T1 = T1[:3, 3]

        R2 = T2[:3, :3]
        T2 = T2[:3, 3]

        # Calculate pose difference using the previously defined function
        rotation_difference = np.linalg.norm(R1 - R2, 'fro')
        translation_difference = np.linalg.norm(T1 - T2)
        pose_difference = np.sqrt(rotation_difference**2 + translation_difference**2)
        return pose_difference
            

    def extend_obs_pc(self, obs_pc, target_pc, scale_factor=0.02):
        # This function "extend" the obstacle pointcloud outward a little,
        # away from middle point of target pointcloud.
        # By doing this, the mesh created afterward can be more reasonable.
        target_middle = np.mean(target_pc, axis=0)
        vector_outward = obs_pc - target_middle
        vector_length = np.linalg.norm(vector_outward, axis=1, keepdims=True)
        normalized_vector = vector_outward / vector_length

        # Scale and move obstacle points away from the middle point
        moved_obs_pc_out = obs_pc + scale_factor * normalized_vector

        # Scale and move obstacle points forward to the middle point
        moved_obs_pc_in = obs_pc - scale_factor * normalized_vector
        
        combined_pc = np.concatenate((moved_obs_pc_in, moved_obs_pc_out), axis=0)

        return combined_pc



    def create_obstacle_from_pc(self, obs_pc, target_pc):
        # This function use the 2 pointcloud to create a object in pybullet
        combined_obs_pc = self.extend_obs_pc(obs_pc=obs_pc, target_pc=target_pc)
        obs_alph = alphashape.alphashape(combined_obs_pc, 16)
        obs_vertices = obs_alph.vertices
        obs_faces = np.array(obs_alph.faces).flatten()

        obs_visualShapeId = p.createVisualShape(
                            shapeType=p.GEOM_MESH,
                            flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
                            vertices=obs_vertices,
                            indices=obs_faces,
                            meshScale=[1, 1, 1]
                        )

        obs_collisionShapeId = p.createCollisionShape(
                            shapeType=p.GEOM_MESH,
                            flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
                            vertices=obs_vertices,
                            indices=obs_faces,
                            meshScale=[1, 1, 1]
                        )
        obs_body_id = p.createMultiBody(
                        baseMass=1,
                        baseInertialFramePosition=[0, 0, 0],
                        baseCollisionShapeIndex=obs_collisionShapeId,
                        baseVisualShapeIndex=obs_visualShapeId,
                        basePosition=[0, 0, 0],
                        baseOrientation=[0, 0, 0, 1]
                    )
        return obs_body_id

    def motion_planning(self, grasp_joint_cfg, start_joint=None, elbow_pos_list=None,
                        grasp_poses_list=None, cart=False, waypoint_num=40, target_pointcloud=None):
        if cart:
            return self.cartesian_motion_planning(grasp_poses_list)
        # Record web Bitstar
        file = open("/home/user/RL_TM5_900_pybullet/RL_scenecollision_ws/src/scenecollision/src/path_record.txt", 'w')
        timestamp = time.time()  # Get the current timestamp
        file.write(f"Timestamp: {timestamp}\n")

        path_list = []
        elbow_path_list = []
        
        gripper_pos_list = []
        gripper_orn_list = []
        init_elbow_pos, _ = p.getLinkState(self.env._panda.pandaUid, 5)[4:6]
        init_elbow_pos = np.array(init_elbow_pos)
        if start_joint is None:
            start_joint = self.init_joint_pose

        print(f"start_joint: {start_joint}")
        # sort the grasp_joint_cfg according to the elbow's distance
        print(f"elbow_pos_list: {elbow_pos_list}, init_elbow_pos: {init_elbow_pos}")
        dis = np.linalg.norm(elbow_pos_list - init_elbow_pos, axis=1)
        sorted_grasp_joint_cfg = grasp_joint_cfg[np.argsort(dis)]
        sorted_grasp_poses_list = grasp_poses_list[np.argsort(dis)]
        sorted_elbow_pos_list = elbow_pos_list[np.argsort(dis)]
        # cfg_pool is used to choose the new start point for web RRT
        self.cfg_pool = []
        # Find first path part
        remain_grasp_joint_cfg = copy.deepcopy(sorted_grasp_joint_cfg)
        for joint_cfg_idx, joint_cfg in enumerate(sorted_grasp_joint_cfg):
            if len(path_list) == 0:
                sub_joint_bounds = copy.deepcopy(self.joint_bounds)
                sub_joint_bounds[0] = (min(joint_cfg[0], start_joint[0]) - 0.03,
                                       max(joint_cfg[0], start_joint[0]) + 0.03)
                sub_joint_bounds[1] = (min(joint_cfg[1], start_joint[1]) - 0.02,
                                       max(joint_cfg[1], start_joint[1]) + 0.02)
                sub_joint_bounds[2] = (min(joint_cfg[2], start_joint[2]) - 0.02,
                                       max(joint_cfg[2], start_joint[2]) + 0.02)
                sub_joint_bounds[3] = (min(joint_cfg[3], start_joint[3]) - 0.02,
                                       max(joint_cfg[3], start_joint[3]) + 0.02)
                
                sub_joint_bounds[4] = (min(joint_cfg[4], start_joint[4]) - 0.02,
                                       max(joint_cfg[4], start_joint[4]) + 0.02)
                sub_joint_bounds[5] = (min(joint_cfg[5], start_joint[5]) - 0.02,
                                       max(joint_cfg[5], start_joint[5]) + 0.02)

                self.pb_ompl_setup(custom_init_joint_pose=start_joint,
                                   custom_joint_bound=sub_joint_bounds)

                # Calculte the relationship between the sim_object and gripper
                pos_orn = pack_pose(sorted_grasp_poses_list[joint_cfg_idx])
                self.sim_target_object_id, (sim_tar_pos, sim_tar_orn) = self.create_object_bounding(target_pointcloud)
                (relative_pos,
                 relative_orn) = self.get_relative_pos_orn(gripper_initial_pos = pos_orn[:3],
                                                           gripper_initial_orn = ros_quat(pos_orn[3:]),
                                                           target_pos = sim_tar_pos,
                                                           target_orn = sim_tar_orn)

                (res, path,
                elbow_path,
                gripper_pos_path,
                gripper_orn_path) = self.pb_ompl_interface.plan(joint_cfg[:6],
                                                                goal_mat = sorted_grasp_poses_list[joint_cfg_idx],
                                                                interpolate_num=waypoint_num,
                                                                allowed_time=4,
                                                                sim_target_object_id=self.sim_target_object_id,
                                                                relative_pos=relative_pos,
                                                                relative_orn=relative_orn)
                # remove sim_target_object after planning
                p.removeBody(self.sim_target_object_id)
                if res:
                    path_list.append(path)
                    elbow_path_list.append(elbow_path)
                    gripper_pos_list.append(gripper_pos_path)
                    gripper_orn_list.append(gripper_orn_path)
                    
                    file.write(f"Path {len(path_list) - 1}:\n")
                    for idx, gripper_pos in enumerate(gripper_pos_path[:-1]):
                        self.cfg_pool.append({"pos": gripper_pos, "orn": gripper_orn_path[idx],
                                         "path_num":len(path_list)-1, "waypoint_num":idx})

                        file.write(f"{gripper_pos}\n")
                    file.write(f"\n")
                    np.delete(remain_grasp_joint_cfg, joint_cfg_idx)
                    np.delete(sorted_elbow_pos_list, joint_cfg_idx)
                    np.delete(sorted_grasp_poses_list, joint_cfg_idx)
                    break
        
        for idx, joint_cfg in enumerate(remain_grasp_joint_cfg):
            
            print(f"web RRT2!!!!!!!!!!!!!!!")
            path_idx, waypoint_idx = self.select_pre_waypoint(grasp_pose_mat=sorted_grasp_poses_list[idx])
            if path_idx == 0 and waypoint_idx == 0:
                start_state = self.init_joint_pose
            else:
                start_state = path_list[path_idx][waypoint_idx][:6]
            extend_length = waypoint_num - waypoint_idx
            # Planning in new configuration subspace
            sub_joint_bounds = copy.deepcopy(self.joint_bounds)
            sub_joint_bounds[0] = (min(joint_cfg[0], start_state[0]) - 0.03,
                                   max(joint_cfg[0], start_state[0]) + 0.03)
            sub_joint_bounds[1] = (min(joint_cfg[1], start_state[1]) - 0.02,
                                    max(joint_cfg[1], start_state[1]) + 0.02)
            sub_joint_bounds[2] = (min(joint_cfg[2], start_state[2]) - 0.02,
                                    max(joint_cfg[2], start_state[2]) + 0.02)
            sub_joint_bounds[3] = (min(joint_cfg[3], start_state[3]) - 0.02,
                                    max(joint_cfg[3], start_state[3]) + 0.02)
            

            sub_joint_bounds[4] = (min(joint_cfg[4], start_state[4]) - 0.02,
                                   max(joint_cfg[4], start_state[4]) + 0.02)
            sub_joint_bounds[5] = (min(joint_cfg[5], start_state[5]) - 0.02,
                                   max(joint_cfg[5], start_state[5]) + 0.02)

            # sub_joint_bounds[4] = (-2., 2)
            self.pb_ompl_setup(custom_init_joint_pose=start_state, custom_joint_bound=sub_joint_bounds)
            
            # Calculte the relationship between the sim_object and gripper
            pos_orn = pack_pose(sorted_grasp_poses_list[joint_cfg_idx])
            self.sim_target_object_id, (sim_tar_pos, sim_tar_orn) = self.create_object_bounding(target_pointcloud)
            (relative_pos,
             relative_orn) = self.get_relative_pos_orn(gripper_initial_pos = pos_orn[:3],
                                                       gripper_initial_orn = ros_quat(pos_orn[3:]),
                                                       target_pos = sim_tar_pos,
                                                       target_orn = sim_tar_orn)
            (res, extend_path,
            extend_elbow_path,
            extend_gripper_pos_path,
            extend_gripper_orn_path) = self.pb_ompl_interface.plan(joint_cfg[:6],
                                                                   goal_mat = sorted_grasp_poses_list[idx],
                                                                   interpolate_num=extend_length,
                                                                   allowed_time=2,
                                                                   sim_target_object_id=self.sim_target_object_id,
                                                                   relative_pos=relative_pos,
                                                                   relative_orn=relative_orn)
            print(f"extend_path: {len(extend_path)}\n\n")

            p.removeBody(self.sim_target_object_id)
            if res:
                path = copy.deepcopy(path_list[path_idx][:-len(extend_path)]) if extend_length < waypoint_num else []
                path.extend(extend_path)
                path_list.append(path)

                elbow_path = copy.deepcopy(elbow_path_list[path_idx][:-len(extend_elbow_path)]) if extend_length < waypoint_num else []
                elbow_path.extend(extend_elbow_path)
                elbow_path_list.append(elbow_path)

                gripper_pos_path = copy.deepcopy(gripper_pos_list[path_idx][:-len(extend_gripper_pos_path)]) if extend_length < waypoint_num else []
                gripper_pos_path.extend(extend_gripper_pos_path)
                gripper_pos_list.append(gripper_pos_path)

                gripper_orn_path = copy.deepcopy(gripper_orn_list[path_idx][:-len(extend_gripper_orn_path)]) if extend_length < waypoint_num else []
                gripper_orn_path.extend(extend_gripper_orn_path)
                gripper_orn_list.append(gripper_orn_path)

                file.write(f"Path {len(path_list) - 1}:\n")
                new_start_idx = waypoint_num - 1 - extend_length # The maximum idx is 39, be careful
                for idx, gripper_pos in enumerate(extend_gripper_pos_path[:-1]):
                    self.cfg_pool.append({"pos": gripper_pos, "orn": extend_gripper_orn_path[idx],
                                            "path_num":len(path_list)-1, "waypoint_num":idx + new_start_idx})
                    file.write(f"{gripper_pos}\n")
                file.write(f"\n")

        path_lengths = [len(path) for path in path_list]
        print(f"path_list lengths: {path_lengths}")

        file.close()  # Close the file after writing
        return path_list, elbow_path_list, gripper_pos_list, gripper_orn_list
    
    def grasp_pose2grasp_joint(self, grasp_poses, grasp_scores):        
        if grasp_scores is not None:
            # This function convert the grasp poses into joint configs
            grasp_joint_list = []
            score_list = []
            grasp_poses_list = []
            for idx, grasp_array in enumerate(grasp_poses):
                pos_orn = pack_pose(grasp_array)
                # pos_orn_list.append(pos_orn)
                joint_cfg = list(p.calculateInverseKinematics(self.env._panda.pandaUid,
                                                        self.env._panda.pandaEndEffectorIndex,
                                                        pos_orn[:3],
                                                        ros_quat(pos_orn[3:]),
                                                        maxNumIterations=500,
                                                        residualThreshold=1e-8))

                grasp_joint_list.append(joint_cfg[:6])                
                score_list.append(grasp_scores[idx])
                grasp_poses_list.append(grasp_array)
                new_joint_cfg = copy.deepcopy(joint_cfg)
                new_grasp_array = copy.deepcopy(grasp_array)
                new_joint_cfg[5] += np.pi if new_joint_cfg[5] < 0 else -np.pi
                # new_grasp_array[0,:3] *= -1
                # new_grasp_array[1,:3] *= -1
                new_grasp_array[:3, 0] *= -1
                new_grasp_array[:3, 1] *= -1
                grasp_joint_list.append(new_joint_cfg[:6])
                score_list.append(grasp_scores[idx])
                grasp_poses_list.append(new_grasp_array)

            grasp_joint_list = np.array(grasp_joint_list)
            self.robot = pb_ompl.PbOMPLRobot(self.env._panda.pandaUid, self.joint_idx, self.init_joint_pose)
            # self.obstacles = [self.env.plane_id, self.sim_furniture_id, self.sim_target_id]
            self.obstacles = [self.env.plane_id, self.sim_furniture_id]
            self.setup_collision_detection(self.obstacles)
            self.pb_ompl_interface = pb_ompl.PbOMPL(self.robot, self.obstacles)
            valid_joint_list = []
            valid_elbow_list = []
            valid_score_list = []
            valid_grasp_list = []
            valid_first3_list = []
            for idx, grasp_joint in enumerate(grasp_joint_list):
                if self.pb_ompl_interface.is_state_valid(grasp_joint):
                    valid_joint_list.append(grasp_joint)
                    elbow_pos, _ = p.getLinkState(self.env._panda.pandaUid, 5)[4:6]
                    first3 = []
                    for i in range(1, 5):
                        first3.append(p.getLinkState(self.env._panda.pandaUid, i)[4:5][0])
                    
                    # eleminate ik error
                    pos, orn = p.getLinkState(self.env._panda.pandaUid, self.env._panda.pandaEndEffectorIndex)[4:6]
                    grasp_array = self.pos_orn2matrix_single(pos, orn)
                    
                    valid_first3_list.append(first3)
                    valid_elbow_list.append(elbow_pos)
                    valid_score_list.append(score_list[idx])
                    valid_grasp_list.append(grasp_array)
                # time.sleep(0.05)

            return valid_joint_list, valid_grasp_list, valid_elbow_list, valid_first3_list, valid_score_list
        else:
            """
            Turn poses in cartesian into joint space and then check they are valid or not 
            """
            # The grasp_poses here are middle waypoint in cartesian space
            mid_joint_list = []
            for idx, mid_waypoint in enumerate(grasp_poses):
                pos_orn = pack_pose(mid_waypoint)
                # pos_orn_list.append(pos_orn)
                mid_joint_cfg = list(p.calculateInverseKinematics(self.env._panda.pandaUid,
                                                        self.env._panda.pandaEndEffectorIndex,
                                                        pos_orn[:3],
                                                        ros_quat(pos_orn[3:]),
                                                        maxNumIterations=500,
                                                        residualThreshold=1e-8))
                mid_joint_list.append(mid_joint_cfg[:6])
            
            mid_joint_list = np.array(mid_joint_list)
            self.robot = pb_ompl.PbOMPLRobot(self.env._panda.pandaUid, self.joint_idx, self.init_joint_pose)
            valid_list = []
            self.obstacles = [self.env.plane_id, self.sim_furniture_id]
            self.setup_collision_detection(self.obstacles)
            self.pb_ompl_interface = pb_ompl.PbOMPL(self.robot, self.obstacles)
            # mid_joint_list contain all joint config and valid_list indicate the config is valid or not, 
            # 1 for valid, 0 for invalid
            for idx, mid_joint in enumerate(mid_joint_list):
                if self.pb_ompl_interface.is_state_valid(mid_joint):
                    valid_list.append(1)
                else:
                    valid_list.append(0)
            valid_list = np.array(valid_list)
            return mid_joint_list, valid_list

    def freeze_release(self, option=True, target_ids=None):
        # This function will freeze target or release object, True for freeze
        if target_ids is None:
            target_ids = []
            for idx, placed in enumerate(self.env.placed_objects):
                if placed:
                    target_ids.append(self.env._objectUids[idx])
        
        if option:
            for target_id in target_ids:
                (target_pos,
                 target_ori) = p.getBasePositionAndOrientation(target_id)
                fixed_joint_constraint = p.createConstraint(parentBodyUniqueId=target_id,
                                                            parentLinkIndex=-1,
                                                            childBodyUniqueId=-1,
                                                            childLinkIndex=-1,
                                                            jointType=p.JOINT_FIXED,
                                                            jointAxis=[0, 0, 0],
                                                            parentFramePosition=[0, 0, 0],
                                                            childFramePosition=target_pos,
                                                            childFrameOrientation=target_ori)
                self.targets_dict[target_id] = [target_pos, target_ori, fixed_joint_constraint]
        else:
            for target_id in target_ids:
                p.removeConstraint(self.targets_dict[target_id][2])


    def replace_target_object(self, placed_object_idx=None):
        if placed_object_idx is None:
            placed_object_idx = self.env._objectUids[self.env.target_idx]
        p.resetBasePositionAndOrientation(placed_object_idx,
                                          self.targets_dict[placed_object_idx][0],
                                          self.targets_dict[placed_object_idx][1])

    def clear_constraints(self):
        self.targets_dict.clear()

    def pb_ompl_setup(self, custom_init_joint_pose=None, custom_joint_bound=None):
        """
        This function set the pb_ompl part up
        """
        if custom_init_joint_pose is None:
            self.robot = pb_ompl.PbOMPLRobot(self.env._panda.pandaUid, self.joint_idx, self.init_joint_pose[:6])
        else:
            self.robot = pb_ompl.PbOMPLRobot(self.env._panda.pandaUid, self.joint_idx, custom_init_joint_pose[:6])
        # self.obstacles = [self.env.plane_id, self.sim_furniture_id, self.sim_target_id]
        self.obstacles = [self.env.plane_id, self.sim_furniture_id]
        self.setup_collision_detection(self.obstacles)
        if custom_joint_bound is None:
            self.pb_ompl_interface = pb_ompl.PbOMPL(self.robot, self.obstacles, joint_bounds=self.joint_bounds)
        else:
            self.pb_ompl_interface = pb_ompl.PbOMPL(self.robot, self.obstacles, joint_bounds=custom_joint_bound)
        self.pb_ompl_interface.set_planner("BITstar")


    def pos_orn2matrix(self, position_path_list, quaternion_path_list):
        all_mat_list = []
        for position_path, quaternion_path in zip(position_path_list, quaternion_path_list):
            tmp_mat_list = []
            for way_pos, way_qua in zip(position_path, quaternion_path):
                tmp_mat = np.eye(4)
                tmp_mat[:3, :3] = quat2mat(tf_quat(way_qua))
                tmp_mat[:3, 3] = way_pos
                tmp_mat_list.append(tmp_mat)
            all_mat_list.append(tmp_mat_list)
        all_mat_list = np.array(all_mat_list)
        return all_mat_list

    def pos_orn2matrix_single(self, pos, orn):
        mat = np.eye(4)
        mat[:3, :3] = quat2mat(tf_quat(orn))
        mat[:3, 3] = pos
        return mat
    
    def dbscan_grouping(self, elbow_pos_list, first3_list, grasp_joint_list,
                        grasp_score_list, grasp_poses_list, pointcloud=None):
        """
        This function use dbscan to group the grasp pose depend on elbow's position
        """
        highest_joint_cfg_list = []
        highest_elbow_pos_list = []
        highest_grasp_poses_list = []
        epsilon = 0.05  # Maximum distance between samples for one to be considered as in the neighborhood of the other
        min_samples = 1  # Minimum number of samples in a neighborhood for a data point to be considered as a core point
        
        # Group poses by gripper's position with dbscan
        tmp_grasp_list = grasp_poses_list[:, :3, 3]
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples).fit(tmp_grasp_list)
        first_pos_groups = {}
        for idx, label in enumerate(dbscan.labels_):
            first_pos_groups.setdefault(label, []).append(first3_list[idx])

        keys = set(first_pos_groups.keys())
        std_val_list = []
        for key in keys:
            tmp_first3_poses_list = first_pos_groups.pop(key)
            std_val_list.append(self.compute_standard_deviation(np.array(tmp_first3_poses_list),
                                                                txt_name = 'gripper_std.txt'))
        std_val_list = np.array(std_val_list)
        print(f"gripper std_val_list: {std_val_list}")
        print(f"gripper mean of std_val_list: {np.mean(std_val_list, axis=0)}\n")


        # Group poses by elbow's position with dbscan
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples).fit(elbow_pos_list)
        elbow_pos_groups = {}
        joint_value_groups = {}
        scores_groups = {}
        grasp_poses_group = {}
        first_pos_groups = {}
        for idx, label in enumerate(dbscan.labels_):
            elbow_pos_groups.setdefault(label, []).append(elbow_pos_list[idx])
            first_pos_groups.setdefault(label, []).append(first3_list[idx])
            joint_value_groups.setdefault(label, []).append(grasp_joint_list[idx])
            scores_groups.setdefault(label, []).append(grasp_score_list[idx])
            grasp_poses_group.setdefault(label, []).append(grasp_poses_list[idx])
        
        # Get the grasp poses with highest score in each groups
        keys = set(joint_value_groups.keys())
        std_val_list = []
        for key in keys:
            tmp_elbow_poses_list = elbow_pos_groups.pop(key)
            tmp_first3_poses_list = first_pos_groups.pop(key)
            tmp_grasp_joint_cfgs_list = joint_value_groups.pop(key)
            tmp_scores_list = scores_groups.pop(key)
            tmp_grasp_poses_list = grasp_poses_group.pop(key)
            print(f"tmp_grasp_joint_cfgs_list: {len(tmp_grasp_joint_cfgs_list)}")

            # Calculate the standard deviation of first 3 links
            std_val_list.append(self.compute_standard_deviation(np.array(tmp_first3_poses_list)))
            # Choose the grasp pose with most "middle" pose
            if len(tmp_grasp_poses_list) > 1:
                idx = self.select_representative_grasp_poses(tmp_grasp_poses_list)
                # Visualize grouped grasp poses
                # self.visualize_points_grasppose(pointcloud, tmp_grasp_poses_list, repre_idx=idx)
                highest_joint_cfg_list.append(tmp_grasp_joint_cfgs_list[idx])
                highest_elbow_pos_list.append(tmp_elbow_poses_list[idx])
                highest_grasp_poses_list.append(tmp_grasp_poses_list[idx])
        std_val_list = np.array(std_val_list)
        print(f"std_val_list: {std_val_list}")
        print(f"mean of std_val_list: {np.mean(std_val_list, axis=0)}")

        highest_joint_cfg_list = np.array(highest_joint_cfg_list)
        highest_elbow_pos_list = np.array(highest_elbow_pos_list)
        highest_grasp_poses_list = np.array(highest_grasp_poses_list)
        grasp_poses_list = highest_grasp_poses_list
        return highest_joint_cfg_list, highest_elbow_pos_list, highest_grasp_poses_list

    def remove_sim_fureniture(self):
        p.removeBody(self.sim_furniture_id)

    def replace_real_furniture(self):
        p.resetBasePositionAndOrientation(self.env.furniture_id, [0, 0, 4], [0, 0, 0, 1])

    def move_directly(self, joint_config):
        for joint, value in zip(self.joint_idx, joint_config):
            p.resetJointState(self.env._panda.pandaUid, joint, value, targetVelocity=0)

    def select_pre_waypoint(self, grasp_pose_mat, path_num=0, waypoint_num=0):
        pos = grasp_pose_mat[:3, 3]
        orn = mat2quat(grasp_pose_mat[:3, :3])
        pos_dist_list = []
        orn_dist_list = []
        # print(f"self.cfg_pool: {self.cfg_pool}")
        for cfg in self.cfg_pool:
            pos_dist_list.append(np.linalg.norm(pos - cfg["pos"]))
            # orn_dist_list.append(np.arccos(np.abs(np.dot(orn, tf_quat(cfg["orn"])))))
            orn_dist_list.append(np.arccos(np.dot(orn, tf_quat(cfg["orn"]))))
        # print(f"orn_dist_list: {orn_dist_list}")
        filtered_dicts = [cfg for cfg in self.cfg_pool if (np.linalg.norm(cfg["pos"] - pos) < 0.2 and
                                                           np.linalg.norm(orn - tf_quat(cfg["orn"])) > 0.45)]
        if len(filtered_dicts) > 0:
            min_waypoint_dict = min(filtered_dicts, key=lambda x: x["waypoint_num"])
            path_num = min_waypoint_dict["path_num"]
            waypoint_num = min_waypoint_dict["waypoint_num"]
        print(f"path_num: {path_num}")
        print(f"waypoint_num: {waypoint_num}")
        
        return path_num, waypoint_num
    
    def select_representative_grasp_poses(self, grasp_poses_group):
        mean_distances = []
    
        for i in range(len(grasp_poses_group)):
            distances = []
            for j in range(len(grasp_poses_group)):
                if i != j:  # Skip comparing a pose with itself
                    matrix_diff = grasp_poses_group[i] - grasp_poses_group[j]
                    distances.append(np.linalg.norm(matrix_diff, ord='fro'))

            mean_distance = np.mean(distances)
            mean_distances.append(mean_distance)
        return np.argmin(mean_distances)

    def create_grasp_geometry(self, grasp_pose, color=[0, 0, 0], length=0.08, width=0.08):
        """Create a geometry representing a grasp pose as a U shape."""
        # Define the grasp frame
        frame = grasp_pose.reshape(4, 4)
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
        line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(line_set.lines))])
        line_set.transform(frame)

        return line_set
    
    def visualize_points_grasppose(self, scene_points, grasp_list=None, repre_idx=None):
        
        if grasp_list is None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(scene_points[:, :3]))
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            o3d.visualization.draw_geometries([pcd, axes])
            # End of visualization
        else:
            # Visualize pointcloud and grasp pose part
            if repre_idx is None:
                grasp_geometries = [self.create_grasp_geometry(grasp_pose) for grasp_pose in grasp_list]
            else:
                grasp_geometries = [self.create_grasp_geometry(grasp_list[repre_idx], color=[0, 1, 0],
                                                            length=0.1, width=0.1)]
                remain_grasp_list = copy.deepcopy(grasp_list)
                del remain_grasp_list[repre_idx]
                if len(grasp_list) > 0:
                    grasp_geometries.extend([self.create_grasp_geometry(grasp_pose)
                                            for grasp_pose in remain_grasp_list])
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(scene_points[:, :3]))
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            o3d.visualization.draw_geometries([pcd, axes, *grasp_geometries])
            # End of visualization

    def compute_standard_deviation(self, group_list, txt_name="elbow_std.txt"):
        dir_name = "/home/user/RL_TM5_900_pybullet/RL_scenecollision_ws/src/scenecollision/src/std_record/"

        with open(dir_name + txt_name, 'a') as file: 
            std_devs = []
            # print(f"group_list: {group_list.shape}, {group_list}")
            for i in range(4):  # There are 3 points in each pose
                # Extract all the xyz coordinates of the i-th point across all poses
                points = group_list[:, i, :]  # Shape will be (num_poses, 3)
                # print(f"points: {points}")
                # Compute the standard deviation along each axis (x, y, z)
                std_devs.append(np.std(points, axis=0))
            std_devs = np.sum(std_devs, axis=-1)
            # print(f"================")
            # print(f"std_devs: {std_devs}\n")
            # Write a new value to the file
            file.write(str(std_devs))
            file.write('\n')
        return std_devs

    def cartesian_motion_planning(self, goal_mat_list):
        self.move_directly(self.init_joint_pose)
        self.obstacles = [self.env.plane_id, self.sim_furniture_id]
        self.cart_planner = GraspPlanner(obstacles = self.obstacles)
        pos, orn = self.env._get_ef_pose()
        ef_pose_list = [*pos, *orn]
        joint_paths = []
        gripper_pos_paths = []
        gripper_ori_paths = []
        for gaol_mat in goal_mat_list:
            goal_pos = pack_pose(gaol_mat)
            goal_pos[3:] = ros_quat(goal_pos[3:])
            solver = self.cart_planner.plan(ef_pose_list, goal_pos, path_length=40, runTime=1.)
            if solver is None:
                print(f"no path")
                None_path_list = [[[None] * 6 for _ in range(40)]]
                return None_path_list, None_path_list, None_path_list, None_path_list
            path = solver.getSolutionPath().getStates()
            joint_path = []
            gripper_ori_path = []
            gripper_pos_path = []
            for i in range(len(path)):
                waypoint = path[i]
                rot = waypoint.rotation()
                pos_action = [waypoint.getX(), waypoint.getY(), waypoint.getZ()]
                ori_action = [rot.x, rot.y, rot.z, rot.w]
                gripper_pos_path.append(pos_action)
                gripper_ori_path.append(ori_action)
                joint_cfg = list(p.calculateInverseKinematics(self.env._panda.pandaUid,
                                                              self.env._panda.pandaEndEffectorIndex,
                                                              pos_action,
                                                              ori_action,
                                                              maxNumIterations=500,
                                                              residualThreshold=1e-8))[:6]
                # self.move_directly(joint_cfg) # Directly move the arm to the joint_cfg
                # time.sleep(0.1)
                joint_path.append(joint_cfg)
            self.move_directly(self.init_joint_pose)
            joint_paths.append(joint_path)
            gripper_pos_paths.append(gripper_pos_path)
            gripper_ori_paths.append(gripper_ori_path)
        # for gripper_pos_path, gripper_ori_path in zip(gripper_pos_paths, gripper_ori_paths):
        #     self.move_directly(self.init_joint_pose)
        #     self.cart_planner.show_path(gripper_pos_path, gripper_ori_path)
        return joint_paths, None, gripper_pos_paths, gripper_ori_paths
    

    def create_object_bounding(self, target_pointcloud):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(target_pointcloud)
        # Get AABB from point cloud
        # aabb = pcd.get_oriented_bounding_box()
        aabb = pcd.get_axis_aligned_bounding_box()
        # Box or sphere
        max_bound = aabb.get_max_bound()
        min_bound = aabb.get_min_bound()
        bound_dimension = np.array([max_bound[i] - min_bound[i] - 0.03 for i in range(3)])
        print(f"bound_dimension: {bound_dimension}")
        center = aabb.get_center()

        box_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=bound_dimension/2)
        # Create the visual shape for the box
        box_visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=bound_dimension/2,
                                            rgbaColor=[1, 0, 0, 1])
        # Create the box multi-body
        box_id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=box_collision_shape,
                                baseVisualShapeIndex=box_visual_shape,
                                basePosition=center)
        return box_id, p.getBasePositionAndOrientation(box_id)

    def object_collision_check(self, target_pointcloud, gripper_pos_list, gripper_orn_list):
        self.sim_target_object_id, (target_pos, target_orn) = self.create_object_bounding(target_pointcloud)
        for gripper_pos_path, gripper_orn_path in zip(gripper_pos_list, gripper_orn_list):
            gripper_initial_pos = gripper_pos_path[-1]
            gripper_initial_orn = gripper_orn_path[-1]
            if gripper_initial_pos[0] == None:
                return
            (relative_pos,
             relative_orn) = self.get_relative_pos_orn(gripper_initial_pos,
                                                       gripper_initial_orn,
                                                       target_pos, target_orn)
            for waypoint_pos, waypoint_orn in zip(gripper_pos_path, gripper_orn_path):
                target_pos, target_orn = self.get_target_pose(relative_pos,
                                                              relative_orn,
                                                              waypoint_pos,
                                                              waypoint_orn)
                p.resetBasePositionAndOrientation(self.sim_target_object_id, target_pos, target_orn)
                time.sleep(0.05)
        p.resetBasePositionAndOrientation(self.sim_target_object_id, [0, 0, 15], [1, 0, 0, 0])

    def get_target_pose(self, relative_pos, relative_orn, gripper_pos, gripper_orn):
        target_pos, target_orn = p.multiplyTransforms(
            gripper_pos, gripper_orn,
            relative_pos, relative_orn
        )
        return target_pos, target_orn

    def get_relative_pos_orn(self, gripper_initial_pos, gripper_initial_orn, target_pos, target_orn):
        return p.multiplyTransforms(*p.invertTransform(gripper_initial_pos, gripper_initial_orn),
                                    target_pos, target_orn)

    def get_oriented_bound_box_numpy(self, pcd):
        points = np.asarray(pcd.points)

        # Compute the covariance matrix and perform PCA
        cov = np.cov(points.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort eigenvalues and eigenvectors in descending order
        sort_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]

        # Transform points to the PCA-aligned coordinate system
        transformed_points = points @ eigenvectors

        # Compute the axis-aligned bounding box (AABB) in the PCA-aligned coordinate system
        min_bound = np.min(transformed_points, axis=0)
        max_bound = np.max(transformed_points, axis=0)
        aabb_center = (min_bound + max_bound) / 2.0
        aabb_extents = max_bound - min_bound

        # Compute the transformation matrix for the OBB
        rotation_matrix = eigenvectors
        translation = aabb_center @ eigenvectors.T

        # Create the oriented bounding box
        obb_center = translation
        obb_extents = aabb_extents
        obb = o3d.geometry.OrientedBoundingBox(obb_center, rotation_matrix, obb_extents)

        # Visualize the point cloud and the oriented bounding box
        pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Set point cloud color
        obb.color = (1, 0, 0)  # Set OBB color
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd, obb] + [axis_pcd])

@ray.remote(num_cpus=1, num_gpus=0.12)
class ActorWrapper012(ActorWrapper):
    pass


if __name__ == "__main__":
    print(f"in main")