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

class ActorWrapper(object):
    """
    wrapper testing, use ray to create multiple pybullet
    """
    def __init__(self, renders=False, simulation_id=None):
        # from env.ycb_scene import SimulatedYCBEnv
        # file = os.path.join("object_index", 'acronym_90.json')
        file = os.path.join(parent_dir, "object_index", 'proper_objects.json')
        with open(file) as f: file_dir = json.load(f)
        file_dir = file_dir['train']
        # file_dir = file_dir['test']
        file_dir = [f[:-5] for f in file_dir]
        test_file_dir = list(set(file_dir))
        test_file_dir = random.sample(test_file_dir, 15)
        self.furniture_name = "carton_box"
        # self.furniture_name = "table"
        # self.furniture_name = "cabinet"
        self.env = SimulatedYCBEnv(renders=renders)
        self.env._load_index_objs(test_file_dir)
        self.env.reset(save=False, enforce_face_target=True, furniture=self.furniture_name)
        self.grasp_checker = ValidGraspChecker(self.env)
        # self.planner = GraspPlanner()
        self.target_points = None   # This is for merging point-cloud from different time
        self.obstacle_points = None
        self.simulation_id = simulation_id
        self.sim_furniture_id = None
        self.joint_bounds = list(zip(self.env._panda._joint_min_limit, self.env._panda._joint_max_limit))[:6]
        self.joint_bounds[0] = (-1.57, 1.57)
        # disable the collision between the basse of TM5 and plane        
        p.setCollisionFilterPair(self.env.plane_id, self.env._panda.pandaUid, -1, 0, enableCollision=False)
        
        # This part is for pybullet_ompl
        self.joint_idx = [1, 2, 3, 4, 5, 6] # The joint idx that is considerated in pynullet_ompl
        self.obstacles = [self.env.plane_id, self.env.furniture_id] # Set up the obstacles
        if self.furniture_name == "cabinet":
            self.init_joint_pose = [-0.15, -1.55, 1.8, -0.1, 1.8, 0.0, 0.0, 0.0, 0.0]
        elif self.furniture_name == "table":
            self.init_joint_pose = [-0., -0.95, 1.9, -0.1, 1.571, 0.0, 0.0, 0.0, 0.0]
        elif self.furniture_name == "carton_box":
            self.init_joint_pose = [0.03, -1., 1.9, -0.1, 1.571, 0.0, 0.0, 0.0, 0.0]

        # Helper3D part, load first and then move the arm in function "get_surface_points"
        self.URDFPATH = f"/home/user/RL_TM5_900_pybullet/env/models/tm5_900/tm5_900_with_gripper_.urdf"
        self.helper3d_arm_urdf = getURDF(self.URDFPATH, JointInfo=False, from_visual=False)
        self.helper3d_link_names = ["shoulder_1_link", "arm_1_link", "arm_2_link", "wrist_1_link", "wrist_2_link", "wrist_3_link"]


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
            if target_points is not None:
                if concat:
                    self.target_points = self.concatenate_pc(self.target_points, target_points)
                    target_points = np.hstack((self.target_points, np.ones((self.target_points.shape[0], 1))))
                else:
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


    def create_simulation_env(self, obs_pc=None, target_pc=None, grasp_poses=None, grasp_scores=None):
        if obs_pc is not None and target_pc is not None:
            if self.sim_furniture_id is not None:
                self.remove_sim_fureniture()
            # This function is only for sim_actor_id, the real_actor_id won't enter here
            self.env.place_back_objects()
            p.resetBasePositionAndOrientation(self.env.furniture_id, [0, 0, 4], [0, 0, 0, 1])


            starttime = time.time()
            # self.sim_furniture_id, self.sim_target_id = self.create_obstacle_from_pc(obs_pc, target_pc)
            self.sim_furniture_id = self.create_obstacle_from_pc(obs_pc, target_pc)
            print(f"simulate duration: {time.time()-starttime}!!!!!!!!!!!!!!!!!")
            # Be careful, the list of score will increase due to the rotate of the 6th joint
            (grasp_joint_list, grasp_poses_list,
            elbow_pos_list, grasp_score_list) = self.grasp_pose2grasp_joint(grasp_poses=grasp_poses,
                                                                        grasp_scores=grasp_scores)
            grasp_joint_list = np.array(grasp_joint_list)
            elbow_pos_list = np.array(elbow_pos_list)
            grasp_poses_list = np.array(grasp_poses_list)
            path_start_time = time.time()
            if len(elbow_pos_list) == 0:
                None_path_list = [[[None] * 6 for _ in range(30)]]
                print(f"None_path_list: {np.array(None_path_list).shape}")
                return None_path_list,None_path_list, None_path_list, None_path_list, None_path_list  # make the None path list's shape like (1, 30, 6)
            elif len(elbow_pos_list) == 1:
                (path_list,
                elbow_path_list,
                gripper_pos_list,
                gripper_orn_list) = self.motion_planning(grasp_joint_cfg=grasp_joint_list,
                                                        elbow_pos_list=elbow_pos_list)
            else:
                (highest_joint_cfg_list,
                 highest_elbow_pos_list,
                 highest_grasp_poses_list) = self.dbscan_grouping(elbow_pos_list,
                                                                  grasp_joint_list,
                                                                  grasp_score_list,
                                                                  grasp_poses_list)
                grasp_poses_list = highest_grasp_poses_list
                (path_list, 
                elbow_path_list, 
                gripper_pos_list, 
                gripper_orn_list) = self.motion_planning(grasp_joint_cfg=highest_joint_cfg_list,
                                                        elbow_pos_list=highest_elbow_pos_list)

            print(f"path_planning_time: {time.time() - path_start_time}")
            
            # p.removeBody(self.sim_furniture_id)
            # p.removeBody(self.sim_target_id)


            return path_list, grasp_poses_list, elbow_path_list, gripper_pos_list, gripper_orn_list
        else:
            return self.grasp_pose2grasp_joint(grasp_poses=grasp_poses, grasp_scores=None)
            

    def extend_obs_pc(self, obs_pc, target_pc, scale_factor=0.02):
        # This function "extend" the obstacle pointcloud outward a little,
        # away from middle point of target pointcloud.
        # By doing this, the mesh created afterward can be more reasonable.
        target_middle = np.mean(target_pc, axis=0)
        vector_outward = obs_pc - target_middle
        vector_length = np.linalg.norm(vector_outward, axis=1, keepdims=True)
        normalized_vector = vector_outward / vector_length

        # Scale and move obstacle points away from the middle point
        moved_obs_pc = obs_pc + scale_factor * normalized_vector
        
        combined_pc = np.concatenate((moved_obs_pc, obs_pc), axis=0)

        return combined_pc
    
    def save_obj_file(self, alphashape, file_name="obs_shape.obj"):
        # This function save the alphashape object into obj file
        with open(file_name, "w") as obj_file:
            # Write vertices
            for vertex in alphashape.vertices:
                obj_file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

            # Write faces (1-based indexing for OBJ format)
            for face in alphashape.faces:
                obj_file.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


    def create_obstacle_from_pc(self, obs_pc, target_pc):
        # This function use the 2 pointcloud to create a object in pybullet
        combined_obs_pc = self.extend_obs_pc(obs_pc=obs_pc, target_pc=target_pc)
        obs_alph = alphashape.alphashape(combined_obs_pc, 8)

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

        # combined_tar_pc = self.extend_obs_pc(obs_pc=target_pc, target_pc=target_pc, scale_factor=-0.002)
        # tar_alph = alphashape.alphashape(combined_tar_pc, 30)

        # tar_vertices = tar_alph.vertices
        # tar_faces = np.array(tar_alph.faces).flatten()
        
        # tar_visualShapeId = p.createVisualShape(
        #                     shapeType=p.GEOM_MESH,
        #                     flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
        #                     vertices=tar_vertices,
        #                     indices=tar_faces,
        #                     meshScale=[1, 1, 1]
        #                 )

        # tar_collisionShapeId = p.createCollisionShape(
        #                     shapeType=p.GEOM_MESH,
        #                     flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
        #                     vertices=tar_vertices,
        #                     indices=tar_faces,
        #                     meshScale=[1, 1, 1]
        #                 )

        # tar_body_id = p.createMultiBody(
        #                 baseMass=1,
        #                 baseInertialFramePosition=[0, 0, 0],
        #                 baseCollisionShapeIndex=tar_collisionShapeId,
        #                 baseVisualShapeIndex=tar_visualShapeId,
        #                 basePosition=[0, 0, 0],
        #                 baseOrientation=[0, 0, 0, 1]
        #             )

        # return obs_body_id, tar_body_id
        return obs_body_id

    def motion_planning(self, grasp_joint_cfg, elbow_pos_list=None):

        path_list = []
        elbow_path_list = []
        
        gripper_pos_list = []
        gripper_orn_list = []
        init_elbow_pos, _ = p.getLinkState(self.env._panda.pandaUid, 5)[4:6]
        init_elbow_pos = np.array(init_elbow_pos)
        # sort the grasp_joint_cfg according to the elbow's distance
        dis = np.linalg.norm(elbow_pos_list - init_elbow_pos, axis=1)
        sorted_grasp_joint_cfg = grasp_joint_cfg[np.argsort(dis)]
        
        mean_waypoint_pos = np.mean(sorted_grasp_joint_cfg, axis=0)[:6]

        mean_waypoint_pos = np.average([mean_waypoint_pos, self.init_joint_pose[:6]],
                                       weights=[0.35, 0.65],
                                       axis=0)
        # mean_waypoint_pos[3:6] = self.init_joint_pose[3:6]
        print(f"mean_waypoint_pos: {mean_waypoint_pos}")
        print(f"self.init_joint_pose: {self.init_joint_pose}")

        print(f"sorted_grasp_joint_cfg: {len(sorted_grasp_joint_cfg)}")
        for idx, joint_cfg in enumerate(sorted_grasp_joint_cfg):    
            if len(path_list) == 0:
                # Plan a approaching sub-path for the whole path, to reduce the searching area
                self.pb_ompl_setup()
                (res, pre_path,
                 pre_elbow_path,
                 pre_gripper_pos_path,
                 pre_gripper_orn_path) = self.pb_ompl_interface.plan(mean_waypoint_pos,
                                                                     interpolate_num=10)
                self.pb_ompl_setup(custom_init_joint_pose=mean_waypoint_pos)

                (res, path,
                elbow_path,
                gripper_pos_path,
                gripper_orn_path) = self.pb_ompl_interface.plan(joint_cfg[:6], interpolate_num=20)
                if res:
                    pre_path.extend(path)
                    pre_elbow_path.extend(elbow_path)
                    pre_gripper_pos_path.extend(gripper_pos_path)
                    pre_gripper_orn_path.extend(gripper_orn_path)

                    path_list.append(pre_path)
                    elbow_path_list.append(pre_elbow_path)
                    gripper_pos_list.append(pre_gripper_pos_path)
                    gripper_orn_list.append(pre_gripper_orn_path)
            else:
                print(f"web RRT2!!!!!!!!!!!!!!!")
                start_state = path_list[0][10][:6]
                # Planning in new configuration subspace
                sub_joint_bounds = copy.deepcopy(self.joint_bounds)
                # sub_joint_bounds[0] = (max(min(joint_cfg[1], start_state[1]- 0.1), self.joint_bounds[0][0]),
                #                         min(max(joint_cfg[1], start_state[1])+0.1, self.joint_bounds[0][1]))
                sub_joint_bounds[1] = (min(joint_cfg[1], start_state[1]) - 0.02,
                                        max(joint_cfg[1], start_state[1]) + 0.02)
                sub_joint_bounds[2] = (min(joint_cfg[2], start_state[2]) - 0.02,
                                        max(joint_cfg[2], start_state[2]) + 0.02)
                sub_joint_bounds[3] = (min(joint_cfg[3], start_state[3]) - 0.02,
                                        max(joint_cfg[3], start_state[3]) + 0.02)
                # sub_joint_bounds[4] = (-2., 2)
                self.pb_ompl_setup(custom_init_joint_pose=start_state, custom_joint_bound=sub_joint_bounds)
                (res, extend_path,
                extend_elbow_path,
                extend_gripper_pos_path,
                extend_gripper_orn_path) = self.pb_ompl_interface.plan(joint_cfg[:6], interpolate_num=20, allowed_time=4)
                print(f"extend_path: {len(extend_path)}\n\n")
                if res:
                    path = copy.deepcopy(path_list[0][:-len(extend_path)])
                    path.extend(extend_path)
                    path_list.append(np.array(path))

                    elbow_path = copy.deepcopy(elbow_path_list[0][:-len(extend_elbow_path)])
                    elbow_path.extend(extend_elbow_path)
                    elbow_path_list.append(np.array(elbow_path))

                    gripper_pos_path = copy.deepcopy(gripper_pos_list[0][:-len(extend_gripper_pos_path)])
                    gripper_pos_path.extend(extend_gripper_pos_path)
                    gripper_pos_list.append(np.array(gripper_pos_path))

                    gripper_orn_path = copy.deepcopy(gripper_orn_list[0][:-len(extend_gripper_orn_path)])
                    gripper_orn_path.extend(extend_gripper_orn_path)
                    gripper_orn_list.append(np.array(gripper_orn_path))


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

                grasp_joint_list.append(joint_cfg)
                score_list.append(grasp_scores[idx])
                grasp_poses_list.append(grasp_array)
                new_joint_cfg = copy.deepcopy(joint_cfg)
                new_grasp_array = copy.deepcopy(grasp_array)
                new_joint_cfg[5] += np.pi if new_joint_cfg[5] < 0 else -np.pi
                new_grasp_array[0,:3] *= -1
                new_grasp_array[1,:3] *= -1
                grasp_joint_list.append(new_joint_cfg)
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
            for idx, grasp_joint in enumerate(grasp_joint_list):
                if self.pb_ompl_interface.is_state_valid(grasp_joint):
                    valid_joint_list.append(grasp_joint)
                    elbow_pos, _ = p.getLinkState(self.env._panda.pandaUid, 5)[4:6]
                    valid_elbow_list.append(elbow_pos)
                    valid_score_list.append(score_list[idx])
                    valid_grasp_list.append(grasp_poses_list[idx])
                # time.sleep(0.05)

            return valid_joint_list, valid_grasp_list, valid_elbow_list, valid_score_list
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
    
    def check_inverse_kinematic(self, pose_orn):
        # This function check wheather the end-effector's pose is close enough to the one
        # before inverse kinematic.

        pos, orn = p.getLinkState(self.env._panda.pandaUid, self.env._panda.pandaEndEffectorIndex)[4:6]
        pos = list(pos)
        orn = list(orn)
        pos_error = (np.square(pos - pose_orn[:3])**2).mean()
        orn_error = (np.square(orn - pose_orn[3:])**2).mean()
        print(f"orn_error: {orn_error} pos_error: {pos_error}\n\n")
        return (orn_error < 0.1 and pos_error < 0.01)

    def freeze_release(self, option=True):
        # This function will freeze target or release object, True for freeze
        if option:
            self.target_pos, self.target_ori = p.getBasePositionAndOrientation(self.env._objectUids[self.env.target_idx])
            self.fixed_joint_constraint = p.createConstraint(
                parentBodyUniqueId=self.env._objectUids[self.env.target_idx],
                parentLinkIndex=-1,
                childBodyUniqueId=-1,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=self.target_pos,
                childFrameOrientation=self.target_ori
                )
        else:
            p.removeConstraint(self.fixed_joint_constraint)
            # self.env.place_back_objects()

    def move2grasp(self, joint_path):

        # This part is for the path after grasping
        # Use copy.deepcopy because the list is 2dlist, use deepcopy to copy the whole list(including 
        # the 1d list in it), otherwise the second dimension(which is 1) still point to the same memory
        joint_path = joint_path.tolist()
        retreat_joint_path = copy.deepcopy(joint_path)
        retreat_joint_path.reverse()

        middle_pc_list = []
        camera2base_list = []
        for idx, joint_con in enumerate(joint_path):
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

            if idx % 6 == 0:
                middle_pc_list.append(self.get_pc_state(frame="camera", target_only=True))
                ef_pose = self.env._get_ef_pose('mat')
                camera2base_list.append(np.dot(ef_pose, np.linalg.inv(self.env.cam_offset)))
            for _ in range(200):
                p.stepSimulation()

        # Slowly move to grasp pose
        for _ in range(4):
            self.env.step([0, 0, 0.01, 0, 0, 0])
        
        # Start to grasp
        p.removeConstraint(self.fixed_joint_constraint)
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
        
        return middle_pc_list, camera2base_list

    def move2approach(self):
        Joints = p.getJointStates(self.env._panda.pandaUid, [i for i in range(1, 7)])
        Joints = np.array([i[0] for i in Joints])
        print(f"Joints: {Joints}")
        cur_state = np.append(Joints, 0)
        # defined_position = [-0.1, -0.35, 1.75, -0.6, 1.571, 0.0, 0]  # right
        # defined_position = [0.1, -0.35, 1.75, -0.6, 1.571, 0.0, 0]  # middle
        defined_position = [0.13, -0.9, 1.9, -0.1, 1.571, 0.0, 0.0]  # left
        way_point_num = 10
        average_action = (defined_position - cur_state) / way_point_num
        for i in range(1, way_point_num+1):
            state = average_action * i + cur_state
            self.env.step(state, config=True, repeat=300)[0]


    def replace_target_object(self):
        p.resetBasePositionAndOrientation(self.env._objectUids[self.env.target_idx],
                                          self.target_pos, self.target_ori)

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

    
    def dbscan_grouping(self, elbow_pos_list, grasp_joint_list, grasp_score_list, grasp_poses_list):
        """
        This function use dbscan to group the grasp pose depend on elbow's position
        """
        highest_joint_cfg_list = []
        highest_elbow_pos_list = []
        highest_grasp_poses_list = []
        epsilon = 0.03  # Maximum distance between samples for one to be considered as in the neighborhood of the other
        min_samples = 1  # Minimum number of samples in a neighborhood for a data point to be considered as a core point
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples).fit(elbow_pos_list)
        elbow_pos_groups = {}
        joint_value_groups = {}
        scores_groups = {}
        grasp_poses_group = {}
        for idx, label in enumerate(dbscan.labels_):
            elbow_pos_groups.setdefault(label, []).append(elbow_pos_list[idx])
            joint_value_groups.setdefault(label, []).append(grasp_joint_list[idx])
            scores_groups.setdefault(label, []).append(grasp_score_list[idx])
            grasp_poses_group.setdefault(label, []).append(grasp_poses_list[idx])
        
        # Get the grasp poses with highest score in each groups
        keys = set(joint_value_groups.keys())
        for key in keys:
            tmp_elbow_poses_list = elbow_pos_groups.pop(key)
            tmp_grasp_joint_cfgs_list = joint_value_groups.pop(key)
            tmp_scores_list = scores_groups.pop(key)
            tmp_grasp_poses_list = grasp_poses_group.pop(key)
            highest_joint_cfg_list.append(tmp_grasp_joint_cfgs_list[np.argmax(tmp_scores_list)])
            highest_elbow_pos_list.append(tmp_elbow_poses_list[np.argmax(tmp_scores_list)])
            highest_grasp_poses_list.append(tmp_grasp_poses_list[np.argmax(tmp_scores_list)])

        highest_joint_cfg_list = np.array(highest_joint_cfg_list)
        highest_elbow_pos_list = np.array(highest_elbow_pos_list)
        highest_grasp_poses_list = np.array(highest_grasp_poses_list)
        grasp_poses_list = highest_grasp_poses_list

        return highest_joint_cfg_list, highest_elbow_pos_list, highest_grasp_poses_list

    def remove_sim_fureniture(self):
        p.removeBody(self.sim_furniture_id)

    def move_directly(self, joint_config):
        for joint, value in zip(self.joint_idx, joint_config):
            p.resetJointState(self.env._panda.pandaUid, joint, value, targetVelocity=0)


    # def linear_elbow_path_planning(self, elbow_pos_list, joint_config_list):
    #     self.env.reset(save=False,
    #                    enforce_face_target=False,
    #                    init_joints=self.init_joint_pose,
    #                    reset_free=True)
    #     init_elbow_pos = p.getLinkState(self.env._panda.pandaUid, 5)[4]
        


@ray.remote(num_cpus=1, num_gpus=0.12)
class ActorWrapper012(ActorWrapper):
    pass


if __name__ == "__main__":
    print(f"in main")