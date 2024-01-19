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
        # self.furniture_name = "carton_box"
        # self.furniture_name = "table"
        self.furniture_name = "cabinet"
        self.env = SimulatedYCBEnv(renders=renders)
        self.env._load_index_objs(test_file_dir)
        self.env.reset(save=False, enforce_face_target=True, furniture=self.furniture_name)
        self.grasp_checker = ValidGraspChecker(self.env)
        # self.planner = GraspPlanner()
        self.target_points = None   # This is for merging point-cloud from different time
        self.obstacle_points = None
        self.simulation_id = simulation_id

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
        self.create_time = 0


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
        
        # return None
        
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
        
        obstacle_points, target_points, scene_points = self.get_pc_state(frame="base", vis=vis, obs_reserve=False)
        

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
        points_world = (np.dot(ef_pose, self.env.cam_offset.dot(pointcloud_tar)).T)[:, :3]

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

    def get_pc_state(self, vis=False, frame="camera", obs_reserve=False):
        """
        The output pointcloud should (N, 4)
        The obs_reserve is for obstacle pointcloud, prevent it vanishing from the furthest_sample, 
        it will stop consider new obstacle points if True
        """
        obstacle_points = self.get_world_pointcloud(raw_data="obstacle")
        target_points = self.get_world_pointcloud(raw_data=False)

        # deal with obstacle points, combine them with previous points if exist, or overwrite it with preious if None
        if self.obstacle_points is not None:
            if obs_reserve:
                obstacle_points = self.obstacle_points
            else:
                if obstacle_points is None:
                    obstacle_points = self.obstacle_points
                else:
                    # combine two pointcloud part, first convert them to tensor
                    obstacle_points_tensor = torch.from_numpy(obstacle_points)
                    self_obstacle_points_tensor = torch.from_numpy(self.obstacle_points)
                    combined_obstacle_points = torch.cat((obstacle_points_tensor, self_obstacle_points_tensor), dim=0).unsqueeze(0)
                    index = farthest_point_sample(combined_obstacle_points, 2048)
                    obstacle_points = index_points(combined_obstacle_points, index).squeeze().detach().numpy()
                    obstacle_points = combined_obstacle_points.squeeze().detach().numpy()

        self.target_points = target_points
        self.obstacle_points = obstacle_points

        # This part is for combining the obstacle pointcloud and target pointcloud,
        # target is obstacle during the motion planning after all
        
        obstacle_points_tensor = torch.from_numpy(self.obstacle_points)
        target_points_tensor = torch.from_numpy(self.target_points)
        combined_points = torch.cat((obstacle_points_tensor, target_points_tensor), dim=0).unsqueeze(0)
        index = farthest_point_sample(combined_points, 2048)
        scene_points = index_points(combined_points, index).squeeze().detach().numpy()

        if obstacle_points is None:
            return None
        obstacle_points = np.hstack((obstacle_points, np.zeros((obstacle_points.shape[0], 1))))

        # # transform back to camera frame
        if frame == "camera":
            obstacle_points = self.base2camera(obstacle_points)

        if vis:
            all_o3d_pc = o3d.geometry.PointCloud()
            all_o3d_pc.points = o3d.utility.Vector3dVector(target_points[:, :3])
            axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            # o3d.visualization.draw_geometries([all_o3d_pc]+[target_o3d_pc]+[axis_pcd])
            o3d.visualization.draw_geometries([all_o3d_pc]+[axis_pcd])

        return obstacle_points, target_points, scene_points

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


    def create_simulation_env(self, obs_pc=None, target_pc=None, grasp_joint_cfg=None):
        # This function is only for sim_actor_id, the real_actor_id won't enter here
        self.env.place_back_objects()
        p.resetBasePositionAndOrientation(self.env.furniture_id, [0, 0, 4], [0, 0, 0, 1])
        # time.sleep(2)

        starttime = time.time()
        # self.sim_furniture_id, self.sim_target_id = self.create_obstacle_from_pc(obs_pc, target_pc)
        self.sim_furniture_id = self.create_obstacle_from_pc(obs_pc, target_pc)
        print(f"simulate duration: {time.time()-starttime}!!!!!!!!!!!!!!!!!")
        path = self.motion_planning(grasp_joint_cfg=grasp_joint_cfg, )
        self.create_time += 1
        
        

        p.removeBody(self.sim_furniture_id)
        # p.removeBody(self.sim_target_id)

        return path

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

        # combined_tar_pc = self.extend_obs_pc(obs_pc=target_pc, target_pc=target_pc, scale_factor=0.005)
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

    def motion_planning(self, grasp_joint_cfg):
        # Deal with the path from reset pose to pregrasp pose, self.robot is initiated here
        # because the point can be tagged valid after searched, so refresh it before every 
        # motion_planning
        print(f"grasp_joint_cfg: {len(grasp_joint_cfg)}\n\n\n\n\n\n\n\n\n\n\n")
        for joint_cfg in grasp_joint_cfg:
            self.robot = pb_ompl.PbOMPLRobot(self.env._panda.pandaUid, self.joint_idx, self.init_joint_pose)
            # self.obstacles = [self.env.plane_id, self.sim_furniture_id, self.sim_target_id]
            self.obstacles = [self.env.plane_id, self.sim_furniture_id]
            self.setup_collision_detection(self.obstacles)
            self.pb_ompl_interface = pb_ompl.PbOMPL(self.robot, self.obstacles)
            self.pb_ompl_interface.set_planner("BITstar")
            self.joint_bounds = []
            for i, joint_id in enumerate(self.joint_idx):
                joint_info = p.getJointInfo(self.env._panda.pandaUid, joint_id)
                low = joint_info[8] # low bounds
                high = joint_info[9] # high bounds
                if low < high:
                    self.joint_bounds.append([low, high])
            self.joint_bounds = np.array(self.joint_bounds)

            # goal_joint_cfg = grasp_joint_cfg[0]
            start_state = np.array(self.env._panda.getJointStates()[0])[:6]
            self.robot.set_state(start_state)
            res, path = self.pb_ompl_interface.plan(joint_cfg[:6])
            if res:
                return path
        # if res:
        #     return path
        return None
    
    def grasp_pose2grasp_joint(self, grasp_poses):        
        # This function convert the grasp poses into joint configs
        grasp_joint_cfg = []
        for grasp_array in grasp_poses:
            pos_orn = pack_pose(grasp_array)
            grasp_joint_cfg.append(p.calculateInverseKinematics(self.env._panda.pandaUid,
                                        self.env._panda.pandaEndEffectorIndex,
                                        pos_orn[:3],
                                        ros_quat(pos_orn[3:]),
                                        maxNumIterations=500,
                                        residualThreshold=1e-8))

        grasp_joint_cfg = np.array(grasp_joint_cfg)

        return grasp_joint_cfg
    
    def freeze_release(self, option=True):
        # This function will freeze target or release object, True for freeze
        if option:
            pos, ori = p.getBasePositionAndOrientation(self.env._objectUids[self.env.target_idx])
            self.fixed_joint_constraint = p.createConstraint(
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
        else:
            p.removeConstraint(self.fixed_joint_constraint)
            self.env.place_back_objects()

    def move2grasp(self, joint_path):

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
        for _ in range(5):
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
        
        self.env.place_back_objects()


@ray.remote(num_cpus=1, num_gpus=0.12)
class ActorWrapper012(ActorWrapper):
    pass


if __name__ == "__main__":
    print(f"in main")