import pybullet as p
import numpy as np
import time
import json
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import ray
import os
import datetime
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils.utils import *
from utils.planner import GraspPlanner
from utils.grasp_checker import ValidGraspChecker
from env.ycb_scene import SimulatedYCBEnv
from utils.planner import GraspPlanner
from replay_buffer import ReplayBuffer


class ActorWrapper(object):
    """
    wrapper testing, use ray to create multiple pybullet
    """
    def __init__(self, buffer_id, policy_id):
        # from env.ycb_scene import SimulatedYCBEnv
        file = os.path.join("object_index", 'acronym_90.json')
        with open(file) as f: file_dir = json.load(f)
        file_dir = file_dir['train']
        # file_dir = file_dir['test']
        file_dir = [f[:-5] for f in file_dir]
        test_file_dir = list(set(file_dir))
        test_file_dir = random.sample(test_file_dir, 15)
        self.env = SimulatedYCBEnv(renders=False)
        self.env._load_index_objs(test_file_dir)
        self.env.reset(save=False, enforce_face_target=True)
        self.grasp_checker = ValidGraspChecker(self.env)
        self.planner = GraspPlanner()
        self.buffer_id = buffer_id
        self.policy_id = policy_id

        self.target_points = None   # This is for merging point-cloud from different time
        self.obstacle_points = None

    def rollout_once(self, expert=True):
        start = time.time()
        self.env.reset(save=False, enforce_face_target=False, reset_free=True)
        if expert is True:
            rewards = self.expert_move()
        else:
            """
            Use actor to react to the state
            """
            rewards = self.policy_move()
            raise NotImplementedError
        duration = time.time() - start
        print(f"actor duration: {duration}")
        return rewards

    def test(self):
        for _ in range(2):
            print(f"{self.id} sleeping")
            time.sleep(1)
        return self.id

    def get_arm_id(self):
        return self.env._panda.pandaUid

    def get_plane_id(self):
        return self.env.plane_id

    def get_grasp_pose(self):
        '''
        Take pre-define grasp dataset of target object as an example.
        Load npy file by object names.
        '''

        scale_str_num = len(f"_{self.env.object_scale[self.env.target_idx]}") * (-1)
        obj_name = self.env.obj_path[self.env.target_idx].split('/')[-2][:scale_str_num]
        current_dir = os.path.abspath('')
        data_dir = current_dir + "/data/grasps/acronym"
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

        # get the nearest grasp pose
        cur_ef_pose = self.env._get_ef_pose(mat=True)
        cur_xyz = cur_ef_pose[:, 3:4].reshape(4, )[:3]
        min_dist = 100
        final_pose = None
        for candidate_pose in grasp_arrays:
            can_xyz = candidate_pose[:, 3:4].reshape(4, )[:3]
            xyz_dis = np.linalg.norm(cur_xyz - can_xyz)
            if min_dist > xyz_dis:
                min_dist = xyz_dis
                final_pose = candidate_pose
        return final_pose

    def expert_plan(self, goal_pose, world=False, visual=False):
        if world:
            pos, orn = self.env._get_ef_pose()
            ef_pose_list = [*pos, *orn]
        else:
            ef_pose_list = [0, 0, 0, 0, 0, 0, 1]
        goal_pos = [*goal_pose[:3], *ros_quat(goal_pose[3:])]

        solver = self.planner.plan(ef_pose_list, goal_pos)
        if visual:
            # path_visulization(solver)
            pass
        path = solver.getSolutionPath().getStates()
        planer_path = []
        for i in range(len(path)):
            waypoint = path[i]
            rot = waypoint.rotation()
            action = [waypoint.getX(), waypoint.getY(), waypoint.getZ(), rot.w, rot.x, rot.y, rot.z]
            planer_path.append(action)

        return planer_path

    def policy_move(self):
        done = 0
        reward = 0
        while not done:
            pc_state, target_points, gripper_points = self.get_pc_state()
            joint_state = self.get_joint_degree()
            state = self.policy_id.get_feature_for_policy(pc_state, joint_state)
            dis_embed, conti_latent = self.policy_id.select_action.remote(state)
            conti_action, state_pre = self.policy_id.cvae.decode(state, dis_embed, conti_latent)
            dis_action = self.policy_id.select_discrete_action.remote(dis_embed)

            obs = self.env.step(conti_action, config=True, repeat=200)[0]
            next_pc_state, next_target_points, next_gripper_points = self.get_pc_state()
            next_joint_state = self.get_joint_degree()
            if dis_action == 1:
                """
                try to lift the target object
                """
                cur_joint = np.array(self._panda.getJointStates()[0])
                cur_joint[-1] = 0.8  # close finger
                observations = [self.step(cur_joint, repeat=300, config=True, vis=False)[0]]
                done = 1
                pos, orn = p.getLinkState(self._panda.pandaUid, self._panda.pandaEndEffectorIndex)[4:6]

                for i in range(10):
                    pos = (pos[0], pos[1], pos[2] + 0.03)
                    jointPoses = np.array(p.calculateInverseKinematics(self._panda.pandaUid,
                                                                       self.env._panda.pandaEndEffectorIndex, pos,
                                                                       maxNumIterations=500,
                                                                       residualThreshold=1e-8))
                    jointPoses[6] = 0.8
                    jointPoses = jointPoses[:7].copy()
                    obs = self.step(jointPoses, config=True)[0]
                if self.env.target_lifted():
                    reward = 1
            else:
                reward = 0
            self.buffer_id.add.remote(pc_state, joint_state, conti_action, dis_action, next_pc_state, next_joint_state, reward, done)
        return reward

    def expert_move(self):
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
        grasp_pose = self.get_grasp_pose()
        if grasp_pose is None:
            p.removeConstraint(fixed_joint_constraint)
            self.env.place_back_objects()
            return 0
        # make the gripper retreat a little
        z_axis_direction = grasp_pose[:3, 2]
        grasp_pose[:3, 3] -= 0.02 * z_axis_direction
        grasp_pose = pack_pose(grasp_pose)
        path = self.expert_plan(grasp_pose, world=True)

        for i in range(len(path)):
            # get the state
            pc_state, target_points, gripper_points = self.get_pc_state()
            # print(f"length of target_points: {target_points.shape}")
            if target_points.shape[0] == 0:
                # print(f"target_points.shape[0]: {target_points.shape[0]}")
                # print(f"no target points!!!")
                p.removeConstraint(fixed_joint_constraint)
                self.env.place_back_objects()
                return 0
            joint_state = self.get_joint_degree()
            dis_action = 0
            distance = self.get_distance(target_points, gripper_points)
            # orientation = self.get_orientation(gripper_points, target_points)

            next_pos = path[i]
            jointPoses = self.env._panda.solveInverseKinematics(next_pos[:3], ros_quat(next_pos[3:]))
            jointPoses[6] = 0
            jointPoses = jointPoses[:7].copy()
            obs = self.env.step(jointPoses, config=True, repeat=200)[0]

            # get next state and done and reward
            # con_action = jointPoses[:6]
            con_action = jointPoses[:6] + joint_state  # Literally the next joint state
            next_pc_state, next_target_points, next_gripper_points = self.get_pc_state()
            next_joint_state = self.get_joint_degree()
            next_distance = self.get_distance(next_target_points, next_gripper_points)
            # next_orientation = self.get_orientation(next_gripper_points, next_target_points)

            dis_reward = distance - next_distance
            # ori_reward = next_orientation - orientation

            # print(f"dis_reward: {dis_reward}")
            # print(f"ori_reward: {ori_reward}")
            # print("================")
            reward = dis_reward
            done = 0
            ray.get([self.buffer_id.add.remote(pc_state, joint_state, con_action, dis_action, next_pc_state, next_joint_state, reward, done)])
            
        for i in range(2):
            # get the state
            pc_state, target_points, gripper_points = self.get_pc_state()
            joint_state = self.get_joint_degree()
            dis_action = 0

            self.env.step(action=np.array([0, 0, 0.015, 0, 0, 0]))

            # get next state and done and reward
            next_pc_state, next_target_points, next_gripper_points = self.get_pc_state()
            next_joint_state = self.get_joint_degree()
            con_action = next_joint_state - joint_state
            reward = 0
            done = 0
            if i == 1:
                done = 1
                dis_action = 1
                p.removeConstraint(fixed_joint_constraint)
                reward = self.env.retract()
            ray.get([self.buffer_id.add.remote(pc_state, joint_state, con_action, dis_action, next_pc_state, next_joint_state, reward, done)])
        return reward

    def get_gripper_points(self, target_pointcloud=None):
        inner_point = list(p.getLinkState(self.env._panda.pandaUid, 7)[0])

        gripper_points = np.array([p.getLinkState(self.env._panda.pandaUid, 10)[0],
                                   p.getLinkState(self.env._panda.pandaUid, 15)[0],
                                   inner_point])
        gripper_points = np.hstack((gripper_points, 2*np.ones((gripper_points.shape[0], 1))))
        if target_pointcloud.any() is not None:
            final_pointcloud = np.concatenate((target_pointcloud, gripper_points), axis=0)
        else:
            final_pointcloud = gripper_points
        return final_pointcloud, gripper_points

    def get_world_pointcloud(self, raw_data=False, no_gripper=False):
        obs, joint_pos, camera_info, pose_info = self.env._get_observation(raw_data=raw_data, vis=False, no_gripper=no_gripper)
        pointcloud = obs[0]
        ef_pose = pose_info[1]
        # transform the pointcloud from camera back to world frame
        pointcloud_tar = np.hstack((pointcloud.T, np.ones((len(pointcloud.T), 1)))).T
        target_points = (np.dot(ef_pose, self.env.cam_offset.dot(pointcloud_tar)).T)[:, :3]
        if raw_data is True or raw_data == "obstacle":
            target_points = regularize_pc_point_count(target_points, 1021)
        return target_points

    def get_joint_degree(self):
        con_action = p.getJointStates(self.env._panda.pandaUid, [i for i in range(6)])
        con_action = np.array([i[0] for i in con_action])
        return con_action

    def get_pc_state(self, vis=False):
        """
        The output pc should be (2048, 4)
        The first 1021 points is for obstacle, then the 1024 is for target, 3 for gripper,
        extra channel is 0, 1, 2 respectively.
        """
        obstacle_points = self.get_world_pointcloud(raw_data="obstacle")
        target_points = self.get_world_pointcloud(raw_data=False)
        if self.target_points is not None and self.obstacle_points is not None:
            # print(f"self.target_points: {self.target_points.shape}")
            # print(f"self.obstacle_points: {self.obstacle_points.shape}")
            target_points = regularize_pc_point_count(np.vstack((target_points, self.target_points)), 1024)
            obstacle_points = regularize_pc_point_count(np.vstack((obstacle_points, self.obstacle_points)), 1021)
        self.target_points = target_points
        self.obstacle_points = obstacle_points
        if vis:
            target_o3d_pc = o3d.geometry.PointCloud()
            target_o3d_pc.points = o3d.utility.Vector3dVector(target_points)
            obstacle_o3d_pc = o3d.geometry.PointCloud()
            obstacle_o3d_pc.points = o3d.utility.Vector3dVector(obstacle_points)
            axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([obstacle_o3d_pc]+[target_o3d_pc]+[axis_pcd])

        obstacle_points = np.hstack((obstacle_points, np.zeros((obstacle_points.shape[0], 1))))
        target_points = np.hstack((target_points, np.ones((target_points.shape[0], 1))))
        # print(f"in get_pc_state, target_points.shape: {target_points.shape}")
        scene_points = np.vstack((obstacle_points, target_points))
        all_pc, gripper_points = self.get_gripper_points(scene_points)
        return all_pc, target_points, gripper_points

    def get_distance(self, source_cloud, target_cloud):
        # Expand dimensions to enable broadcasting
        source_cloud = source_cloud[:, np.newaxis, :]  # Shape: (N, 1, 3)
        target_cloud = target_cloud[np.newaxis, :, :]  # Shape: (1, M, 3)

        # Compute the Euclidean distance between all pairs of points
        distance_matrix = np.linalg.norm(source_cloud - target_cloud, axis=-1)  # Shape: (N, M)

        # Find the minimum distance for each source point
        min_distances = np.min(distance_matrix, axis=1)  # Shape: (N,)

        # Find the overall minimum distance
        min_distance = np.min(min_distances)  # Scalar

        return min_distance

    def get_orientation(self, gripper_points, target_points):
        # Extract the individual gripper points
        gripper_left = gripper_points[0]
        gripper_right = gripper_points[1]
        gripper_back = gripper_points[2]

        # Calculate the middle point between the left and right gripper points
        gripper_middle = (gripper_left + gripper_right) / 2.0
        # Compute the vector from the back to the middle of the left and right gripper points
        gripper_vector = gripper_middle - gripper_back
        # Calculate the mean of the target points
        target_mean = np.mean(target_points, axis=0)
        # Compute the vector pointing from the back gripper point to the mean of the target point cloud
        target_vector = target_mean - gripper_back
        # Normalize the vectors
        gripper_vector = gripper_vector / np.linalg.norm(gripper_vector)
        target_vector = target_vector / np.linalg.norm(target_vector)
        # Calculate the dot product between the gripper vector and the target vector
        dot_product = np.dot(gripper_vector, target_vector)
        print(f"dot_product: {dot_product}")
        return dot_product


@ray.remote(num_cpus=1, num_gpus=0.12)
class ActorWrapper012(ActorWrapper):
    pass


@ray.remote(num_cpus=1)
class ReplayMemoryWrapper(ReplayBuffer):
    pass


if __name__ == "__main__":
    ray.init()

    buffer_id = ReplayMemoryWrapper.remote(state_dim=2048, con_action_dim=64)
    print(type(buffer_id))
    actors = [ActorWrapper012.remote(buffer_id, 0) for _ in range(3)]
    rewards = [actor.rollout_once.remote() for actor in actors]

    for reward in rewards:
        print(ray.get(reward))

    size = ray.get(buffer_id.get_size.remote())
    ray.get(buffer_id.save_data.remote("RL_ws/offline_data/offline_data.npz"))
    print(size)
