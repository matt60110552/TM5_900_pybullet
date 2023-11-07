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
from scipy.spatial import cKDTree
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils.utils import *
from utils.planner import GraspPlanner
from utils.grasp_checker import ValidGraspChecker
from env.ycb_scene import SimulatedYCBEnv
from utils.planner import GraspPlanner
from replay_buffer_plain import ReplayBuffer
from visdom import Visdom
from pointmlp import farthest_point_sample, index_points


class ActorWrapper(object):
    """
    wrapper testing, use ray to create multiple pybullet
    """
    def __init__(self, online_buffer_id, buffer_id, policy_id, renders=False, scene_level=True):
        # from env.ycb_scene import SimulatedYCBEnv
        # file = os.path.join("object_index", 'acronym_90.json')
        file = os.path.join("object_index", 'small_objects.json')
        with open(file) as f: file_dir = json.load(f)
        file_dir = file_dir['train']
        # file_dir = file_dir['test']
        file_dir = [f[:-5] for f in file_dir]
        test_file_dir = list(set(file_dir))
        test_file_dir = random.sample(test_file_dir, 15)
        self.env = SimulatedYCBEnv(renders=renders)
        self.env._load_index_objs(test_file_dir)
        # self.env.reset(save=False, enforce_face_target=True)
        self.env.reset(save=False, enforce_face_target=True, furniture="cabinet")
        self.grasp_checker = ValidGraspChecker(self.env)
        self.planner = GraspPlanner()
        self.buffer_id = buffer_id
        self.policy_id = policy_id
        self.online_buffer_id = online_buffer_id
        self.scene_level = scene_level
        self.target_points = None   # This is for merging point-cloud from different time
        self.obstacle_points = None
        self.vis = Visdom(port=8097)
        self.win_id = self.vis.image(np.zeros([3, 224, 224]))
        # disable the collision between the basse of TM5 and plane        
        p.setCollisionFilterPair(self.env.plane_id, self.env._panda.pandaUid, -1, 0, enableCollision=False)

    def rollout_once(self, mode="expert", explore_ratio=0, vis=False):
        start = time.time()
        self.env.reset(save=False, enforce_face_target=False, reset_free=True)
        if mode == "expert":
            rewards = self.expert_move(vis=False)
        elif mode == "onpolicy":
            """
            Use actor to react to the state
            """
            rewards = self.policy_move(explore_ratio=explore_ratio)
            # rewards = self.policy_move(vis=True)
        elif mode == "both":
            rewards = self.policy_move(explore_ratio=explore_ratio) if random.random() < 0.5 else self.expert_move()

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
        """
        This part is for filter the grasp pose that is too hard for the arm to approaching, like too vertical or too side
        """
        if len(grasp_arrays) == 0:
            return None
        obj_center_2d = np.append(np.array(obj_pos[0][:2]), [0], axis=0)
        grasp_vectors = grasp_arrays[:, :3, :3] @ np.array([0, 0, 1])
        grasp_idx = [np.dot(vector, obj_center_2d) > 0.5 for vector in grasp_vectors]
        grasp_arrays =  grasp_arrays[grasp_idx]
        # print(f"grasp_vector: {grasp_vector}")
        # for idx in range(5):
        #     print(f"dot product: {np.dot(grasp_vector[idx], obj_center_2d)}")
        """
        End of additional part
        """



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
        if solver is None:
            return None
        path = solver.getSolutionPath().getStates()
        planer_path = []
        for i in range(len(path)):
            waypoint = path[i]
            rot = waypoint.rotation()
            action = [waypoint.getX(), waypoint.getY(), waypoint.getZ(), rot.w, rot.x, rot.y, rot.z]
            planer_path.append(action)

        return planer_path

    def policy_move(self, vis=False, explore_ratio=0):
        # data_list is for store data, unlike expert_move, here we store all the procees, including fail one
        data_list = []
        self.target_points = None
        self.obstacle_points = None
        done = 0
        reward = 0
        movement_num = 0
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
        while not done:
            movement_num += 1
            reward = 0
            # get pointcloud
            (pc_state, target_points, manipulator_points,
             obstacle_points) = self.get_pc_state(frame="base", obs_reserve=(movement_num > 3))

            if target_points is None:
                # print(f"target_points.shape[0]: {target_points.shape[0]}")
                # print(f"no target points!!!")
                print(f"targe_name: {self.env.target_name}")
                p.removeConstraint(fixed_joint_constraint)
                self.env.place_back_objects()
                return (1, 0)

            joint_state = self.get_joint_degree()
            goal_pos = self.get_target_center(target_points)
            # check if the object fall into other layer
            if goal_pos[2] > 0.75 or goal_pos[2] < 0.5:
                data_list = []
                break
            cur_gripper_pose = self.get_cur_gripper_pose()
            scene_point = np.concatenate((obstacle_points, manipulator_points),axis=0)
            # scene_point = pc_state
            
            # get action from policy
            conti_action = ray.get([self.policy_id.select_action.remote(goal_pos, joint_state, cur_gripper_pose,
                                                                        scene_point, explore_ratio)])[0]
            conti_action = np.append(conti_action, 0)
            obs = self.env.step(conti_action, delta=True, config=True, repeat=300)[0]

            """visdom part visualize the image of the process"""
            self.vis.image(obs[0][1][:3].transpose(0, 2, 1), win=self.win_id, opts={"title": "policy"})

            (next_pc_state, next_target_points, next_manipulator_points,
             next_obstacle_points) = self.get_pc_state(frame="base", obs_reserve=(movement_num>3))

            next_joint_state = self.get_joint_degree()
            next_gripper_pose = self.get_cur_gripper_pose()
            next_scene_point = np.concatenate((next_obstacle_points, next_manipulator_points),axis=0)
            # next_scene_point = next_pc_state
            
            # Bonus reward part
            distance = self.get_distance(next_manipulator_points[-3:, :3], goal_pos)
            orientation = self.get_orientation(next_manipulator_points[-3:, :3], goal_pos)
            bonus_reward = self.approaching_reward(distance, orientation)
            if bonus_reward == 1:
                done = 1
            reward += bonus_reward

            if vis:
                vis_pc = next_pc_state[:, :3]
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(vis_pc)
                axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                o3d.visualization.draw_geometries([point_cloud] + [axis_pcd])

            if self.collision_check():
                done = 1
                reward = 0

            # reach predefined step_num 
            if movement_num == 25:
                done = 1

            conti_action = next_joint_state - joint_state  # This is for the situation that arm doesn't reach teh goal position
            data_list.append((goal_pos, joint_state, cur_gripper_pose, scene_point, conti_action[:6],
                              next_joint_state, next_gripper_pose, next_scene_point, reward, done))


        if reward == 1:
            for i in range(len(data_list)):
                (goal_pos, joint_state, cur_gripper_pose, scene_point, conti_action[:6],
                next_joint_state, next_gripper_pose, next_scene_point,
                reward, done) = data_list[i]
                for i in range(3):
                    ray.get([self.online_buffer_id.add.remote(goal_pos, joint_state, cur_gripper_pose, scene_point, conti_action[:6],
                                                              next_joint_state, next_gripper_pose, next_scene_point, reward, done)])
        else:
            for i in range(len(data_list)):
                (goal_pos, joint_state, cur_gripper_pose, scene_point, conti_action[:6],
                next_joint_state, next_gripper_pose, next_scene_point,
                reward, done) = data_list[i]
                ray.get([self.online_buffer_id.add.remote(goal_pos, joint_state, cur_gripper_pose, scene_point, conti_action[:6],
                                                          next_joint_state, next_gripper_pose, next_scene_point, reward, done)])
            # ray.get([self.online_buffer_id.add.remote(goal_pos, joint_state, cur_gripper_pose, conti_action[:6],
            #                                           next_joint_state, next_gripper_pose, reward, done)])
        p.removeConstraint(fixed_joint_constraint)
        self.env.place_back_objects()

        # return the value according to the last reward in the episode
        if reward == 1:
            return (1, 1)
        else:
            return (1, -0.1)

    def expert_move(self, vis=False):
        # data_list is for store data, in order to save successful grasp process only
        data_list = []
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
            return (0, 0)
        # make the gripper retreat a little
        z_axis_direction = grasp_pose[:3, 2]
        grasp_pose[:3, 3] -= 0.02 * z_axis_direction
        grasp_pose = pack_pose(grasp_pose)
        path = self.expert_plan(grasp_pose, world=True)

        reward = 0
        dis_action = -1
        done = 0
        if path is None:
            return (0, 0)
        for i in range(len(path)):
            reward = 0
            # get the state
            (pc_state, target_points, manipulator_points,
             obstacle_points) = self.get_pc_state(frame="base", obs_reserve=(i>3))
            if target_points is None:
                p.removeConstraint(fixed_joint_constraint)
                self.env.place_back_objects()
                return (0, 0)
            
            joint_state = self.get_joint_degree()
            goal_pos = self.get_target_center(target_points)
            # check if the object fall into other layer
            if goal_pos[2] > 0.7 or goal_pos[2] < 0.5:
                data_list = []
                break
            cur_gripper_pose = self.get_cur_gripper_pose()
            scene_point = np.concatenate((obstacle_points, manipulator_points),axis=0)
            # scene_point = pc_state

            next_pos = path[i]
            jointPoses = self.env._panda.solveInverseKinematics(next_pos[:3], ros_quat(next_pos[3:]))
            jointPoses[6] = 0
            jointPoses = jointPoses[:7].copy()

            obs = self.env.step(jointPoses, config=True, repeat=250)[0]

            # get next state and done and reward
            (next_pc_state, next_target_points, next_manipulator_points,
             next_obstacle_points) = self.get_pc_state(frame="base", obs_reserve=(i>3))
            next_joint_state = self.get_joint_degree()
            next_gripper_pose = self.get_cur_gripper_pose()
            next_scene_point = np.concatenate((next_obstacle_points, next_manipulator_points),axis=0)
            # next_scene_point = next_pc_state
            conti_action = next_joint_state - joint_state

            if vis:
                point_cloud = np.array(next_scene_point[:, :3])  # Convert to NumPy array
                vis_pc = point_cloud[:, :3]
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(vis_pc)
                axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                o3d.visualization.draw_geometries([point_cloud] + [axis_pcd])

           
            distance = self.get_distance(next_manipulator_points[-3:, :3], goal_pos)
            orientation = self.get_orientation(next_manipulator_points[-3:, :3], goal_pos)
            bonus_reward = self.approaching_reward(distance, orientation)
            if bonus_reward == 1:
                done = 1
            reward += bonus_reward
            
            if self.collision_check():
                done = 1
                reward = 0

            data_list.append((goal_pos, joint_state, cur_gripper_pose, scene_point, conti_action[:6],
                              next_joint_state, next_gripper_pose, next_scene_point, reward, done))

            """visdom part visualize the image of the process"""
            self.vis.image(obs[0][1][:3].transpose(0, 2, 1), win=self.win_id, opts={"title": "expert"})

            if done == 1:
                break

            # check for pointcloud
            # vis_pc = pc_state[:, :3]
            # point_cloud = o3d.geometry.PointCloud()
            # point_cloud.points = o3d.utility.Vector3dVector(vis_pc)
            # axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            # o3d.visualization.draw_geometries([point_cloud] + [axis_pcd])
        print(f"reward: {reward}")
        if reward == 1:
            for i in range(len(data_list)):
                (goal_pos, joint_state, cur_gripper_pose, scene_point, conti_action[:6],
                next_joint_state, next_gripper_pose, next_scene_point, reward, done) = data_list[i]

                for i in range(3):
                    ray.get([self.buffer_id.add.remote(goal_pos, joint_state, cur_gripper_pose, scene_point, conti_action[:6],
                                                       next_joint_state, next_gripper_pose, next_scene_point, reward, done)])
        else:
            for i in range(len(data_list)):
                (goal_pos, joint_state, cur_gripper_pose, scene_point, conti_action[:6],
                next_joint_state, next_gripper_pose, next_scene_point, reward, done) = data_list[i]
                ray.get([self.buffer_id.add.remote(goal_pos, joint_state, cur_gripper_pose, scene_point, conti_action[:6],
                                                   next_joint_state, next_gripper_pose, next_scene_point, reward, done)])
            # ray.get([self.buffer_id.add.remote(goal_pos, joint_state, cur_gripper_pose, conti_action[:6],
            #                                    next_joint_state, next_gripper_pose, reward, done)])
        p.removeConstraint(fixed_joint_constraint)
        self.env.place_back_objects()
        return (0, reward)

    def get_gripper_points(self, target_pointcloud=None):
        inner_point = list(p.getLinkState(self.env._panda.pandaUid, 7)[0])

        gripper_points = np.array([p.getLinkState(self.env._panda.pandaUid, 10)[0],
                                   p.getLinkState(self.env._panda.pandaUid, 15)[0],
                                   inner_point])
        gripper_points = np.hstack((gripper_points, 2*np.ones((gripper_points.shape[0], 1))))

        arm_points = np.array([p.getLinkState(self.env._panda.pandaUid, i)[0] for i in range(7)])
        arm_points = np.hstack((arm_points, 2 * np.ones((arm_points.shape[0], 1))))

        manipulator_points = np.concatenate((arm_points, gripper_points), axis=0)

        if target_pointcloud.any() is not None:
            final_pointcloud = np.concatenate((target_pointcloud, manipulator_points), axis=0)
        else:
            final_pointcloud = manipulator_points
        return final_pointcloud, manipulator_points

    def get_world_pointcloud(self, raw_data=False, no_gripper=False, object_level=False):
        obs, joint_pos, camera_info, pose_info = self.env._get_observation(raw_data=raw_data, vis=False, no_gripper=no_gripper)
        pointcloud = obs[0]
        ef_pose = pose_info[1]
        
        # # transform the pointcloud from camera back to world frame
        pointcloud_tar = np.hstack((pointcloud.T, np.ones((len(pointcloud.T), 1)))).T
        target_points = (np.dot(ef_pose, self.env.cam_offset.dot(pointcloud_tar)).T)[:, :3]

        if len(target_points) == 0:
            return None
        if raw_data is True or raw_data == "obstacle" or object_level:
            target_points = regularize_pc_point_count(target_points, 502)
        else:
            target_points = regularize_pc_point_count(target_points, 512)
        return target_points

    def get_joint_degree(self):
        con_action = p.getJointStates(self.env._panda.pandaUid, [i for i in range(1, 7)])
        con_action = np.array([i[0] for i in con_action])
        return con_action

    def get_pc_state(self, vis=False, frame="camera", obs_reserve=False):
        """
        The output pc should be (2048, 4)
        The first 1021 points is for obstacle, then the 1024 is for target, 3 for gripper,
        extra channel is 0, 1, 2 respectively.
        """
        obstacle_points = self.get_world_pointcloud(raw_data="obstacle")
        target_points = self.get_world_pointcloud(raw_data=False, object_level=not self.scene_level)

        # deal with target points, combine them with previous points if exist, or overwrite it with preious if None
        if self.target_points is not None:
            if target_points is None:
                target_points = self.target_points
            else:
                if self.scene_level:
                    # combine two pointcloud part, first convert them to tensor
                    target_points_tensor = torch.from_numpy(target_points)
                    self_target_points_tensor = torch.from_numpy(self.target_points)
                    combined_target_points = torch.cat((target_points_tensor, self_target_points_tensor), dim=0).unsqueeze(0)
                    index = farthest_point_sample(combined_target_points, 512)
                    target_points = index_points(combined_target_points, index).squeeze().detach().numpy()
                else:
                    # combine two pointcloud part, first convert them to tensor
                    target_points_tensor = torch.from_numpy(target_points)
                    self_target_points_tensor = torch.from_numpy(self.target_points)
                    combined_target_points = torch.cat((target_points_tensor, self_target_points_tensor), dim=0).unsqueeze(0)
                    index = farthest_point_sample(combined_target_points, 502)
                    target_points = index_points(combined_target_points, index).squeeze().detach().numpy()
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
                    index = farthest_point_sample(combined_obstacle_points, 502)
                    obstacle_points = index_points(combined_obstacle_points, index).squeeze().detach().numpy()

        self.target_points = target_points
        self.obstacle_points = obstacle_points

        if target_points is None or obstacle_points is None:
            return None, None, None, None
        obstacle_points = np.hstack((obstacle_points, np.zeros((obstacle_points.shape[0], 1))))
        target_points = np.hstack((target_points, np.ones((target_points.shape[0], 1))))
        scene_points = np.vstack((obstacle_points, target_points))
        if self.scene_level:
            all_pc, manipulator_points = self.get_gripper_points(scene_points)
        else:
            all_pc, manipulator_points = self.get_gripper_points(target_points)
        # if target_points is None or obstacle_points is None:
        #     return None, None, None, None

        # # transform back to camera frame
        if frame == "camera":
            all_pc = self.base2camera(all_pc)
            target_points = self.base2camera(target_points)
            manipulator_points = self.base2camera(manipulator_points)

        if vis:
            all_o3d_pc = o3d.geometry.PointCloud()
            all_o3d_pc.points = o3d.utility.Vector3dVector(all_pc[:, :3])
            target_o3d_pc = o3d.geometry.PointCloud()
            target_o3d_pc.points = o3d.utility.Vector3dVector(target_points[:, :3])
            axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            # o3d.visualization.draw_geometries([all_o3d_pc]+[target_o3d_pc]+[axis_pcd])
            o3d.visualization.draw_geometries([all_o3d_pc]+[axis_pcd])

        return all_pc, target_points, manipulator_points, obstacle_points

    def get_distance(self, source_cloud, target_center):
        # source_cloud is for gripper and target_center is for target object
        
        source_center_point = np.mean(source_cloud, axis=0)
        distance = np.linalg.norm(source_center_point - target_center)
        
        return distance

    def get_orientation(self, gripper_points, target_center):
        # Extract the individual gripper points
        gripper_left = gripper_points[-3, :3]
        gripper_right = gripper_points[-2, :3]
        gripper_back = gripper_points[-1, :3]

        # Calculate the middle point between the left and right gripper points
        gripper_middle = (gripper_left + gripper_right) / 2.0
        # Compute the vector from the back to the middle of the left and right gripper points
        gripper_vector = gripper_middle - gripper_back

        # Compute the vector pointing from the back gripper point to the mean of the target point cloud
        target_vector = target_center - gripper_back
        # Normalize the vectors
        gripper_vector = gripper_vector / np.linalg.norm(gripper_vector)
        target_vector = target_vector / np.linalg.norm(target_vector)
        # Calculate the dot product between the gripper vector and the target vector
        dot_product = np.dot(gripper_vector, target_vector)
        # print(f"dot_product: {dot_product}")
        return dot_product

    def collision_check(self):
        # self-collision part
        all_collision = p.getContactPoints()
        for x in all_collision:
            target_id, source_id = x[1:3]
            if target_id == self.env._panda.pandaUid and source_id == self.env._panda.pandaUid:
                return True
        # plane-collision part
        for x in all_collision:
            target_id, source_id = x[1:3]
            collision_set = set((target_id, source_id))
            if self.env._panda.pandaUid in collision_set and self.env.plane_id in collision_set:
                return True
        # furniture-collision part
        for x in all_collision:
            target_id, source_id = x[1:3]
            collision_set = set((target_id, source_id))
            if self.env._panda.pandaUid in collision_set and self.env.furniture_id in collision_set:
                return True

        return False
    
    def base2camera(self, pointcloud):
        inverse_camera_matrix = np.linalg.inv(self.env.cam_offset)
        inverse_ef_pose_matrix = np.linalg.inv(self.env._get_ef_pose('mat'))
        original_fourth_column = pointcloud[:, 3].copy()
        pointcloud[:, 3] = 1
        pointcloud_ef_pose = np.dot(inverse_ef_pose_matrix, pointcloud.T).T[:, :3]
        pointcloud_camera = np.dot(inverse_camera_matrix, np.hstack((pointcloud_ef_pose, np.ones((pointcloud_ef_pose.shape[0], 1)))).T).T
        pointcloud_camera[:, 3] = original_fourth_column
        return pointcloud_camera
    
    def approaching_reward(self, distance, orientation):
        if distance <= 0.22 and orientation > 0.8:
            return 1
        elif distance <= 0.25 and orientation > 0.8:
            return 0.2
        elif distance <= 0.3 and orientation > 0.8:
            return 0.1
        else:
            return 0
    
    def get_target_center(self, taraget_pointcloud):
        taraget_pointcloud_np = np.array(taraget_pointcloud[:, :3])
        # Assuming you have calculated the minimum and maximum values as shown in the previous example
        min_x, min_y, min_z = np.min(taraget_pointcloud_np, axis=0)
        max_x, max_y, max_z = np.max(taraget_pointcloud_np, axis=0)

        # Calculate the center of the bounding box
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        center_z = (min_z + max_z) / 2

        # The variables center_x, center_y, and center_z now contain the coordinates of the center of the bounding box.
        center = (center_x, center_y, center_z)
        return center

    def get_cur_gripper_pose(self):
        # return xyz and euler orientation 
        pos, orn = p.getLinkState(self.env._panda.pandaUid, self.env._panda.pandaEndEffectorIndex)[4:6] # orn here is xyzw
        cur_gripper_pose = np.array(list(pos) + list(quat2euler(tf_quat(orn))))  # x, y, z, euler
        return cur_gripper_pose

    
    def euler2vector(self, euler):
        # Convert Euler angles to a direction vector
        # Assuming XYZ order (roll-pitch-yaw)
        direction_vector = np.array([
            np.cos(yaw) * np.cos(pitch),
            np.sin(yaw) * np.cos(pitch),
            -np.sin(pitch)
        ])

        # Normalize the direction vector to make it a unit vector
        direction_vector /= np.linalg.norm(direction_vector)

        print("Direction Vector:", direction_vector)


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
