import pybullet as p
import numpy as np
import sys
import os
import json
import open3d as o3d
import matplotlib.pyplot as plt
import time

sys.path.append("../")
from utils.utils import *
from env.ycb_scene import SimulatedYCBEnv
from utils.planner import GraspPlanner
from utils.grasp_checker import ValidGraspChecker

import os.path as osp
# import pybullet as p
import math
# import sys
import pybullet_data
sys.path.insert(0, osp.join(osp.dirname(osp.abspath("file")), '../pybullet_ompl'))
import pb_ompl
import pb_ompl_utils
from itertools import product




class ActorWrapper(object):
    def __init__(self, renders=True):
        '''
        get data file name in json file and load mesh in pybullet
        then reset robot and object position
        '''

        file = os.path.join("../object_index", 'small_objects.json')
        with open(file) as f: file_dir = json.load(f)
        file_dir = file_dir['train']
        file_dir = [f[:-5] for f in file_dir]
        test_file_dir = list(set(file_dir))
        test_file_dir = random.sample(test_file_dir, 15)

        self.env = SimulatedYCBEnv(renders=renders)
        self.env._load_index_objs(test_file_dir)
        state = self.env.reset(save=False, enforce_face_target=True, furniture="cabinet")
        self.grasp_checker = ValidGraspChecker(self.env)
        self.planner = GraspPlanner()

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # self.joint_idx = [1, 2, 3, 4, 5, 6]
        # self.robot = pb_ompl.PbOMPLRobot(self.env._panda.pandaUid, self.joint_idx)
        # self.obstacles = []
        # self.obstacles.append(self.env.furniture_id)
        # self.pb_ompl_interface = pb_ompl.PbOMPL(self.robot, self.obstacles)
        # self.pb_ompl_interface.set_planner("BITstar")
        # self.setup_collision_detection(self.robot, self.obstacles)


    def simple_demonstrate(self):
        self.env.reset(save=False, enforce_face_target=False, reset_free=True)
        start_state = np.array(self.env._panda.getJointStates()[0])[:6]
        print(f"start_state: {start_state}")
        # self.robot.set_state(start_state)
        # self.robot.reset()
        self.get_grasp_pose()
        # goal = [0.25, -0.35, 1.75, -0.6, 1.571, 0.0]
        # res, path = self.pb_ompl_interface.plan(goal)
        # if res:
        #     self.pb_ompl_interface.execute(path)


    def get_grasp_pose(self):
        
        '''
        Take pre-define grasp dataset of target object as an example.
        Load npy file by object names.
        '''

        scale_str_num = len(f"_{self.env.object_scale[self.env.target_idx]}") * (-1)
        obj_name = self.env.obj_path[self.env.target_idx].split('/')[-2][:scale_str_num]
        current_dir = os.path.abspath('')
        data_dir = current_dir + "/../data/grasps/acronym"
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

        # # get the nearest grasp pose
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

        if final_pose is None:
            return
        pos_orn = pack_pose(final_pose)
        goal_joint = p.calculateInverseKinematics(self.env._panda.pandaUid,
                                    self.env._panda.pandaEndEffectorIndex,
                                    pos_orn[:3],
                                    ros_quat(pos_orn[3:]),
                                    maxNumIterations=500,
                                    residualThreshold=1e-8)
        
        self.joint_idx = [1, 2, 3, 4, 5, 6]
        self.robot = pb_ompl.PbOMPLRobot(self.env._panda.pandaUid, self.joint_idx)
        self.robot.reset()
        self.obstacles = []
        self.obstacles.append(self.env.furniture_id)
        self.obstacles.append(self.env.plane_id)
        self.pb_ompl_interface = pb_ompl.PbOMPL(self.robot, self.obstacles)
        self.pb_ompl_interface.set_planner("BITstar")
        self.setup_collision_detection(self.robot, self.obstacles)
        
        res, path = self.pb_ompl_interface.plan(goal_joint[:6])


        if len(grasp_arrays) == 0:
            return None

        return grasp_arrays


    def is_state_valid(self, state):
        # satisfy bounds TODO
        # Should be unecessary if joint bounds is properly set

        # check self-collision
        self.robot.set_state(self.state_to_list(state))
        for link1, link2 in self.check_link_pairs:
            if pb_ompl_utils.pairwise_link_collision(self.env._panda.pandaUid, link1, self.env._panda.pandaUid, link2):
                # print(get_body_name(body), get_link_name(body, link1), get_link_name(body, link2))
                return False

        # check collision against environment
        for body1, body2 in self.check_body_pairs:
            if pb_ompl_utils.pairwise_collision(body1, body2):
                # print('body collision', body1, body2)
                # print(get_body_name(body1), get_body_name(body2))
                return False
        return True

    def setup_collision_detection(self, robot, obstacles, self_collisions = True, allow_collision_links = []):
        self.check_link_pairs = pb_ompl_utils.get_self_link_pairs(robot.id, robot.joint_idx) if self_collisions else []
        # print(f"self.check_link_pairs: {self.check_link_pairs}")
        moving_links = frozenset(
            [item for item in pb_ompl_utils.get_moving_links(robot.id, robot.joint_idx) if not item in allow_collision_links])
        moving_bodies = [(robot.id, moving_links)]
        self.check_body_pairs = list(product(moving_bodies, obstacles))


    def final_pose_collision_check(self, grasp_poses):
        valid_pose = []
        for pose in grasp_poses:
            pose_packed = pack_pose(pose)
            jointPoses = self.env._panda.solveInverseKinematics(pose_packed[:3], ros_quat(pose_packed[3:]))  # fed xyzw for orn

            if self.is_state_valid(jointPoses):
                valid_pose.append(jointPoses)
            # pb_ompl_utils.pairwise_collision(self.env._panda.pandaUid, )
        return valid_pose

    def _set_joint_positions(self, joints, positions):
        for joint, value in zip(joints, positions):
            p.resetJointState(self.env._panda.pandaUid, joint, value, targetVelocity=0)


    def state_to_list(self, state):
        return [state[i] for i in range(self.robot.num_dim)]

if __name__ == "__main__":
    actor = ActorWrapper(True)
    for i in range(10):
        actor.simple_demonstrate()  # This line is for showing the environment with fixed path



