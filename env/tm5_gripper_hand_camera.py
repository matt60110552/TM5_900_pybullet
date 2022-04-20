# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import pybullet as p
import numpy as np
import IPython
import os


class TM5:
    def __init__(self, stepsize=1e-3, realtime=0, init_joints=None, base_shift=[0, 0, 0]):
        self.t = 0.0
        self.stepsize = stepsize
        self.realtime = realtime
        self.control_mode = "position"

        self.position_control_gain_p = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        self.position_control_gain_d = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        f_max = 250
        self.max_torque = [f_max, f_max, f_max, f_max, f_max, f_max, 100, 100]

        # connect pybullet
        p.setRealTimeSimulation(self.realtime)

        # load models
        current_dir = os.path.dirname(os.path.abspath(__file__))
        p.setAdditionalSearchPath(current_dir + "/models")
        print(current_dir + "/models")
        self.robot = p.loadURDF("tm5_900/tm5_900_with_gripper.urdf",
                                useFixedBase=True,
                                flags=p.URDF_USE_SELF_COLLISION)
        self._base_position = [-0.05 - base_shift[0], 0.0 - base_shift[1], -0.65 - base_shift[2]]
        self.pandaUid = self.robot

        # robot parameters
        self.dof = p.getNumJoints(self.robot)

        # To control the gripper
        mimic_parent_name = 'finger_joint'
        mimic_children_names = {'right_outer_knuckle_joint': 1,
                                'left_inner_knuckle_joint': 1,
                                'right_inner_knuckle_joint': 1,
                                'left_inner_finger_joint': -1,
                                'right_inner_finger_joint': -1}
        mimic_parent_id = []
        mimic_child_multiplier = {}
        for i in range(p.getNumJoints(self.robot)):
            inf = p.getJointInfo(self.robot, i)
            name = inf[1].decode('utf-8')
            if name == mimic_parent_name:
                mimic_parent_id.append(inf[0])
            if name in mimic_children_names:
                mimic_child_multiplier[inf[0]] = mimic_children_names[name]
            if inf[2] != p.JOINT_FIXED:
                p.setJointMotorControl2(self.robot, inf[0], p.VELOCITY_CONTROL, targetVelocity=0, force=0)

        self.mimic_parent_id = mimic_parent_id[0]

        for joint_id, multiplier in mimic_child_multiplier.items():
            c = p.createConstraint(self.robot, self.mimic_parent_id,
                                   self.robot, joint_id,
                                   jointType=p.JOINT_GEAR,
                                   jointAxis=[0, 1, 0],
                                   parentFramePosition=[0, 0, 0],
                                   childFramePosition=[0, 0, 0])
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)

        p.setCollisionFilterPair(self.robot, self.robot, 10, 12, 0)
        p.setCollisionFilterPair(self.robot, self.robot, 15, 17, 0)

        self.joints = []
        self.q_min = []
        self.q_max = []
        self.target_pos = []
        self.target_torque = []
        self.pandaEndEffectorIndex = 7
        self._joint_min_limit = np.array([-4.712385, -3.14159, -2.70526, -3.14159, -3.14159, -4.712385, 0, 0])
        self._joint_max_limit = np.array([4.712385, 3.14159, 2.70526,  3.14159,  3.14159,  4.712385, 0, 0.8])

        for j in range(self.dof):
            p.changeDynamics(self.robot, j, linearDamping=0, angularDamping=0)
            joint_info = p.getJointInfo(self.robot, j)
            if j in range(1, 9):
                self.joints.append(j)
                self.q_min.append(joint_info[8])
                self.q_max.append(joint_info[9])
                self.target_pos.append((self.q_min[j-1] + self.q_max[j-1])/2.0)
                self.target_torque.append(0.)
        self.reset(init_joints)

    def reset(self, joints=None):
        self.t = 0.0
        self.control_mode = "position"
        p.resetBasePositionAndOrientation(self.pandaUid, self._base_position,
                                          [0.000000, 0.000000, 0.000000, 1.000000])
        if joints is None:
            self.target_pos = [
                    0.2, -1, 2, 0, 1.571, 0.0, 0.0, 0.0]

            self.target_pos = self.standardize(self.target_pos)
            for j in range(1, 9):
                self.target_torque[j-1] = 0.
                p.resetJointState(self.robot, j, targetValue=self.target_pos[j-1])

        else:
            joints = self.standardize(joints)
            for j in range(1, 9):
                self.target_pos[j-1] = joints[j-1]
                self.target_torque[j-1] = 0.
                p.resetJointState(self.robot, j, targetValue=self.target_pos[j-1])
        self.resetController()
        self.setTargetPositions(self.target_pos)

    def step(self):
        self.t += self.stepsize
        p.stepSimulation()

    def resetController(self):
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=self.joints,
                                    controlMode=p.VELOCITY_CONTROL,
                                    forces=[0. for i in range(1, 9)])

    def standardize(self, target_pos):
        if len(target_pos) == 7:
            if type(target_pos) == list:
                target_pos.insert(6, 0)
            else:
                target_pos = np.insert(target_pos, 6, 0)

        target_pos = np.array(target_pos)

        target_pos = np.minimum(np.maximum(target_pos, self._joint_min_limit), self._joint_max_limit)
        return target_pos

    def setTargetPositions(self, target_pos):
        self.target_pos = self.standardize(target_pos)
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=self.joints,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=self.target_pos,
                                    forces=self.max_torque,
                                    positionGains=self.position_control_gain_p,
                                    velocityGains=self.position_control_gain_d)

    def getJointStates(self):
        joint_states = p.getJointStates(self.robot, self.joints)

        joint_pos = [x[0] for x in joint_states]
        joint_vel = [x[1] for x in joint_states]

        del joint_pos[6]
        del joint_vel[6]

        return joint_pos, joint_vel

    def solveInverseKinematics(self, pos, ori):
        return list(p.calculateInverseKinematics(self.robot,
                    7, pos, ori,
                    maxNumIterations=500,
                    residualThreshold=1e-8))


if __name__ == "__main__":
    robot = TM5(realtime=1)
    while True:
        pass
