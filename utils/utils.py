# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import numpy as np
import os
import sys
from transforms3d.quaternions import *
from transforms3d.euler import *
from transforms3d.axangles import *
import random

from scipy import interpolate
import scipy.io as sio
import IPython
from torch import nn
from torch import optim

import torch.nn.functional as F
import cv2

import matplotlib.pyplot as plt
import tabulate
import torch

from easydict import EasyDict as edict
import GPUtil
import open3d as o3d
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.sum_2 = 0
        self.count_2 = 0
        self.means = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.sum_2 += val * n
        self.count_2 += n

    def set_mean(self):
        self.means.append(self.sum_2 / self.count_2)
        self.sum_2 = 0
        self.count_2 = 0

    def std(self):
        return np.std(np.array(self.means) + 1e-4)

    def __repr__(self):
        return "{:.3f} ({:.3f})".format(self.val, self.avg)


def normalize(v, axis=None, eps=1e-10):
    """L2 Normalize along specified axes."""
    return v / max(np.linalg.norm(v, axis=axis, keepdims=True), eps)


def inv_lookat(eye, target=[0, 0, 0], up=[0, 1, 0]):
    """Generate LookAt matrix."""
    eye = np.float32(eye)
    forward = normalize(target - eye)
    side = normalize(np.cross(forward, up))
    up = np.cross(side, forward)
    R = np.stack([side, up, -forward], axis=-1)
    return R


def rand_sample_joint(env, init_joints=None, near=0.2, far=0.5):
    """
    randomize initial joint configuration
    """
    init_joints_ = env.randomize_arm_init(near, far)
    init_joints = init_joints_ if init_joints_ is not None else init_joints
    return init_joints


def check_scene(env, state, start_rot, object_performance=None, scene_name=None,
                init_dist_low=0.2, init_dist_high=0.5, run_iter=0):
    """
    check if a scene is valid by its distance, view, hand direction, target object state, and object counts
    """
    MAX_TEST_PER_OBJ = 10
    dist = np.linalg.norm(env._get_target_relative_pose('tcp')[:3, 3])
    dist_flag = dist > init_dist_low and dist < init_dist_high
    pt_flag = state[0][0].shape[1] > 100
    z = start_rot[:3, 0] / np.linalg.norm(start_rot[:3, 0])
    hand_dir_flag = z[-1] > -0.3
    target_obj_flag = env.target_name != 'noexists'
    if object_performance is None:
        full_flag = True
    else:
        full_flag = env.target_name not in object_performance or object_performance[env.target_name][0].count < (run_iter + 1) * MAX_TEST_PER_OBJ
    name_flag = 'pitcher' not in env.target_name
    return full_flag and target_obj_flag and pt_flag and name_flag


def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


def process_image_input(state):
    state[:, :3] *= 255
    if state.shape[1] >= 4:
        state[:, 3] *= 5000
    if state.shape[1] == 5:
        state[:, -1][state[:, -1] == -1] = 50
    return state.astype(np.uint16)


def check_ngc():
    GPUs = GPUtil.getGPUs()
    gpu_limit = max([GPU.memoryTotal for GPU in GPUs])
    return (gpu_limit > 14000)


def process_image_output(sample):
    sample = sample.astype(np.float32).copy()
    n = len(sample)
    if len(sample.shape) <= 2:
        return sample

    sample[:, :3] /= 255.0
    if sample.shape[0] >= 4:
        sample[:, 3] /= 5000
    sample[:, -1] = sample[:, -1] != 0
    return sample


def get_valid_index(arr, index):
    return arr[min(len(arr) - 1, index)]


def deg2rad(deg):
    if type(deg) is list:
        return [x/180.0*np.pi for x in deg]
    return deg/180.0*np.pi


def rad2deg(rad):
    if type(rad) is list:
        return [x/np.pi*180 for x in rad]
    return rad/np.pi*180


def make_video_writer(name, window_width, window_height):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # MJPG
    return cv2.VideoWriter(name, fourcc, 10.0, (window_width, window_height))


def projection_to_intrinsics(mat, width=224, height=224):
    intrinsic_matrix = np.eye(3)
    mat = np.array(mat).reshape([4, 4]).T
    fv = width / 2 * mat[0, 0]
    fu = height / 2 * mat[1, 1]
    u0 = width / 2
    v0 = height / 2

    intrinsic_matrix[0, 0] = fu
    intrinsic_matrix[1, 1] = fv
    intrinsic_matrix[0, 2] = u0
    intrinsic_matrix[1, 2] = v0
    return intrinsic_matrix


def view_to_extrinsics(mat):
    pose = np.linalg.inv(np.array(mat).reshape([4, 4]).T)
    return np.linalg.inv(pose.dot(rotX(np.pi)))


def safemat2quat(mat):
    quat = np.array([1, 0, 0, 0])
    try:
        quat = mat2quat(mat)
    except:
        print(f"{bcolors.FAIL}Mat to quat Error.{bcolors.RESET}")
    quat[np.isnan(quat)] = 0
    return quat


def se3_transform_pc(pose, point):
    if point.shape[1] == 3:
        return pose[:3, :3].dot(point) + pose[:3, [3]]
    else:
        point_ = point.copy()
        point_[:3] = pose[:3, :3].dot(point[:3]) + pose[:3, [3]]
        return point_


def _cross_matrix(x):
    """
    cross product matrix
    """
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def a2e(q):
    p = np.array([0, 0, 1])
    r = _cross_matrix(np.cross(p, q))
    Rae = np.eye(3) + r + r.dot(r) / (1 + np.dot(p, q))
    return mat2euler(Rae)


def get_camera_constant(width):
    K = np.eye(3)
    K[0, 0] = K[0, 2] = K[1, 1] = K[1, 2] = width / 2.0

    offset_pose = np.zeros([4, 4])
    offset_pose[0, 1] = -1.
    offset_pose[1, 0] = offset_pose[2, 2] = offset_pose[3, 3] = 1.
    offset_pose[2, 3] = offset_pose[1, 3] = -0.036
    return offset_pose, K


def se3_inverse(RT):
    R = RT[:3, :3]
    T = RT[:3, 3].reshape((3, 1))
    RT_new = np.eye(4, dtype=np.float32)
    RT_new[:3, :3] = R.transpose()
    RT_new[:3, 3] = -1 * np.dot(R.transpose(), T).reshape((3))
    return RT_new


def backproject_camera_target(im_depth, K, target_mask):
    Kinv = np.linalg.inv(K)

    width = im_depth.shape[1]
    height = im_depth.shape[0]
    depth = im_depth.astype(np.float32, copy=True).flatten()
    mask = (depth != 0) * (target_mask.flatten() == 0)

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones((height, width), dtype=np.float32)
    x2d = np.stack((x, y, ones), axis=2).reshape(width * height, 3)  # each pixel

    # backprojection
    R = Kinv.dot(x2d.transpose())
    X = np.multiply(
        np.tile(depth.reshape(1, width * height), (3, 1)), R
    )
    X[1] *= -1  # flip y OPENGL. might be required for real-world
    return X[:, mask]


def backproject_camera_target_realworld(im_depth, K, target_mask):
    Kinv = np.linalg.inv(K)

    width = im_depth.shape[1]
    height = im_depth.shape[0]
    depth = im_depth.astype(np.float32, copy=True).flatten()
    mask = (depth != 0) * (target_mask.flatten() == 0)

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones((height, width), dtype=np.float32)
    x2d = np.stack((x, y, ones), axis=2).reshape(width * height, 3)  # each pixel

    # backprojection
    R = Kinv.dot(x2d.transpose())
    X = np.multiply(
        np.tile(depth.reshape(1, width * height), (3, 1)), R
    )
    return X[:, mask]


def proj_point_img(img, K, offset_pose, points, color=(255, 0, 0), vis=False, neg_y=True, real_world=False):
    xyz_points = offset_pose[:3, :3].dot(points) + offset_pose[:3, [3]]
    if real_world:
        pass
    elif neg_y:
        xyz_points[:2] *= -1
    p_xyz = K.dot(xyz_points)
    p_xyz = p_xyz[:, p_xyz[2] > 0.03]
    x, y = (p_xyz[0] / p_xyz[2]).astype(np.int), (p_xyz[1] / p_xyz[2]).astype(np.int)
    valid_idx_mask = (x > 0) * (x < img.shape[1] - 1) * (y > 0) * (y < img.shape[0] - 1)
    img[y[valid_idx_mask], x[valid_idx_mask]] = (0, 255, 0)
    return img


def unpack_action(action):
    pose_delta = np.eye(4)
    pose_delta[:3, :3] = euler2mat(action[3], action[4], action[5])
    pose_delta[:3, 3] = action[:3]
    return pose_delta


def unpack_pose(pose, rot_first=False):
    unpacked = np.eye(4)
    if rot_first:
        unpacked[:3, :3] = quat2mat(pose[:4])
        unpacked[:3, 3] = pose[4:]
    else:
        unpacked[:3, :3] = quat2mat(pose[3:])
        unpacked[:3, 3] = pose[:3]
    return unpacked


def quat2euler(quat):
    return mat2euler(quat2mat(quat))


def pack_pose(pose, rot_first=False):
    packed = np.zeros(7)
    if rot_first:
        packed[4:] = pose[:3, 3]
        packed[:4] = safemat2quat(pose[:3, :3])
    else:
        packed[:3] = pose[:3, 3]
        packed[3:] = safemat2quat(pose[:3, :3])
    return packed


def rotZ(rotz):
    RotZ = np.array(
        [
            [np.cos(rotz), -np.sin(rotz), 0, 0],
            [np.sin(rotz), np.cos(rotz), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    return RotZ


def rotY(roty):
    RotY = np.array(
        [
            [np.cos(roty), 0, np.sin(roty), 0],
            [0, 1, 0, 0],
            [-np.sin(roty), 0, np.cos(roty), 0],
            [0, 0, 0, 1],
        ]
    )
    return RotY


def rotX(rotx):
    RotX = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(rotx), -np.sin(rotx), 0],
            [0, np.sin(rotx), np.cos(rotx), 0],
            [0, 0, 0, 1],
        ]
    )
    return RotX

def unpack_pose_rot_first(pose):
    unpacked = np.eye(4)
    unpacked[:3, :3] = quat2mat(pose[:4])
    unpacked[:3, 3] = pose[4:]
    return unpacked


def pack_pose_rot_first(pose):
    packed = np.zeros(7)
    packed[4:] = pose[:3, 3]
    packed[:4] = safemat2quat(pose[:3, :3])
    return packed


def inv_pose(pose):
    return pack_pose(np.linalg.inv(unpack_pose(pose)))


def relative_pose(pose1, pose2):
    return pack_pose(np.linalg.inv(unpack_pose(pose1)).dot(unpack_pose(pose2)))


def compose_pose(pose1, pose2):
    return pack_pose(unpack_pose(pose1).dot(unpack_pose(pose2)))


def skew_matrix(r):
    """
    Get skew matrix of vector.
    r: 3 x 1
    r_hat: 3 x 3
    """
    return np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])


def inv_relative_pose(pose1, pose2, decompose=False):
    """
    pose1: b2a
    pose2: c2a
    relative_pose:  b2c
    shape: (7,)
    """

    from_pose = np.eye(4)
    from_pose[:3, :3] = quat2mat(pose1[3:])
    from_pose[:3, 3] = pose1[:3]
    to_pose = np.eye(4)
    to_pose[:3, :3] = quat2mat(pose2[3:])
    to_pose[:3, 3] = pose2[:3]
    relative_pose = se3_inverse(to_pose).dot(from_pose)
    return relative_pose


def ros_quat(tf_quat):  # wxyz -> xyzw
    quat = np.zeros(4)
    quat[-1] = tf_quat[0]
    quat[:-1] = tf_quat[1:]
    return quat


def tf_quat(ros_quat):  # xyzw -> wxyz
    quat = np.zeros(4)
    quat[0] = ros_quat[-1]
    quat[1:] = ros_quat[:-1]
    return quat


def distance_by_translation_point(p1, p2):
    """
    Gets two nx3 points and computes the distance between point p1 and p2.
    """
    return np.sqrt(np.sum(np.square(p1 - p2), axis=-1))


def regularize_pc_point_count(pc, npoints, use_farthest_point=False):
    """
    If point cloud pc has less points than npoints, it oversamples.
    Otherwise, it downsample the input pc to have npoint points.
    use_farthest_point: indicates whether to use farthest point sampling
    to downsample the points. Farthest point sampling version runs slower.
    """
    if pc.shape[0] > npoints:
        if use_farthest_point:
            pc = torch.from_numpy(pc).cuda()[None].float()
            new_xyz = (
                gather_operation(
                    pc.transpose(1, 2).contiguous(), furthest_point_sample(pc[..., :3].contiguous(), npoints)
                )
                .contiguous()
                )
            pc = new_xyz[0].T.detach().cpu().numpy()

        else:
            center_indexes = np.random.choice(
                range(pc.shape[0]), size=npoints, replace=False
            )
            pc = pc[center_indexes, :]
    else:
        required = npoints - pc.shape[0]
        if required > 0:
            index = np.random.choice(range(pc.shape[0]), size=required)
            pc = np.concatenate((pc, pc[index, :]), axis=0)
    return pc


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


class bcolors:
    OK = '\033[92m'  # GREEN
    WARNING = '\033[93m'  # YELLOW
    FAIL = '\033[91m'  # RED
    RESET = '\033[0m'  # RESET COLOR
