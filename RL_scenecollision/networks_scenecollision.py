# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch
from torch import nn
import IPython
import numpy as np
import sys
from pointmlp import pointMLPElite, Model
from pointnet2_ops.pointnet2_modules import PointnetSAModuleMSG


import torch.nn.functional as F
from torch.distributions import Normal
LOG_SIG_MAX = 2
LOG_SIG_MIN = -10
epsilon = 1e-6


class Identity(nn.Module):
    def forward(self, input):
        return input


def weights_init_(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm1d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)


class PointNetFeature(nn.Module):
    def __init__(
        self,
        points=512
    ):
        super(PointNetFeature, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.point_encoder = EncP(in_channels=4, input_points=2048, num_stages=4, embed_dim=36, k_neighbors=40, beta=100, alpha=1000, LGA_block=[2,1,1,1], dim_expansion=[2,2,2,1], type='mn40')
        # self.linear_out = torch.nn.Linear(288, 512)
        # self.point_encoder = pointMLPElite(points=points, in_channel=4, feature_dim=512)
        self.point_encoder = Model(points=points, embed_dim=32, groups=1, res_expansion=0.25, in_channel=4, feature_dim=512,
                                   activation="relu", bias=False, use_xyz=False, normalize="anchor",
                                   dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                                   k_neighbors=[24,24,24,24], reducers=[2, 2, 2, 2])
        
        self.point_encoder.apply(weights_init_)

    def forward(
        self,
        x,
    ):
        """
        The input shape of pointcloud has to be (batch_size, n, 4)
        """
        batch_size, _, channel = x.size()
        x = x.contiguous()
        # if channel == 3:
        #     xyz = x
        # else:
        #     xyz = x[:, :, :3]
        point_feats = self.point_encoder(x.permute(0, 2, 1))
        # xyz_feats = pc.permute(0, 2, 1)
        # points = pc[:, :, :3].clone()
        # point_feats = self.linear_out(self.point_encoder(points, xyz_feats))
        return point_feats


class SkeletonFeature(nn.Module):
    def __init__(
        self,
        point_num = 11
    ):
        super(SkeletonFeature, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.skeletonmlp = self.get_mlp(point_num * 4, 64, 64)

    def get_mlp(self, num_input, num_hidden, num_output, layer_type="linear"):
        if layer_type == "linear":
            return nn.Sequential(nn.Linear(num_input, num_hidden),
                                 nn.ReLU(),
                                 nn.Linear(num_hidden, num_hidden),
                                 nn.ReLU(),
                                 nn.Linear(num_hidden, num_output))
        else:
            return nn.Sequential(nn.Conv1d(num_input, num_hidden, 1),
                                 nn.ReLU(),
                                 nn.Conv1d(num_hidden, num_hidden, 1),
                                 nn.ReLU(),
                                 nn.Conv1d(num_hidden, num_output, 1))
    def forward(
        self,
        x,
    ):
        flatten_data = x.view(x.shape[0], -1)

        arm_feature = self.skeletonmlp(flatten_data)
        return arm_feature



"""
Below is scenecollisionnet part
"""
SCENE_PT_MLP = [3, 128, 256]
SCENE_VOX_MLP = [256, 512, 1024, 512]
CLS_FC = [2057, 1024, 256]


class SceneCollisionNet(nn.Module):
    def __init__(self, bounds, vox_size):
        super().__init__()
        self.bounds = nn.Parameter(
            torch.from_numpy(np.asarray(bounds)).float(), requires_grad=False
        )
        self.vox_size = nn.Parameter(
            torch.from_numpy(np.asarray(vox_size)).float(), requires_grad=False
        )
        self.num_voxels = nn.Parameter(
            ((self.bounds[1] - self.bounds[0]) / self.vox_size).long(),
            requires_grad=False,
        )

        self.scene_pt_mlp = nn.Sequential()
        for i in range(len(SCENE_PT_MLP) - 1):
            self.scene_pt_mlp.add_module(
                "pt_layer{}".format(i),
                nn.Conv1d(SCENE_PT_MLP[i], SCENE_PT_MLP[i + 1], first=(i == 0)),
            )

        self.scene_vox_mlp = nn.ModuleList()
        for i in range(len(SCENE_VOX_MLP) - 1):
            scene_conv = nn.Sequential()
            if SCENE_VOX_MLP[i + 1] > SCENE_VOX_MLP[i]:
                scene_conv.add_module(
                    "3d_conv_layer{}".format(i),
                    nn.Conv1d(
                        SCENE_VOX_MLP[i],
                        SCENE_VOX_MLP[i + 1],
                        kernel_size=3,
                        padding=1,
                    ),
                )
                scene_conv.add_module(
                    "3d_max_layer{}".format(i), nn.MaxPool3d(2, stride=2)
                )
            else:
                scene_conv.add_module(
                    "3d_convt_layer{}".format(i),
                    nn.ConvTranspose3d(
                        SCENE_VOX_MLP[i],
                        SCENE_VOX_MLP[i + 1],
                        kernel_size=2,
                        stride=2,
                    ),
                )
            self.scene_vox_mlp.append(scene_conv)


        self.classifier = nn.Sequential(
            nn.Linear(CLS_FC[0], CLS_FC[1], first=True),
            nn.Linear(CLS_FC[1], CLS_FC[2]),
            nn.Linear(CLS_FC[2], 1, activation=None),
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3
            else None
        )

        return xyz, features

    def _inds_to_flat(self, inds, scale=1):
        flat_inds = inds * torch.cuda.IntTensor(
            [
                self.num_voxels[1:].prod() // (scale ** 2),
                self.num_voxels[2] // scale,
                1,
            ],
            device=self.num_voxels.device,
        )
        return flat_inds.sum(axis=-1)

    def _inds_from_flat(self, flat_inds, scale=1):
        ind0 = flat_inds // (self.num_voxels[1:].prod() // (scale ** 2))
        ind1 = (flat_inds % (self.num_voxels[1:].prod() // (scale ** 2))) // (
            self.num_voxels[2] // scale
        )
        ind2 = (flat_inds % (self.num_voxels[1:].prod() // (scale ** 2))) % (
            self.num_voxels[2] // scale
        )
        return torch.stack((ind0, ind1, ind2), dim=-1)

    def voxel_inds(self, xyz, scale=1):
        inds = ((xyz - self.bounds[0]) // (scale * self.vox_size)).int()
        return self._inds_to_flat(inds, scale=scale)

    def get_scene_features(self, scene_pc):
        scene_xyz, scene_features = self._break_up_pc(scene_pc)
        scene_inds = self.voxel_inds(scene_xyz)

        # Featurize scene points and max pool over voxels
        scene_vox_centers = (
            self._inds_from_flat(scene_inds) * self.vox_size
            + self.vox_size / 2
            + self.bounds[0]
        )
        scene_xyz_centered = (scene_pc[..., :3] - scene_vox_centers).transpose(
            2, 1
        )
        if scene_features is not None:
            scene_features = self.scene_pt_mlp(
                torch.cat((scene_xyz_centered, scene_features), dim=1)
            )
        else:
            scene_features = self.scene_pt_mlp(scene_xyz_centered)
        max_vox_features = torch.zeros(
            (*scene_features.shape[:2], self.num_voxels.prod())
        ).to(scene_pc.device)
        if scene_inds.max() >= self.num_voxels.prod():
            print(
                scene_xyz[range(len(scene_pc)), scene_inds.max(axis=-1)[1]],
                scene_inds.max(),
            )
        assert scene_inds.max() < self.num_voxels.prod()
        assert scene_inds.min() >= 0

        with autocast(enabled=False):
            max_vox_features[
                ..., : scene_inds.max() + 1
            ] = torch_scatter.scatter_max(
                scene_features.float(), scene_inds[:, None, :]
            )[
                0
            ]
        max_vox_features = max_vox_features.reshape(
            *max_vox_features.shape[:2], *self.num_voxels.int()
        )

        # 3D conv over voxels
        l_vox_features = [max_vox_features]
        for i in range(len(self.scene_vox_mlp)):
            li_vox_features = self.scene_vox_mlp[i](l_vox_features[i])
            l_vox_features.append(li_vox_features)

        # Stack features from different levels
        stack_vox_features = torch.cat(
            (l_vox_features[1], l_vox_features[-1]), dim=1
        )
        stack_vox_features = stack_vox_features.reshape(
            *stack_vox_features.shape[:2], -1
        )
        return stack_vox_features


class RobotCollisionNet(nn.Module):
    def __init__(self, num_joints):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(num_joints, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, centers):
        return self.classifier(centers)

"""
End of scenecollisionnet part
"""




class Feature_extractor(nn.Module):
    def __init__(self):
        super(Feature_extractor, self).__init__()

        # Define the layers for the goal position
        # self.goal_position = nn.Sequential(
        #     nn.Linear(3, 64),
        #     nn.ReLU(),
        # )

        self.goal_position = self.get_mlp(40, 64, 64)

        # Define the layers for joint degrees
        self.joint_degrees = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
        )

        # Define the layers for the gripper position
        self.gripper_position = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
        )

        self.skeletonmlp = SkeletonFeature()


        # Combine all the arm's information
        self.arm_combine = self.get_mlp(
            128 + 128 + 64 + 64, 512, 512)
        

    def forward(self, goal_position, joint_degrees, gripper_position, skeleton_points):
        # First flatten teh input data
        flatten_goal_position = goal_position.view(goal_position.shape[0], -1)
        x1 = self.goal_position(flatten_goal_position)
        x2 = self.joint_degrees(joint_degrees)
        x3 = self.gripper_position(gripper_position)
        x4 = self.skeletonmlp(skeleton_points)

        # Concatenate the outputs
        x = torch.cat((x1, x2, x3, x4), dim=1)

        # Process the concatenated features through the MLP layer
        output = self.arm_combine(x)

        return output


    def get_mlp(self, num_input, num_hidden, num_output):
        return nn.Sequential(nn.Linear(num_input, num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden, num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden, num_output))



# https://github.com/pranz24/pytorch-soft-actor-critic/
class QNetwork(nn.Module):
    def __init__(
        self,
        num_inputs=512,
        hidden_dim=512,
        conti_dim=6
    ):
        super(QNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_inputs = num_inputs
        # Q1 architecture
        self.Q1_subnet = self.get_mlp(num_inputs + 6, hidden_dim, 1)

        # Q2 architecture
        self.Q2_subnet = self.get_mlp(num_inputs + 6, hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, conti_action):

        sa1 = torch.cat((state, conti_action), 1)
        sa2 = torch.cat((state, conti_action), 1)

        x1 = self.Q1_subnet(sa1)
        x2 = self.Q1_subnet(sa2)

        return x1, x2

    def get_mlp(self, num_input, num_hidden, num_output):
        return nn.Sequential(nn.Linear(num_input, num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden, num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden, num_hidden//2),
                                nn.ReLU(),
                                nn.Linear(num_hidden//2, num_output))


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        num_inputs=512,
        hidden_dim=512
    ):
        super(GaussianPolicy, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.MLP1 = self.get_mlp(num_inputs, hidden_dim, hidden_dim)
        self.MLP2 = self.get_mlp(hidden_dim, hidden_dim//2, hidden_dim//2)

        self.conti = nn.Linear(hidden_dim//2, 6)
        self.max_joint_limit = torch.tensor([0.15, 0.15, 0.15, 0.15, 0.15, 0.15]).to(self.device)
        self.apply(weights_init_)

    def forward(self, state):

        x1 = self.MLP1(state)
        x2 = self.MLP2(x1)

        conti_action = torch.tanh(self.conti(x2)) * self.max_joint_limit

        return conti_action

    def get_mlp(self, num_input, num_hidden, num_output):
        return nn.Sequential(nn.Linear(num_input, num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden, num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden, num_hidden//2),
                                nn.ReLU(),
                                nn.Linear(num_hidden//2, num_output))

if __name__ == "__main__":
    feat_extractor = PointNetFeature()
    data = torch.rand(2, 3, 1024).to("cuda")
    result = feat_extractor(data)
    print(result.shape)
