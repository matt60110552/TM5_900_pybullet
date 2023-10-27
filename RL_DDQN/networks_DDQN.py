# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch
from torch import nn
import IPython
import numpy as np
import sys
from pointmlp import pointMLPElite


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
        points=2048
    ):
        super(PointNetFeature, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.point_encoder = EncP(in_channels=4, input_points=2048, num_stages=4, embed_dim=36, k_neighbors=40, beta=100, alpha=1000, LGA_block=[2,1,1,1], dim_expansion=[2,2,2,1], type='mn40')
        # self.linear_out = torch.nn.Linear(288, 512)
        self.point_encoder = pointMLPElite(points=points, in_channel=4, feature_dim=512)
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


# class JointFeature(nn.Module):
#     def __init__(
#         self
#     ):
#         """
#         This class is used to extract the joint's feature
#         The input should be (batch_size, 6)
#         """
#         super(JointFeature, self).__init__()
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.jointmlp = self.get_mlp(6, 64, 64)

    # def get_mlp(self, num_input, num_hidden, num_output, layer_type="linear"):
    #     if layer_type == "linear":
    #         return nn.Sequential(nn.Linear(num_input, num_hidden),
    #                              nn.ReLU(),
    #                              nn.Linear(num_hidden, num_hidden),
    #                              nn.ReLU(),
    #                              nn.Linear(num_hidden, num_output))
    #     else:
    #         return nn.Sequential(nn.Conv1d(num_input, num_hidden, 1),
    #                              nn.ReLU(),
    #                              nn.Conv1d(num_hidden, num_hidden, 1),
    #                              nn.ReLU(),
    #                              nn.Conv1d(num_hidden, num_output, 1))

#     def forward(self, jointstate):

#         jointfeature = self.jointmlp(jointstate)
#         return jointfeature


# class Feature_extractor(nn.Module):
#     def __init__(
#         self,
#         points=2048
#     ):
#         super(Feature_extractor, self).__init__()
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.jointextractor = JointFeature()
#         self.pointextractor = PointNetFeature(points=points)

#     def forward(self, pc, joint):

#         point_feature = self.pointextractor(pc)
#         joint_feature = self.jointextractor(joint)
#         all_feature = torch.cat((point_feature, joint_feature), dim=1)
#         return all_feature


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init()

        # Define the layers for the goal position
        self.goal_position = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
        )

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

        # Combine the input pathways
        self.combine = self.get_mlp(
            128 + 128 + 64, 512, 512)

    def forward(self, goal_position, joint_degrees, link_positions, gripper_position):
        x1 = self.goal_position(goal_position)
        x2 = self.joint_degrees(joint_degrees)
        x3 = self.gripper_position(gripper_position)

        # Concatenate the outputs
        x = torch.cat((x1, x2, x3), dim=1)

        # Process the concatenated features through the MLP layer
        output = self.combine(x)

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
                                nn.Linear(num_hidden, num_output))


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        num_inputs=512,
        hidden_dim=512
    ):
        super(GaussianPolicy, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.MLP = self.get_mlp(num_inputs, hidden_dim, hidden_dim)

        self.conti = nn.Linear(hidden_dim, 6)
        self.max_joint_limit = torch.tensor([0.15, 0.15, 0.15, 0.15, 0.15, 0.15]).to(self.device)
        self.apply(weights_init_)

    def forward(self, state):

        x1 = nn.ReLU(self.MLP(state))

        conti_action = torch.tanh(self.conti(x1)) * self.max_joint_limit

        return conti_action

    def get_mlp(self, num_input, num_hidden, num_output):
        return nn.Sequential(nn.Linear(num_input, num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden, num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden, num_output))

if __name__ == "__main__":
    feat_extractor = PointNetFeature()
    data = torch.rand(2, 3, 1024).to("cuda")
    result = feat_extractor(data)
    print(result.shape)
