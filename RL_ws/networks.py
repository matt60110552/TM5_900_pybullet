# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch
from torch import nn
import IPython
import numpy as np
import sys
from Point_NN.models.point_pn import EncP
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
        self
    ):
        super(PointNetFeature, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.point_encoder = EncP(in_channels=4, input_points=2048, num_stages=4, embed_dim=36, k_neighbors=40, beta=100, alpha=1000, LGA_block=[2,1,1,1], dim_expansion=[2,2,2,1], type='mn40')
        # self.linear_out = torch.nn.Linear(288, 512)
        self.point_encoder = pointMLPElite(points=2048, in_channel=4, feature_dim=512)
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


class JointFeature(nn.Module):
    def __init__(
        self
    ):
        """
        This class is used to extract the joint's feature
        The input should be (batch_size, 6)
        """
        super(JointFeature, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.jointmlp = self.get_mlp(6, 64, 64)

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

    def forward(self, jointstate):

        jointfeature = self.jointmlp(jointstate)
        return jointfeature


class Feature_extractor(nn.Module):
    def __init__(
        self
    ):
        super(Feature_extractor, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.jointextractor = JointFeature()
        self.pointextractor = PointNetFeature()

    def forward(self, pc, joint):

        point_feature = self.pointextractor(pc)
        joint_feature = self.jointextractor(joint)
        all_feature = torch.cat((point_feature, joint_feature), dim=1)
        return all_feature


# https://github.com/pranz24/pytorch-soft-actor-critic/
class QNetwork(nn.Module):
    def __init__(
        self,
        num_inputs=576,
        hidden_dim=512,
        latent_size=64,
        conti_dim=6
    ):
        super(QNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_inputs = num_inputs
        """
        576 is for point feat(512) and joint feat(64)
        The two below is for dis_action(64) and conti_action(64)
        """
        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + 2 * latent_size, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + 2 * latent_size, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, discrete_action, parameter_action):

        sa1 = torch.cat((state, discrete_action, parameter_action), 1)
        sa2 = torch.cat((state, discrete_action, parameter_action), 1)
        x3 = None
        x1 = F.relu(self.linear1(sa1))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(sa2))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2, x3

    def state_discriminate(self, state):
        return F.sigmoid(self.discriminator(state))


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        num_inputs=576,
        hidden_dim=512,
        latent_size=64,
        max_action=1.0,
    ):
        super(GaussianPolicy, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_inputs = num_inputs
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.max_action = max_action

        self.discrete = nn.Linear(hidden_dim, latent_size)
        self.conti = nn.Linear(hidden_dim, latent_size)

        self.apply(weights_init_)

    def forward(self, state):

        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        continue_action = self.max_action * torch.tanh(self.conti(x))
        discrete_action = self.max_action * torch.tanh(self.discrete(x))
        return discrete_action, continue_action


class ConditionalPredictNetwork(nn.Module):
    def __init__(
        self,
        num_inputs=576,  # 512 for point_feat and 64 for joint_feat
        num_actions=6,
        hidden_dim=512,
        latent_size=64,
        action_space=None,
    ):
        super(ConditionalPredictNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        init_tensor = torch.rand(2, latent_size) * 2 - 1  # Don't initialize near the extremes.
        self.emb_table = torch.nn.Parameter(init_tensor.type(torch.float32), requires_grad=True)
        # self.max_joint_limit = np.array([4.712385, 3.14159, 3.14159,  3.14159,  3.14159,  4.712385])  # min_limit has same value with negative
        self.max_joint_limit = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # min_limit has same value with negative

        # Encoder
        # Input size is 576(512 for point, 64 for joint) plus 64(discrete action)
        self.encode_discrete = self.get_mlp(num_inputs + latent_size, hidden_dim, hidden_dim)
        self.linear_continue = self.get_mlp(num_actions, hidden_dim, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, latent_size)
        self.log_std_linear = nn.Linear(hidden_dim, latent_size)

        # Decoder
        # Input size is 576(512 for point, 64 for joint) plus 64(discrete action)
        self.decode_discrete = self.get_mlp(num_inputs + latent_size, hidden_dim, hidden_dim)
        self.linear_decode = self.get_mlp(latent_size, hidden_dim, hidden_dim)
        self.linear2 = self.get_mlp(hidden_dim, hidden_dim, hidden_dim)
        self.reconstruct = nn.Linear(hidden_dim, num_actions)
        self.linear3 = self.get_mlp(hidden_dim, hidden_dim, hidden_dim)
        self.state_predict = nn.Linear(hidden_dim, 3*3)

        self.apply(weights_init_)
        self.action_space = action_space

    def forward(self, state_feat, discrete_emb, conti_action, state_pred=True):

        action_z, mean, log_std = self.encode(state_feat, discrete_emb, conti_action)
        action_recon, state_next = self.decode(state_feat, discrete_emb, action_z, state_pred)
        return action_recon, state_next, action_z, mean, log_std

    def encode(self, state_feat, discrete_emb, conti_action):

        x1 = F.relu(self.encode_discrete(torch.cat((state_feat, discrete_emb), dim=1)))
        x2 = F.relu(self.linear_continue(conti_action))
        x = F.relu(self.linear1(x1 * x2))
        mean = self.mean(x)
        log_std = self.log_std_linear(x)  # .clamp(-4, 15)
        action_z = self.reparameterize(mean, log_std)

        return action_z, mean, log_std

    def decode(self, state_feat, discrete_emb, action_z, state_recon=True):

        state_feat = torch.cat((state_feat, discrete_emb), dim=1)
        x1 = F.relu(self.decode_discrete(state_feat))
        x2 = F.relu(self.linear_decode(action_z))
        x = F.relu(self.linear2(x1 * x2))
        x_t = self.reconstruct(x)

        action_recon = torch.tanh(x_t) * torch.from_numpy(self.max_joint_limit).float().to(self.device)
        if state_recon:
            state_pred = self.state_predict(F.relu(self.linear3(x)))
            state_pred = state_pred.view(-1, 3, 3)
            return action_recon, state_pred
        else:
            return action_recon, None

    def reparameterize(self, mu, log_std):
        std = torch.exp(0.5 * log_std)
        eps = torch.randn_like(std)
        return mu + eps * std

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


if __name__ == "__main__":
    feat_extractor = PointNetFeature()
    data = torch.rand(2, 3, 1024).to("cuda")
    result = feat_extractor(data)
    print(result.shape)
