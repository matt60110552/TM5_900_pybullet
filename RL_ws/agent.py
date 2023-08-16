import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as StepLR
import os
import ray
import sys
import time
import datetime
from networks import Feature_extractor, GaussianPolicy, ConditionalPredictNetwork, QNetwork
from actor import ActorWrapper012
from replay_buffer import ReplayBuffer


class AgentWrapper(object):
    def __init__(self, replay_buffer_id):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cvae, self.cvae_optim, self.cvae_scheduler = self.get_cvae()
        self.policy, self.policy_optim, self.policy_scheduler = self.get_policy()
        self.critic, self.critic_optim, self.critic_scheduler = self.get_critic()
        self.policy_target, _, _ = self.get_policy()
        self.critic_target, _, _ = self.get_critic()
        self.policy_feat_extractor, self.policy_feat_extractor_optim = self.get_feature_extractor()
        self.encoder_feat_extractor, self.encoder_feat_extractor_optim = self.get_feature_extractor()
        self.offline_collect_times = 0  # How many episode should each actor to run for collecting the offline data
        self.offline_train_times = 0
        self.replay_buffer_id = replay_buffer_id

        # self.beta = 0.01
        self.tau = 0.005
        self.timestep = 0
        self.policy_freq = 5
        self.discount = 0.99
        self.cvae_loop_time = 20
        self.policy_loop_time = 20
        self.beta = 0.00005
        self.alpha = 0.005

    def select_action(self, state):
        """
        Select actions for actors
        The state is combination of point and joint, should be 576
        """
        dis_action, conti_action = self.policy(state)
        return dis_action, conti_action

    def get_feature_for_policy(self, pc, joints):
        state = self.policy_feat_extractor(pc, joints)
        return state

    def get_feature_for_encoder(self, pc, joints):
        state = self.encoder_feat_extractor(pc, joints)
        return state

    def get_cvae(self):
        cvae = ConditionalPredictNetwork().to(self.device)
        cvae_optim = optim.Adam(
            cvae.parameters(), lr=1e-4, eps=1e-5, weight_decay=1e-5)
        cvae_scheduler = StepLR.MultiStepLR(
            cvae_optim,
            milestones=[50, 80, 100, 120],
            gamma=0.5,
        )
        return cvae, cvae_optim, cvae_scheduler

    def get_policy(self):
        policy = GaussianPolicy().to(self.device)
        policy_optim = optim.Adam(
            policy.parameters(), lr=1e-4, eps=1e-5, weight_decay=1e-5)
        policy_scheduler = StepLR.MultiStepLR(
            policy_optim,
            milestones=[30, 80, 120],
            gamma=0.5,
        )
        return policy, policy_optim, policy_scheduler

    def get_critic(self):
        critic = QNetwork().to(self.device)
        critic_optim = optim.Adam(
            critic.parameters(), lr=1e-4, eps=1e-5, weight_decay=1e-5)
        critic_scheduler = StepLR.MultiStepLR(
            critic_optim,
            milestones=[30, 80, 120],
            gamma=0.5,
        )
        return critic, critic_optim, critic_scheduler

    def get_feature_extractor(self):
        extractor = Feature_extractor()
        extractor.to(self.device)
        extractor_optim = optim.Adam(
            extractor.parameters(), lr=1e-4, eps=1e-5, weight_decay=1e-5)
        return extractor, extractor_optim

    def get_match_scores(self, action):
        # compute similarity probability based on L2 norm
        embeddings = self.cvae.emb_table
        embeddings = torch.tanh(embeddings)
        action = action.to(self.device)
        # compute similarity probability based on L2 norm
        similarity = - self.pairwise_distances(action, embeddings)  # Negate euclidean to convert diff into similarity score
        return similarity

    def select_discrete_action(self, action, cuda=False):
        similarity = self.get_match_scores(action)
        val, pos = torch.max(similarity, dim=1)
        if cuda:
            return pos
        if len(pos) == 1:
            return pos.cpu().item()  # data.numpy()[0]
        else:
            return pos.cpu().numpy()

    def pairwise_distances(self, x, y):
        '''
        Input: x is a Nxd matrix
            y is a Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2

        Advantage: Less memory requirement O(M*d + N*d + M*N) instead of O(N*M*d)
        Computationally more expensive? Maybe, Not sure.
        adapted from: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
        '''

        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        y_t = torch.transpose(y, 0, 1)
        # a^2 + b^2 - 2ab
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        return dist

    def cvae_train(self, batch_size, timestep, all_step):
        start = time.time()
        self.cvae.train()
        self.encoder_feat_extractor.train()
        con_recon_loss_list = []
        kl_loss_list = []
        gripper_pre_loss_list = []
        for _ in range(self.cvae_loop_time):
            ####################################################################################
            # The return of ray.get([self.replay_buffer_id.sample.remote(batch_size)])
            # is list, so use the [0] to get the tuple at index 0
            ####################################################################################
            (pc_state, joint_state, conti_action,
                dis_action, next_pc_state, next_joint_state,
                reward, done) = ray.get([self.replay_buffer_id.sample.remote(batch_size)])[0]

            self.pc_state = self.prepare_data(pc_state)
            self.joint_state = self.prepare_data(joint_state)
            self.conti_action = self.prepare_data(conti_action)
            self.dis_action = dis_action  # This one has to be int for index
            self.next_pc_state = self.prepare_data(next_pc_state)
            self.next_joint_state = self.prepare_data(next_joint_state)
            self.reward = self.prepare_data(reward)
            self.done = self.prepare_data(done)

            self.all_feat = self.get_feature_for_encoder(self.pc_state, self.joint_state)
            self.dis_embeddings = self.cvae.emb_table[self.dis_action]
            self.action_recon, self.gripper_next, self.mean, self.log_std = self.cvae(self.all_feat, self.dis_embeddings, self.conti_action)


            con_recon_loss = F.mse_loss(self.conti_action, self.action_recon)
            # print(f"(conti_action: {self.conti_action} action_recon: {self.action_recon}")
            kl_loss = self.kl_divergence_loss(self.mean, self.log_std)
            gripper_pre_loss = F.mse_loss(self.next_pc_state[:, -3:, :3], self.gripper_next)

            self.beta = 0.0001 * min(1.0, timestep / (all_step//2))
            total_loss = con_recon_loss + self.beta * kl_loss + self.alpha * gripper_pre_loss

            self.encoder_feat_extractor_optim.zero_grad()
            self.cvae_optim.zero_grad()
            total_loss.backward()
            self.cvae_optim.step()
            self.encoder_feat_extractor_optim.step()

            con_recon_loss_list.append(con_recon_loss.detach().cpu().numpy())
            kl_loss_list.append(kl_loss.detach().cpu().numpy())
            gripper_pre_loss_list.append(gripper_pre_loss.detach().cpu().numpy())

        duration = time.time() - start
        print(f"cvae duration: {duration}", end="\n")

        mean_con_recon_loss = sum(con_recon_loss_list)/len(con_recon_loss_list)
        mean_kl_loss = sum(kl_loss_list)/len(kl_loss_list)
        mean_gripper_pre_loss = sum(gripper_pre_loss_list)/len(gripper_pre_loss_list)
        return (mean_con_recon_loss, mean_kl_loss, mean_gripper_pre_loss)

    def kl_divergence_loss(self, mean, log_std):
        # Compute the element-wise KL divergence for each sample
        kl_loss = -0.5 * torch.sum(1 + 2 * log_std - mean**2 - torch.exp(2 * log_std), dim=1)

        # Compute the mean over the batch
        kl_loss = torch.mean(kl_loss)

        return kl_loss

    def critic_train(self, batch_size, timestep):
        """
        Update critic, update policy depend on frequency
        Be careful, the backward propagation might cause error:

        Trying to backward through the graph a second time 
        (or directly access saved tensors after they have already been freed).
        Saved intermediate values of the graph are freed when you call .backward() or autograd.grad().
        Specify retain_graph=True if you need to backward through the graph a second time or
        if you need to access saved tensors after calling backward.

        Do all the process again can avoid the error, like this:

        all_feat = self.get_feature_for_policy(self.pc_state, self.joint_state)

        By doing this line again, the error disappear.

        """
        critic_loss_list = []
        policy_loss_list = []
        self.critic.train()
        self.critic_target.train()
        self.policy.train()
        self.policy_target.train()
        self.policy_feat_extractor.train()
        start = time.time()
        for _ in range(self.policy_loop_time):
            (pc_state, joint_state, conti_action,
                dis_action, next_pc_state, next_joint_state,
                reward, done) = ray.get([self.replay_buffer_id.sample.remote(batch_size)])[0]
            self.pc_state = self.prepare_data(pc_state)
            self.joint_state = self.prepare_data(joint_state)
            self.conti_action = self.prepare_data(conti_action)
            self.dis_action = dis_action  # This one has to be int for index
            self.dis_embeddings = self.cvae.emb_table[self.dis_action]
            self.next_pc_state = self.prepare_data(next_pc_state)
            self.next_joint_state = self.prepare_data(next_joint_state)
            self.reward = self.prepare_data(reward)
            self.done = self.prepare_data(done)

            with torch.no_grad():
                next_all_feat = self.get_feature_for_policy(self.next_pc_state, self.next_joint_state)
                target_Q = self.get_target_q_value(next_all_feat)
                target_Q = self.reward + (1 - self.done) * self.discount * target_Q

            all_feat = self.get_feature_for_policy(self.pc_state, self.joint_state)
            action_z, mean, log_std = self.cvae.encode(all_feat, self.dis_embeddings, self.conti_action)
            current_q1, current_q2, _ = self.critic(all_feat, self.dis_embeddings, action_z)

            critic1_loss = nn.MSELoss()(current_q1, target_Q)
            critic2_loss = nn.MSELoss()(current_q2, target_Q)
            critic_loss = critic1_loss + critic2_loss
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            critic_loss_list.append(critic_loss.detach().cpu().numpy())
            # Delayed Actor update
            if timestep % self.policy_freq == 0:
                # print(f"training policy net")
                # print(f"====================")
                all_feat = self.get_feature_for_policy(self.pc_state, self.joint_state)
                discrete_action, continue_action = self.policy(all_feat)
                q1, q2, _ = self.critic(all_feat, discrete_action, continue_action)
                policy_loss = -torch.min(q1, q2).mean()

                self.policy_optim.zero_grad()
                policy_loss.backward()
                self.policy_optim.step()

                # Update target networks with Polyak averaging
                self.soft_update(source=self.policy, target=self.policy_target, tau=self.tau)
                self.soft_update(source=self.critic, target=self.critic_target, tau=self.tau)
                policy_loss_list.append(policy_loss.detach().cpu().numpy())
            else:
                policy_loss = None

        duration = time.time() - start
        print(f"policy duration: {duration}", end="\n")

        mean_critic_loss = sum(critic_loss_list)/len(critic_loss_list)
        mean_policy_loss = sum(policy_loss_list)/len(policy_loss_list) if timestep % self.policy_freq == 0 else None
        return (mean_critic_loss, mean_policy_loss)

    def get_target_q_value(self, next_all_feat):
        dis_action, conti_action = self.policy_target(next_all_feat)
        q1, q2, _ = self.critic_target(next_all_feat, dis_action, conti_action)
        target_q_value = torch.min(q1, q2)
        return target_q_value

    def soft_update(self, source, target, tau):
        for (target_name, target_param), (name, source_param) in zip(
            target.named_parameters(), source.named_parameters()
        ):
            target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)

    def save(self, filename, timestep):
        save_dict = {
            "cvae": self.cvae.state_dict(),
            "cvae_optim": self.cvae_optim.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "policy": self.policy.state_dict(),
            "policy_optim": self.policy_optim.state_dict(),
            "timestep": timestep
        }
        torch.save(save_dict, filename + ".pth")

    def load(self, filename):
        load_dict = torch.load(filename)
        self.cvae.load_state_dict(load_dict["cvae"])
        self.cvae_optim.load_state_dict(load_dict["cvae_optim"])
        self.critic.load_state_dict(load_dict["critic"])
        self.critic_optim.load_state_dict(load_dict["critic_optim"])
        self.critic_target = copy.deepcopy(self.critic)
        self.policy.load_state_dict(load_dict["policy"])
        self.policy_optim.load_state_dict(load_dict["policy_optim"])
        self.policy_target = copy.deepcopy(self.policy)
        return load_dict["timestep"]

    def prepare_data(self, input):
        if not isinstance(input, torch.Tensor):
            return torch.from_numpy(input).float().to(self.device)
        else:
            return input

    def get_emb_table(self):
        return self.cvae.emb_table


@ray.remote(num_cpus=1, num_gpus=1)
class AgentWrapper012(AgentWrapper):
    pass


@ray.remote(num_cpus=1, num_gpus=0.12)
class RolloutWrapper012(AgentWrapper):
    pass


@ray.remote(num_cpus=1)
class ReplayMemoryWrapper(ReplayBuffer):
    pass


if __name__ == "__main__":
    ray.init()

    buffer_id = ReplayMemoryWrapper.remote(state_dim=2048, con_action_dim=64)
    agents = [ActorWrapper012.remote(buffer_id) for _ in range(7)]
    rewards = [agent.move_to_nearest_grasppose.remote() for agent in agents]

    for reward in rewards:
        print(ray.get(reward))

    size = ray.get(buffer_id.get_size.remote())
    ray.get(buffer_id.save_data.remote("RL_ws/offline_data/offline_data.npz"))
