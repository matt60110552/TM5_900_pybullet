import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as StepLR
import open3d as o3d
import os
import ray
import sys
import time
import datetime
from networks_DDQN import Feature_extractor, GaussianPolicy, QNetwork
from actor_DDQN import ActorWrapper012
from replay_buffer_DDQN import ReplayBuffer
from pid import PIDControl


class AgentWrapper(object):
    def __init__(self, online_replay_buffer_id, replay_buffer_id, train=True, scene_level=True, batch_size=40):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy, self.policy_optim, self.policy_scheduler = self.get_policy()
        self.critic, self.critic_optim, self.critic_scheduler = self.get_critic()
        self.policy_target, _, _ = self.get_policy()
        self.critic_target, _, _ = self.get_critic()
        self.policy_feat_extractor, self.policy_feat_extractor_optim, self.policy_feat_extractor_scheduler = self.get_feature_extractor(scene_level)
        self.offline_collect_times = 0  # How many episode should each actor to run for collecting the offline data
        self.offline_train_times = 0
        self.replay_buffer_id = replay_buffer_id
        self.online_replay_buffer_id = online_replay_buffer_id
        self.batch_size = batch_size
        self.PIDControl = PIDControl()

        if not train:
            self.policy.eval()
            self.critic.eval()
            self.policy_target.eval()
            self.critic_target.eval()
            self.policy_feat_extractor.eval()
        # self.beta = 0.01
        self.tau = 0.005
        self.timestep = 0
        self.policy_freq = 2
        self.cvae_retrain_frequency = 10
        self.discount = 0.99
        self.policy_loop_time = 15
        self.beta = 0.00005
        self.noise_scale = 0.05
        self.noise_clip = 0.08

    def select_action(self, goal_pos, joint_state, cur_gripper_pos, explore_rate=0):
        """
        Select actions for actors
        The state is combination of point and joint, should be 576
        """
        with torch.no_grad():
            goal_pos_tmp = self.prepare_data(goal_pos).unsqueeze(0)
            joint_state_tmp = self.prepare_data(joint_state).unsqueeze(0)
            cur_gripper_pos_tmp = self.prepare_data(cur_gripper_pos).unsqueeze(0)
            all_state = self.get_feature_for_policy(goal_pos_tmp, joint_state_tmp, cur_gripper_pos_tmp)

            conti_action = self.policy(all_state)
            #randomly explore part
            conti_action = (conti_action + explore_rate * self.noise_scale * torch.randn(conti_action.size()).to(self.device)).clamp(-0.15, 0.15)
        return conti_action.detach().squeeze().cpu().numpy()

    def get_feature_for_policy(self, goal_pos, joint_state, cur_gripper_pos):
        state = self.policy_feat_extractor(goal_pos, joint_state, cur_gripper_pos)
        return state

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
        extractor = Feature_extractor().to(self.device)
        extractor_optim = optim.Adam(
            extractor.parameters(), lr=1e-4, eps=1e-5, weight_decay=1e-5)
        extractor_scheduler = StepLR.MultiStepLR(
            extractor_optim,
            milestones=[30, 80, 120],
            gamma=0.5,
        )
        return extractor, extractor_optim, extractor_scheduler

    def critic_train(self, batch_size, timestep, ratio=None):
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
        or use detach()

        By doing this line again, the error disappear.


        ratio is for onpolicy data
        """
        critic_loss_list = []
        policy_loss_list = []
        bc_loss_list = []
        self.critic.train()
        self.critic_target.eval()
        self.policy.train()
        self.policy_target.eval()
        self.policy_feat_extractor.train()
        start = time.time()

        expert_batch_size = batch_size if not ratio else int(batch_size * (1-ratio))
        policy_batch_size = batch_size - expert_batch_size
        print(f"expert_batch_size: {expert_batch_size}")
        print(f"policy_batch_size: {policy_batch_size}")

        for _ in range(self.policy_loop_time):
            if expert_batch_size > 0:
                (goal_pos, joint_state, cur_gripper_pos, conti_action,
                next_joint_state, next_gripper_pos,
                reward, done) = ray.get([self.replay_buffer_id.sample.remote(expert_batch_size)])[0]
            elif policy_batch_size > 0:
                (policy_goal_pos, policy_joint_state, policy_cur_gripper_pos, policy_conti_action,
                policy_next_joint_state, policy_next_gripper_pos,
                policy_reward, policy_done) = ray.get([self.replay_buffer_id.sample.remote(policy_batch_size)])[0]

                # Combine the data
                print(f"expert_batch_size: {expert_batch_size}")
                print(f"policy_batch_size: {policy_batch_size}")
                goal_state = np.concatenate((goal_pos, policy_goal_pos), axis=0)
                joint_state = np.concatenate((joint_state, policy_joint_state), axis=0)
                cur_gripper_pos = np.concatenate((cur_gripper_pos, policy_cur_gripper_pos), axis=0)
                conti_action = np.concatenate((conti_action, policy_conti_action), axis=0)
                next_joint_state = np.concatenate((next_joint_state, policy_next_joint_state), axis=0)
                next_gripper_pos = np.concatenate((next_gripper_pos, policy_next_gripper_pos), axis=0)
                reward = np.concatenate((reward, policy_reward), axis=0)
                done = np.concatenate((done, policy_done), axis=0)

            self.goal_state = self.prepare_data(goal_state)
            self.joint_state = self.prepare_data(joint_state)
            self.cur_gripper_pos = self.prepare_data(cur_gripper_pos)
            self.conti_action = self.prepare_data(conti_action)
            self.next_joint_state = self.prepare_data(next_joint_state)
            self.next_gripper_pos = self.prepare_data(next_gripper_pos)
            self.reward = self.prepare_data(reward)
            self.done = self.prepare_data(done)


            with torch.no_grad():

                # get target_q
                next_policy_feat = self.get_feature_for_policy(self.goal_state, self.next_joint_state, self.next_gripper_pos)
                target_Q = self.get_target_q_value(next_policy_feat)
                target_Q = self.reward + (1 - self.done) * self.discount * target_Q

            feat_critic = self.get_feature_for_policy(self.goal_state, self.joint_state, self.cur_gripper_pos)
            current_q1, current_q2, _ = self.critic(feat_critic, self.conti_action)
            current_q1 = current_q1.squeeze()
            current_q2 = current_q2.squeeze()
            print(f"target_Q: {torch.mean(target_Q)}, current_q1: {torch.mean(current_q1)}, current_q2: {torch.mean(current_q2)}")

            critic1_loss = F.mse_loss(current_q1, target_Q)
            critic2_loss = F.mse_loss(current_q2, target_Q)
            critic_loss = critic1_loss + critic2_loss
            self.policy_feat_extractor_optim.zero_grad()
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
            self.critic_scheduler.step()
            self.policy_feat_extractor_optim.step()
            self.policy_feat_extractor_scheduler.step()

            critic_loss_list.append(critic_loss.detach().cpu().numpy())
            # Delayed Actor update
            if timestep % self.policy_freq == 0:
                feat_policy = feat_critic.detach()
                conti_action= self.policy(feat_policy)


                if expert_batch_size != 0:
                    bc_loss = F.mse_loss(conti_action[:expert_batch_size, :], self.conti_action[:expert_batch_size, :])
                else:
                    bc_loss = 0
                print(f"bc_loss: {bc_loss}")
                # get policy_loss through critic
                q1, q2, _ = self.critic(feat_policy, conti_action)
                policy_loss = -torch.min(q1, q2).mean()

                # combine all loss
                # all_policy_loss = policy_loss + 1.2 * bc_loss + terminal_loss
                # all_policy_loss = policy_loss + 15 * bc_loss + 10 * terminal_loss
                all_policy_loss = policy_loss + 10 * bc_loss

                self.critic_optim.zero_grad()
                self.policy_optim.zero_grad()
                all_policy_loss.backward()
                self.policy_optim.step()
                self.policy_scheduler.step()
                self.critic_optim.zero_grad()

                # Update target networks with Polyak averaging
                self.soft_update(source=self.policy, target=self.policy_target, tau=self.tau)
                self.soft_update(source=self.critic, target=self.critic_target, tau=self.tau)
                policy_loss_list.append(policy_loss.detach().cpu().numpy())
                bc_loss_list.append(bc_loss.detach().cpu().numpy())
        duration = time.time() - start
        print(f"policy duration: {duration}", end="\n")

        mean_critic_loss = sum(critic_loss_list)/len(critic_loss_list)
        mean_policy_loss = sum(policy_loss_list)/len(policy_loss_list) if timestep % self.policy_freq == 0 else None
        mean_bc_loss = sum(bc_loss_list)/len(bc_loss_list) if timestep % self.policy_freq == 0 else None
        return (mean_critic_loss, mean_policy_loss, mean_bc_loss)

    def get_target_q_value(self, next_all_feat):
        with torch.no_grad():
            conti_action = self.policy_target(next_all_feat)
            # Add noise to continuous action
            noise = (torch.randn_like(conti_action)*self.noise_scale).clamp(-self.noise_clip, self.noise_clip).to(self.device)
            # noisy_conti_action = torch.clamp(conti_action + noise, -self.noise_clip, self.noise_clip)
            noisy_conti_action = (conti_action + noise).clamp(-0.15, 0.15)

            q1, q2, _ = self.critic_target(next_all_feat, noisy_conti_action)
            target_q_value = torch.min(q1, q2).squeeze()

        return target_q_value

    def soft_update(self, source, target, tau):
        for (target_name, target_param), (name, source_param) in zip(
            target.named_parameters(), source.named_parameters()
        ):
            target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)

    def save(self, filename, timestep):
        save_dict = {
            "critic": self.critic.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "policy": self.policy.state_dict(),
            "policy_optim": self.policy_optim.state_dict(),
            "policy_feat_extractor": self.policy_feat_extractor.state_dict(),
            "policy_feat_extractor_optim": self.policy_feat_extractor_optim.state_dict(),
            "timestep": timestep
        }
        torch.save(save_dict, filename + ".pth")

    def load(self, filename, dict=False):
        load_dict = torch.load(filename) if not dict else filename
        self.critic.load_state_dict(load_dict["critic"])
        self.critic_optim.load_state_dict(load_dict["critic_optim"])
        self.critic_target = copy.deepcopy(self.critic)
        self.policy.load_state_dict(load_dict["policy"])
        self.policy_optim.load_state_dict(load_dict["policy_optim"])
        self.policy_target = copy.deepcopy(self.policy)
        self.policy_feat_extractor.load_state_dict(load_dict["policy_feat_extractor"])
        self.policy_feat_extractor_optim.load_state_dict(load_dict["policy_feat_extractor_optim"])
        return load_dict["timestep"]

    def prepare_data(self, input):
        if not isinstance(input, torch.Tensor):
            return torch.from_numpy(input).float().to(self.device)
        else:
            return input

    def get_weight(self):
        update_dict = {
            "critic": self.critic.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "policy": self.policy.state_dict(),
            "policy_optim": self.policy_optim.state_dict(),
            "policy_feat_extractor": self.policy_feat_extractor.state_dict(),
            "policy_feat_extractor_optim": self.policy_feat_extractor_optim.state_dict(),
            "timestep": None
        }
        return update_dict

@ray.remote(num_cpus=1, num_gpus=0.12)
class AgentWrapper012(AgentWrapper):
    pass


@ray.remote(num_cpus=1, num_gpus=1)
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
