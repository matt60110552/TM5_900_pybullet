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
from networks_plain import Feature_extractor, GaussianPolicy, ConditionalPredictNetwork, QNetwork
from actor_plain import ActorWrapper012
from replay_buffer_plain import ReplayBuffer
from pid import PIDControl


class AgentWrapper(object):
    def __init__(self, online_replay_buffer_id, replay_buffer_id, train=True, scene_level=True, batch_size=40):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cvae, self.cvae_optim, self.cvae_scheduler = self.get_cvae()
        self.policy, self.policy_optim, self.policy_scheduler = self.get_policy()
        self.critic, self.critic_optim, self.critic_scheduler = self.get_critic()
        self.policy_target, _, _ = self.get_policy()
        self.critic_target, _, _ = self.get_critic()
        self.policy_feat_extractor, self.policy_feat_extractor_optim = self.get_feature_extractor(scene_level)
        self.encoder_feat_extractor, self.encoder_feat_extractor_optim = self.get_feature_extractor(scene_level)
        self.offline_collect_times = 0  # How many episode should each actor to run for collecting the offline data
        self.offline_train_times = 0
        self.replay_buffer_id = replay_buffer_id
        self.online_replay_buffer_id = online_replay_buffer_id
        self.batch_size = batch_size
        self.PIDControl = PIDControl()

        if not train:
            self.cvae.eval()
            self.policy.eval()
            self.critic.eval()
            self.policy_target.eval()
            self.critic_target.eval()
            self.policy_feat_extractor.eval()
            self.encoder_feat_extractor.eval()
        # self.beta = 0.01
        self.tau = 0.005
        self.timestep = 0
        self.policy_freq = 2
        self.cvae_retrain_frequency = 10
        self.discount = 0.95
        self.cvae_loop_time = 40
        self.policy_loop_time = 20
        self.beta = 0.00005
        self.noise_scale = 0.1
        self.noise_clip = 0.5

        self.median = 1
        self.offset = 0

    def select_action(self, pc_state, joint_state, explore_rate=0):
        """
        Select actions for actors
        The state is combination of point and joint, should be 576
        """
        with torch.no_grad():
            pc_state_tmp = self.prepare_data(pc_state).unsqueeze(0)
            joint_state_tmp = self.prepare_data(joint_state).unsqueeze(0)
            all_state = self.get_feature_for_policy(pc_state_tmp, joint_state_tmp)

            conti_action = self.policy(all_state)
            # random explore part
            # noise = (explore_rate * torch.randn(conti_emb.size())).clamp(-self.noise_clip, self.noise_clip).to(self.device)
            # conti_emb = (conti_emb + noise).clamp(-1, 1)
            # conti_para = self.rescale_to_new_crange(conti_emb)

            # all_one_tensor = self.rescale_to_new_crange(torch.ones(conti_emb.size()).to(self.device))
            # all_minus_one_tensor = self.rescale_to_new_crange(-torch.ones(conti_emb.size()).to(self.device))
            # action_recon_one, _ = self.cvae.decode(all_state, all_one_tensor)
            # action_recon_minus_one, _ = self.cvae.decode(all_state, all_minus_one_tensor)
            # print(f"action_recon_one: {action_recon_one}, action_recon_minus_one: {action_recon_minus_one}")
            # action_recon, _ = self.cvae.decode(all_state, conti_para)
            # print(f"=====================crange{self.median}, {self.offset}")
        return conti_action.detach().squeeze().cpu().numpy()

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

    def get_feature_extractor(self, scene_level=True):
        points = 2048 if scene_level else 1024
        extractor = Feature_extractor(points=points)
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

    def cvae_train(self, batch_size, timestep, all_step, ratio=None, sample=True):
        start = time.time()
        self.cvae.train()
        self.encoder_feat_extractor.train()
        con_recon_loss_list = []
        kl_loss_list = []
        gripper_pre_loss_list = []
        manipulator_pre_loss_list = []

        expert_batch_size = batch_size if not ratio else int(batch_size * ratio)
        policy_batch_size = batch_size - expert_batch_size

        for _ in range(self.cvae_loop_time):
            ####################################################################################
            # The return of ray.get([self.replay_buffer_id.sample.remote(batch_size)])
            # is list, so use the [0] to get the tuple at index 0
            ####################################################################################
            if sample:
                (pc_state, joint_state, conti_action,
                 conti_para, next_pc_state, next_joint_state,
                 reward, done) = ray.get([self.replay_buffer_id.sample.remote(expert_batch_size)])[0]

                if policy_batch_size > 0:
                    (policy_pc_state, policy_joint_state, policy_conti_action,
                     policy_conti_para, policy_next_pc_state, policy_next_joint_state,
                     policy_reward, policy_done) = ray.get([self.replay_buffer_id.sample.remote(policy_batch_size)])[0]

                    # combine the datas
                    pc_state = np.concatenate((pc_state, policy_pc_state), axis=0)
                    joint_state = np.concatenate((joint_state, policy_joint_state), axis=0)
                    conti_action = np.concatenate((conti_action, policy_conti_action), axis=0)
                    conti_para = np.concatenate((conti_para, policy_conti_para), axis=0)
                    next_pc_state = np.concatenate((next_pc_state, policy_next_pc_state), axis=0)
                    next_joint_state = np.concatenate((next_joint_state, policy_next_joint_state), axis=0)
                    reward = np.concatenate((reward, policy_reward), axis=0)
                    done = np.concatenate((done, policy_done), axis=0)
                # # visualize to see if something wrong
                # vis_pc = pc_state[0, :, :3]
                # point_cloud = o3d.geometry.PointCloud()
                # point_cloud.points = o3d.utility.Vector3dVector(vis_pc)
                # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
                # o3d.visualization.draw_geometries([point_cloud, axis])
                # vis_pc = next_pc_state[0, :, :3]
                # point_cloud.points = o3d.utility.Vector3dVector(vis_pc)
                # o3d.visualization.draw_geometries([point_cloud, axis])

                self.pc_state = self.prepare_data(pc_state)
                self.joint_state = self.prepare_data(joint_state)
                self.conti_action = self.prepare_data(conti_action)
                self.conti_para = self.prepare_data(conti_para)
                self.next_pc_state = self.prepare_data(next_pc_state)
                self.next_joint_state = self.prepare_data(next_joint_state)
                self.reward = self.prepare_data(reward)
                self.done = self.prepare_data(done)

            self.all_feat = self.get_feature_for_encoder(self.pc_state, self.joint_state)
            (self.action_recon, self.manipulator_next,
             self.action_z, self.mean, self.log_std) = self.cvae(self.all_feat, self.conti_action)

            con_recon_loss = F.mse_loss(self.conti_action, self.action_recon)
            # print(f"(conti_action: {self.conti_action} action_recon: {self.action_recon}")
            # print(f"mean: {self.mean}, log_std: {self.log_std}")
            kl_loss = self.kl_divergence_loss(self.mean, self.log_std)
            gripper_pre_loss = F.mse_loss(self.next_pc_state[:, -3:, :3], self.manipulator_next[:, -3:, :3])
            manipulator_pre_loss = F.mse_loss(self.next_pc_state[:, -10:, :3], self.manipulator_next[:, -10:, :3])
            self.beta, _ = self.PIDControl.pid(20, kl_loss.item(), Kp=0.01, Ki=(-0.00001), Kd=0.)
            # print(f"beta: {self.beta}")
            # print(f"kl_loss: {kl_loss.item()}")
            total_loss = 500 * con_recon_loss + self.beta * kl_loss + 1 * gripper_pre_loss + 1 * manipulator_pre_loss
            # total_loss = 1000 * con_recon_loss + self.beta * kl_loss

            self.encoder_feat_extractor_optim.zero_grad()
            self.cvae_optim.zero_grad()
            total_loss.backward()
            self.cvae_optim.step()
            self.encoder_feat_extractor_optim.step()

            con_recon_loss_list.append(con_recon_loss.detach().cpu().numpy())
            kl_loss_list.append(kl_loss.detach().cpu().numpy())
            gripper_pre_loss_list.append(gripper_pre_loss.detach().cpu().numpy())
            manipulator_pre_loss_list.append(manipulator_pre_loss.detach().cpu().numpy())

        duration = time.time() - start
        # print(f"self.conti_action: {self.conti_action}")
        # print(f"self.action_recon: {self.action_recon}")
        print(f"cvae duration: {duration}", end="\n")

        mean_con_recon_loss = sum(con_recon_loss_list)/len(con_recon_loss_list)
        mean_kl_loss = sum(kl_loss_list)/len(kl_loss_list)
        mean_gripper_pre_loss = sum(gripper_pre_loss_list)/len(gripper_pre_loss_list)
        mean_manipulator_pre_loss = sum(manipulator_pre_loss_list)/len(manipulator_pre_loss_list)
        return (mean_con_recon_loss, mean_kl_loss, mean_gripper_pre_loss, mean_manipulator_pre_loss)

    def kl_divergence_loss(self, mean, log_std):
        # # Compute the element-wise KL divergence for each sample
        # kl_loss = -0.5 * torch.sum(1 + 2 * log_std - mean**2 - torch.exp(2 * log_std), dim=1)

        # # Compute the mean over the batch
        # kl_loss = torch.mean(kl_loss)

        klds = -0.5 * (1 + log_std - mean.pow(2) - log_std.exp())
        kl_loss = klds.sum(1).mean(0, True)
        return kl_loss

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


        ratio is for expert data and onpolicy data
        """
        critic_loss_list = []
        policy_loss_list = []
        bc_loss_list = []
        # terminal_loss_list = []
        self.critic.train()
        self.critic_target.eval()
        self.policy.train()
        self.policy_target.eval()
        self.policy_feat_extractor.train()
        self.cvae.eval()
        start = time.time()

        expert_batch_size = batch_size if not ratio else int(batch_size * ratio)
        policy_batch_size = batch_size - expert_batch_size

        for _ in range(self.policy_loop_time):
            if expert_batch_size > 0:
                (pc_state, joint_state, conti_action,
                 next_pc_state, next_joint_state,
                 reward, done) = ray.get([self.replay_buffer_id.sample.remote(expert_batch_size)])[0]
            if policy_batch_size > 0:
                (policy_pc_state, policy_joint_state, policy_conti_action,
                 policy_next_pc_state, policy_next_joint_state,
                 policy_reward, policy_done) = ray.get([self.replay_buffer_id.sample.remote(policy_batch_size)])[0]

                # combine the datas, not sure why some of data need to use concatenate rather than vstack
                pc_state = np.concatenate((pc_state, policy_pc_state), axis=0)
                joint_state = np.concatenate((joint_state, policy_joint_state), axis=0)
                conti_action = np.concatenate((conti_action, policy_conti_action), axis=0)
                next_pc_state = np.concatenate((next_pc_state, policy_next_pc_state), axis=0)
                next_joint_state = np.concatenate((next_joint_state, policy_next_joint_state), axis=0)
                reward = np.concatenate((reward, policy_reward), axis=0)
                done = np.concatenate((done, policy_done), axis=0)

            self.pc_state = self.prepare_data(pc_state)
            self.joint_state = self.prepare_data(joint_state)
            self.conti_action = self.prepare_data(conti_action)
            self.next_pc_state = self.prepare_data(next_pc_state)
            self.next_joint_state = self.prepare_data(next_joint_state)
            self.reward = self.prepare_data(reward)
            self.done = self.prepare_data(done)

            with torch.no_grad():

                # get target_q
                next_policy_feat = self.get_feature_for_policy(self.next_pc_state, self.next_joint_state)
                target_Q = self.get_target_q_value(next_policy_feat)
                target_Q = self.reward + (1 - self.done) * self.discount * target_Q


            feat_critic = self.get_feature_for_policy(self.pc_state, self.joint_state)
            current_q1, current_q2, _ = self.critic(feat_critic, self.conti_action)
            print(f"target_Q: {torch.mean(target_Q)}, current_q1: {torch.mean(current_q1)}, current_q2: {torch.mean(current_q2)}")

            critic1_loss = nn.MSELoss()(current_q1, target_Q)
            critic2_loss = nn.MSELoss()(current_q2, target_Q)
            critic_loss = critic1_loss + critic2_loss
            self.policy_feat_extractor_optim.zero_grad()
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
            self.policy_feat_extractor_optim.step()

            critic_loss_list.append(critic_loss.detach().cpu().numpy())
            # Delayed Actor update
            if timestep % self.policy_freq == 0:
                feat_policy = feat_critic.detach()
                conti_action = self.policy(feat_policy)


                if expert_batch_size != 0:
                    bc_loss = nn.MSELoss()(conti_action[:expert_batch_size], self.conti_action[:expert_batch_size])
                else:
                    bc_loss = 0

                # get policy_loss through critic
                q1, q2, _ = self.critic(feat_policy, conti_action)
                policy_loss = -torch.min(q1, q2).mean()
                # get termination loss
                # terminal_loss = nn.MSELoss()(discrete_emb, self.dis_embeddings.detach())

                # combine all loss
                # all_policy_loss = policy_loss + 1.2 * bc_loss + terminal_loss
                all_policy_loss = policy_loss + 15 * bc_loss

                self.policy_optim.zero_grad()
                all_policy_loss.backward()
                self.policy_optim.step()
                self.critic_optim.zero_grad()

                # Update target networks with Polyak averaging
                self.soft_update(source=self.policy, target=self.policy_target, tau=self.tau)
                self.soft_update(source=self.critic, target=self.critic_target, tau=self.tau)
                policy_loss_list.append(policy_loss.detach().cpu().numpy())
                bc_loss_list.append(bc_loss.detach().cpu().numpy())
                # terminal_loss_list.append(terminal_loss.detach().cpu().numpy())
        # retrain the cvae part in order to keep updating the latent space
        self.cvae_loop_time = 1
        if timestep % self.cvae_retrain_frequency == 0:
            self.cvae.train()
            self.pc_state = self.pc_state.detach()
            self.joint_state = self.joint_state.detach()
            self.conti_action = self.conti_action.detach()
            self.next_pc_state = self.next_pc_state.detach()
            self.next_joint_state = self.next_joint_state.detach()
            self.reward = self.reward.detach()
            self.done = self.done.detach()
            self.cvae_train(batch_size=batch_size, timestep=timestep, all_step=1, ratio=None, sample=False)
            self.cvae.eval()
            self.get_median_offset(self.batch_size)
            print(f"time6: {time.time() - start}")
        duration = time.time() - start
        print(f"policy duration: {duration}", end="\n")

        mean_critic_loss = sum(critic_loss_list)/len(critic_loss_list)
        mean_policy_loss = sum(policy_loss_list)/len(policy_loss_list) if timestep % self.policy_freq == 0 else None
        mean_bc_loss = sum(bc_loss_list)/len(bc_loss_list) if timestep % self.policy_freq == 0 else None
        # mean_terminal_loss = sum(terminal_loss_list)/len(terminal_loss_list) if timestep % self.policy_freq == 0 else None
        return (mean_critic_loss, mean_policy_loss, mean_bc_loss)

    def get_target_q_value(self, next_all_feat):
        with torch.no_grad():
            conti_emb = self.policy_target(next_all_feat)
            # Add noise to continuous action
            noise = (torch.randn_like(conti_emb)*self.noise_scale).clamp(-self.noise_clip, self.noise_clip).to(self.device)
            # noisy_conti_action = torch.clamp(conti_action + noise, -self.noise_clip, self.noise_clip)
            noisy_conti_emb = (conti_emb + noise).clamp(-1, 1)

            q1, q2, _ = self.critic_target(next_all_feat, noisy_conti_emb)
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
            "policy_feat_extractor": self.policy_feat_extractor.state_dict(),
            "policy_feat_extractor_optim": self.policy_feat_extractor_optim.state_dict(),
            "encoder_feat_extractor": self.encoder_feat_extractor.state_dict(),
            "encoder_feat_extractor_optim": self.encoder_feat_extractor_optim.state_dict(),
            "timestep": timestep,
            "median": self.median if self.median is not None else 0,
            "offset": self.offset if self.offset is not None else 0

        }
        torch.save(save_dict, filename + ".pth")

    def load(self, filename, dict=False):
        load_dict = torch.load(filename) if not dict else filename
        self.cvae.load_state_dict(load_dict["cvae"])
        self.cvae_optim.load_state_dict(load_dict["cvae_optim"])
        self.critic.load_state_dict(load_dict["critic"])
        self.critic_optim.load_state_dict(load_dict["critic_optim"])
        self.critic_target = copy.deepcopy(self.critic)
        self.policy.load_state_dict(load_dict["policy"])
        self.policy_optim.load_state_dict(load_dict["policy_optim"])
        self.policy_target = copy.deepcopy(self.policy)
        self.policy_feat_extractor.load_state_dict(load_dict["policy_feat_extractor"])
        self.policy_feat_extractor_optim.load_state_dict(load_dict["policy_feat_extractor_optim"])
        self.encoder_feat_extractor.load_state_dict(load_dict["encoder_feat_extractor"])
        self.encoder_feat_extractor_optim.load_state_dict(load_dict["encoder_feat_extractor_optim"])
        self.median = load_dict["median"]
        self.offset = load_dict["offset"]
        return load_dict["timestep"]

    def prepare_data(self, input):
        if not isinstance(input, torch.Tensor):
            return torch.from_numpy(input).float().to(self.device)
        else:
            return input

    def get_emb_table(self, action):
        action = torch.tanh(self.cvae.emb_table[action])
        return action

    def get_median_offset(self, batch_size, c_rate=0.1):
        """
        Get the median and offset of the data's each dimension
        use _tmp to avoid update the data in main task
        """
        action_z_list = []
        sample_time = (5000 // batch_size) + int(bool(5000 % batch_size))
        all_data_num = sample_time * batch_size
        for i in range(sample_time):
            (pc_state, joint_state, conti_action,
                conti_para, next_pc_state, _, _, _) = ray.get([self.replay_buffer_id.sample.remote(batch_size)])[0]
            pc_state_tmp = self.prepare_data(pc_state)
            joint_state_tmp = self.prepare_data(joint_state)
            conti_action_tmp = self.prepare_data(conti_action)
            with torch.no_grad():
                all_feat = self.get_feature_for_encoder(pc_state_tmp, joint_state_tmp)
                _, manipulator_pred, action_z, _, _ = self.cvae(all_feat, conti_action_tmp)

                action_z = action_z.detach().cpu().numpy()
                for x in action_z:
                    action_z_list.append(x)

        
        action_z_list = np.asarray(action_z_list)
        action_z_list.sort(axis=0)
        max_idx = int((1 - c_rate) * all_data_num)
        min_idx = int(c_rate * all_data_num)
        up_boundary = action_z_list[max_idx]
        down_boundary = action_z_list[min_idx]
        median = (up_boundary - down_boundary)/2
        offset = up_boundary - median

        # action_recon, state_pred = self.cvae.decode(cvae_state, dis_para, conti_para)
        # print(f"in reassign con state_pred: {state_pred.size()}")
        manipulator_pred_np = manipulator_pred.detach().cpu().numpy()
        # print(f"next_pc_state[:, -3:, :]; {next_pc_state[:, -3:, :]}")
        next_gripper_points = next_pc_state[:, -3:, :3]
        delta_loss = np.sqrt((np.square(manipulator_pred_np[:, -3:, :3] - next_gripper_points)).sum(axis=-1)).sum(-1).reshape(-1, 1)
        print(f"delta_loss_mean: {delta_loss}")

        self.median = torch.from_numpy(median).to(self.device)
        self.offset = torch.from_numpy(offset).to(self.device)
        self.rec_rate = np.mean(delta_loss)

    def capsulate_to_emb(self, parameter):
        # from [median - offset, median + offset] to [-1, 1]
        parameter = (parameter - self.offset) / self.median
        return parameter

    def rescale_to_new_crange(self, parameter):
        # from [-1, 1] to [median - offset, median + offset]
        parameter = parameter * self.median + self.offset
        return parameter

    def reassign_continue(self, conti_para, cvae_state,
                          conti_para_, next_pc_state, batch_size, recon_s_rate):
        """
        conti_para is from replayer buffer,
        conti_para_ is from the encoder after feeding the current state and action to it.
        """
        conti_emb = self.capsulate_to_emb(conti_para)
        # discrete_emb = self.capsulate_to_emb(dis_parameter)
        conti_emb_ = self.capsulate_to_emb(conti_para_)
        # print(f"in reasssign con cvae_state: {cvae_state.size()}, dis_para: {dis_para.size()}, conti_para: {conti_para.size()}")
        action_recon, state_pred = self.cvae.decode(cvae_state, conti_para)
        # print(f"in reassign con state_pred: {state_pred.size()}")
        state_pred_np = state_pred.detach().cpu().numpy()
        next_gripper_points = next_pc_state[:, -10:, :3].detach().cpu().numpy()
        delta_loss = np.sqrt((np.square(state_pred_np - next_gripper_points)).sum(axis=-1)).sum(-1).reshape(-1, 1)
        print(f"delta_loss: {np.mean(delta_loss)}")
        s_bing = (abs(delta_loss) < recon_s_rate) * 1
        parameter_relable_rate = sum(s_bing.reshape(1, -1)[0]) / batch_size
        s_bing = torch.FloatTensor(s_bing).float().to(self.device)
        self.conti_emb = s_bing * conti_emb + (1 - s_bing) * conti_emb_

    def reassign_discrete(self, dis_action, dis_emb_old, batch_size):
        dis_emb_new = self.get_emb_table(dis_action)
        # discrete relable need noise
        noise_discrete = (
                torch.randn_like(dis_emb_new) * 0.1
        ).clamp(-self.noise_clip, self.noise_clip)
        dis_emb_noise = (dis_emb_new.clamp(-1, 1) + noise_discrete).clamp(-1, 1)

        dis_action_old = self.select_discrete_action(dis_emb_old).reshape(-1, 1)
        d_new = dis_action
        d_old = dis_action_old.squeeze()
        d_bing = (d_new == d_old) * 1
        # discrete_relable_rate
        discrete_relable_rate = sum(d_bing.reshape(1, -1)[0]) / batch_size
        d_bing = torch.FloatTensor(np.expand_dims(d_bing, axis=1)).float().to(self.device)
        dis_emb_new = (d_bing * dis_emb_old + (1.0 - d_bing) * dis_emb_noise).clamp(-1, 1)
        self.dis_embeddings = dis_emb_new

        return dis_emb_new

    def get_weight(self):
        update_dict = {
            "cvae": self.cvae.state_dict(),
            "cvae_optim": self.cvae_optim.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "policy": self.policy.state_dict(),
            "policy_optim": self.policy_optim.state_dict(),
            "policy_feat_extractor": self.policy_feat_extractor.state_dict(),
            "policy_feat_extractor_optim": self.policy_feat_extractor_optim.state_dict(),
            "encoder_feat_extractor": self.encoder_feat_extractor.state_dict(),
            "encoder_feat_extractor_optim": self.encoder_feat_extractor_optim.state_dict(),
            "timestep": None,
            "median": self.median,
            "offset": self.offset
        }
        return update_dict

    def get_conti_para(self, pc_state, joint_state, conti_action):
        pc_state = self.prepare_data(pc_state)
        joint_state = self.prepare_data(joint_state)
        conti_action = self.prepare_data(conti_action)
        with torch.no_grad():
            state_feat = self.encoder_feat_extractor(pc_state.unsqueeze(dim=0), joint_state.unsqueeze(dim=0))
            conti_para, _, _ = self.cvae.encode(state_feat, conti_action.unsqueeze(dim=0))
            conti_para = conti_para.detach().cpu().numpy().squeeze()
        return conti_para


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
