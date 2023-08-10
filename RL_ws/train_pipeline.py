import copy
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import ray
import time
from cvae_model import CVAE
# from replay_buffer import ReplayMemoryWrapper
from agent import AgentWrapper012, ReplayMemoryWrapper, RolloutWrapper012
from actor import ActorWrapper012
from torch.utils.tensorboard import SummaryWriter
import argparse
import datetime
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

parser = argparse.ArgumentParser(description="Description of your script.")
parser.add_argument("--cvae_train_times", type=int, default=10, help="How many time should cvae train")
parser.add_argument("--policy_train_times", type=int, default=1, help="How many time should cvae and policy train")
parser.add_argument("--log_dir", type=str, default="RL_ws/logs", help="where is the record")
parser.add_argument("--cvae_save_frequency", type=int, default=10, help="How many steps cvae take to save once")
parser.add_argument("--policy_save_frequency", type=int, default=1, help="How many steps policy take to save once")


if __name__ == "__main__":
    args = parser.parse_args()
    cvae_train_times = args.cvae_train_times
    policy_train_times = args.policy_train_times
    cvae_save_frequency = args.cvae_save_frequency
    policy_save_frequency = args.policy_save_frequency
    actor_num = 3
    batch_size = 16
    timestep = 0
    

    ray.init(num_cpus=12)
    replay_buffer_id = ReplayMemoryWrapper.remote(state_dim=2048, con_action_dim=64)
    rollout_agent_id = RolloutWrapper012.remote(replay_buffer_id)
    learner_id = AgentWrapper012.remote(replay_buffer_id)
    actor_ids = [ActorWrapper012.remote(replay_buffer_id, rollout_agent_id) for _ in range(actor_num)]
    current_file_path = os.path.abspath(__file__).replace('/train_pipeline.py', '/')
    model_path = current_file_path + "checkpoints/"
    log_path = current_file_path + "logs/"
    writer = SummaryWriter(log_path)

    ray.get([actor.rollout_once.remote() for actor in actor_ids])
    for _ in range(cvae_train_times):
        roll = []
        roll.extend([actor.rollout_once.remote() for actor in actor_ids])
        roll.extend([learner_id.cvae_train.remote(batch_size)])
        result = ray.get(roll)
        con_recon_loss, kl_loss = result[-1]
        writer.add_scalar("con_recon_loss", con_recon_loss, timestep)
        writer.add_scalar("kl_loss", kl_loss, timestep)
        if timestep % cvae_save_frequency == 0:
            ray.get([learner_id.save.remote(model_path + str(datetime.datetime.now()) + "cvae")])
        timestep += 1

    for _ in range(policy_train_times):
        roll = []
        roll.extend([actor.rollout_once.remote() for actor in actor_ids])
        roll.extend([learner_id.critic_train.remote(batch_size, timestep)])
        ray.get(roll)
        critic_loss, policy_loss = result[-1]
        writer.add_scalar("critic_loss", critic_loss, timestep)
        if policy_loss is not None:
            writer.add_scalar("policy_loss", policy_loss, timestep)

        if timestep % policy_save_frequency == 0:
            ray.get([learner_id.save.remote(model_path + str(datetime.datetime.now()) + "policy")])
        timestep += 1
