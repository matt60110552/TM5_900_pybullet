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
from agent import AgentWrapper012, ReplayMemoryWrapper
from actor import ActorWrapper012
from torch.utils.tensorboard import SummaryWriter
import argparse
import datetime
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

parser = argparse.ArgumentParser(description="Description of your script.")
parser.add_argument("--cvae_train_times", type=int, default=1, help="How many time should cvae train")
parser.add_argument("--policy_train_times", type=int, default=1, help="How many time should cvae and policy train")
parser.add_argument("--log_dir", type=str, default="RL_ws/logs", help="where is the record")
parser.add_argument("--cvae_save_frequency", type=int, default=1000, help="How many steps cvae take to save once")
parser.add_argument("--policy_save_frequency", type=int, default=1000, help="How many steps policy take to save once")


if __name__ == "__main__":
    args = parser.parse_args()
    cvae_train_times = 1
    policy_train_times = 1
    actor_num = 2
    batch_size = 10
    timestep = 0
    cvae_save_frequency = args.cvae_save_frequency
    policy_save_frequency = args.policy_save_frequency

    replay_buffer_id = ReplayMemoryWrapper.remote(state_dim=2048, con_action_dim=64)
    rollout_agent_id = AgentWrapper012.remote(replay_buffer_id)
    learner_id = AgentWrapper012.remote(replay_buffer_id)
    actor_ids = [ActorWrapper012.remote(replay_buffer_id, rollout_agent_id) for _ in range(actor_num)]
    current_file_path = os.path.abspath(__file__).replace('/train_pipeline.py', '/')
    model_path = current_file_path + "checkpoints/"
    log_path = current_file_path + "logs/"

    ray.get([actor.rollout_once.remote() for actor in actor_ids])
    # for _ in range(cvae_train_times):
    #     roll = []
    #     roll.extend([actor.rollout_once.remote() for actor in actor_ids])
    #     roll.extend([learner_id.cvae_train.remote(batch_size)])
    #     ray.get(roll)
    roll = []
    roll.extend([actor.rollout_once.remote() for actor in actor_ids])
    roll.extend([learner_id.cvae_train.remote(batch_size)])
    ray.get(roll)

    for _ in range(policy_train_times):
        roll = []
        roll.extend([actor.rollout_once.remote() for actor in actor_ids])
        roll.extend(learner_id.critic_train.remote(batch_size, timestep))
        ray.get(roll)
