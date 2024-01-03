import copy
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import ray
import time
# from replay_buffer import ReplayMemoryWrapper
from agent_scenecollision import AgentWrapper012, ReplayMemoryWrapper, RolloutWrapper012
from actor_scenecollision import ActorWrapper012
from torch.utils.tensorboard import SummaryWriter
import argparse
import datetime
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
parser = argparse.ArgumentParser(description="Description of your script.")
parser.add_argument("--num_cpus", type=int, default=12, help="number of cpus")
parser.add_argument("--visual", type=int, default=0, help="visualize or not")


if __name__ == "__main__":
    args = parser.parse_args()

    visual = args.visual
    ray.init(num_cpus=args.num_cpus)

    timestep = 1
    sim_actor_id = ActorWrapper012.remote(renders=visual)
    real_actor_id = ActorWrapper012.remote(renders=visual, simulation_id=sim_actor_id)
    current_file_path = os.path.abspath(__file__).replace('/train_pipeline_scenecollision.py', '/')
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M')

    npz_data_path = os.path.join(current_file_path,'npz_data')
    save_npz_data_path = npz_data_path
    # Create the directories if they don't exist
    os.makedirs(save_npz_data_path, exist_ok=True)

    # Start to demostrate
    for i in range(100):
        # roll = []
        # roll.extend(real_actor_id.rollout_once.remote())
        # result = ray.get(roll)

        ray.get(real_actor_id.rollout_once.remote())
        time.sleep(2)
        print(f"finish once")
