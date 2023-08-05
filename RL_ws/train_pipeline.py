import copy
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import ray
from cvae_model import CVAE
from replay_buffer import ReplayMemoryWrapper
from agent import AgentWrapper012
from actor import ActorWrapper012
from torch.utils.tensorboard import SummaryWriter
import argparse
import datetime
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

parser = argparse.ArgumentParser(description="Description of your script.")
parser.add_argument("--cvae_train_times", type=int, default=1, help="How many time should cvae train")
parser.add_argument("--policy_cvae_train_times", type=int, default=1000, help="How many time should cvae and policy train")


class train_pipeline(object):
    def __init__(
        self,
        replay_buffer_id,
        actor_num=2,
        batch_size=16
    ):
        """
        This class is for all wrapper to work together.
        Given a replay_buffer_id, making two agentwrapper, one for rollout and one for learner.
        Use the replay_buffer_id to make multiple actors to interact with env.
        """
        self.replay_buffer_id = replay_buffer_id
        self.rollout_agent_id = AgentWrapper012.remote(self.replay_buffer_id)
        self.learner_id = AgentWrapper012.remote(self.replay_buffer_id)
        self.actor_num = actor_num
        self.actor_ids = [ActorWrapper012.remote(self.replay_buffer_id, self.rollout_agent_id) for _ in range(self.actor_num)]
        self.batch_size = batch_size
        self.timestep = 0
        self.current_file_path = os.path.abspath(__file__).replace('/train_pipeline.py', '/') + "checkpoints/"

    def collect_offline_data(self):
        return [actor.rollout_once.remote() for actor in self.actor_ids]

    def cvae_train(self):
        return [self.learner_id.cvae_train.remote(self.batch_size)]

    def learner_save_model(self, filename):
        filename = self.current_file_path + filename
        return [self.learner_id.save.remote(filename=filename)]

    def learner_load_model(self, filename):
        filename = self.current_file_path + filename
        return [self.learner_id.load.remote(filename=filename)]

    def get_datetime(self):
        current_year = str(datetime.datetime.now().year)
        current_month = str(datetime.datetime.now().month)
        current_day = str(datetime.datetime.now().day)
        current_hour = str(datetime.datetime.now().hour)
        current_minute = str(datetime.datetime.now().minute)
        current_second = str(datetime.datetime.now().second)

        current_time = current_year + current_month + current_day + current_hour + current_minute + current_second
        return current_time


if __name__ == "__main__":
    ray.init(num_gpus=2)
    args = parser.parse_args()

    buffer_id = ReplayMemoryWrapper.remote(state_dim=2048, con_action_dim=64)
    train_pipeline = train_pipeline(replay_buffer_id=buffer_id)

    # First collect data for the empty buffer
    rewards = train_pipeline.collect_offline_data()
    ray.get(rewards)

    # test for loading model
    # ray.get(train_pipeline.learner_load_model(filename="20238415152"))

    for train_pipeline.timestep in range(args.cvae_train_times):
        rollout = []
        rollout.extend(train_pipeline.collect_offline_data())
        rollout.extend(train_pipeline.cvae_train())
        result = ray.get(rollout)
        con_recon_loss, kl_loss = result[-1]
        train_pipeline.timestep += 1

    current_time = train_pipeline.get_datetime()
    ray.get(train_pipeline.learner_save_model(filename=current_time))
