import copy
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import ray
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


class train_pipeline(object):
    def __init__(
        self,
        replay_buffer_id,
        cvae_train_times,
        policy_train_times,
        actor_num=2,
        batch_size=10
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
        self.current_file_path = os.path.abspath(__file__).replace('/train_pipeline.py', '/')
        self.model_path = self.current_file_path + "checkpoints/"
        self.log_path = self.current_file_path + "logs/"
        self.cvae_train_times = cvae_train_times
        self.policy_train_times = policy_train_times
        self.cvae_save_frequency = args.cvae_save_frequency
        self.policy_save_frequency = args.policy_save_frequency

    def collect_offline_data(self):
        return [actor.rollout_once.remote() for actor in self.actor_ids]

    def collect_online_data(self):
        raise NotImplementedError
        return [actor.rollout_once.remote() for actor in self.actor_ids]

    def cvae_train(self, batch_size):
        return [self.learner_id.cvae_train.remote(batch_size)]

    def policy_train(self, batch_size, timestep):
        """
        This function will train policy and critic simultaneously.
        """
        return [self.learner_id.critic_train.remote(batch_size, timestep)]

    def learner_save_model(self, filename):
        filename = self.model_path + filename
        return [self.learner_id.save.remote(filename=filename)]

    def learner_load_model(self, filename):
        filename = self.model_path + filename
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

    def get_emb_table(self):
        return [self.learner_id.get_emb_table.remote()]


if __name__ == "__main__":
    ray.init(num_cpus=9, num_gpus=2)
    buffer_id = ReplayMemoryWrapper.remote(state_dim=2048, con_action_dim=64)
    args = parser.parse_args()
    train_pipeline = train_pipeline(replay_buffer_id=buffer_id,
                                    cvae_train_times=args.cvae_train_times,
                                    policy_train_times=args.policy_train_times)
    writer = SummaryWriter(train_pipeline.log_path)

    # First collect data for the empty buffer
    rewards = train_pipeline.collect_offline_data()
    ray.get(rewards)

    # test for loading model
    # ray.get(train_pipeline.learner_load_model(filename="20238415152"))

    """
    First train cvae
    """
    for train_pipeline.timestep in range(train_pipeline.cvae_train_times):
        rollout = []
        rollout.extend(train_pipeline.collect_offline_data())
        rollout.extend(train_pipeline.cvae_train(train_pipeline.batch_size))
        result = ray.get(rollout)
        con_recon_loss, kl_loss = result[-1]
        writer.add_scalar("con_recon_loss", con_recon_loss, train_pipeline.timestep)
        writer.add_scalar("kl_loss", kl_loss, train_pipeline.timestep)

        if train_pipeline.timestep % train_pipeline.cvae_save_frequency == 0:
            current_time = train_pipeline.get_datetime()
            ray.get(train_pipeline.learner_save_model(filename=current_time))
        train_pipeline.timestep += 1

    """
    Second train policy
    """
    for train_pipeline.timestep in range(train_pipeline.cvae_train_times, train_pipeline.cvae_train_times + train_pipeline.policy_train_times):
        rollout = []
        rollout.extend(train_pipeline.collect_offline_data())
        rollout.extend(train_pipeline.policy_train(train_pipeline.batch_size, train_pipeline.timestep))
        result = ray.get(rollout)
        critic_loss, policy_loss = result[-1]
        writer.add_scalar("critic_loss", critic_loss, train_pipeline.timestep)
        if policy_loss is not None:
            writer.add_scalar("policy_loss", policy_loss, train_pipeline.timestep)

        if train_pipeline.timestep % train_pipeline.policy_save_frequency == 0:
            current_time = train_pipeline.get_datetime()
            ray.get(train_pipeline.learner_save_model(filename=current_time))
        train_pipeline.timestep += 1

    ray.shutdown()
