import copy
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import ray
import time
# from replay_buffer import ReplayMemoryWrapper
from agent_DDQN import AgentWrapper012, ReplayMemoryWrapper, RolloutWrapper012
from actor_DDQN import ActorWrapper012
from torch.utils.tensorboard import SummaryWriter
import argparse
import datetime
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
parser = argparse.ArgumentParser(description="Description of your script.")
parser.add_argument("--policy_train_times", type=int, default=1000, help="How many time should cvae and policy train")
parser.add_argument("--policy_save_frequency", type=int, default=50, help="How many steps policy take to save once")
parser.add_argument("--buffer_save_frequency", type=int, default=10, help="How many steps buffer take to save once")
parser.add_argument("--warmup_times", type=int, default=2, help="times for collecting data only")
parser.add_argument("--load_filename", type=str, default=None, help="The name of load file")
parser.add_argument("--mode", type=str, default="Train", help="Test or Train")
parser.add_argument("--num_cpus", type=int, default=12, help="number of cpus")
parser.add_argument("--actor_num", type=int, default=7, help="number of actors")
parser.add_argument("--batch_size", type=int, default=40, help="number of batch_size")
parser.add_argument("--visual", type=int, default=0, help="visualize or not")
parser.add_argument("--load_memory", type=int, default=1, help="load data or not")
parser.add_argument("--scene_level", type=int, default=1, help="convet to 0 if use target points only(object level)")


if __name__ == "__main__":
    args = parser.parse_args()

    if args.mode == "Train":
        policy_train_times = args.policy_train_times
        policy_save_frequency = args.policy_save_frequency
        buffer_save_frequency = args.buffer_save_frequency
        visual = args.visual
        load_memory = args.load_memory
        actor_num = args.actor_num
        batch_size = args.batch_size
        scene_level = bool(args.scene_level)
        ray.init(num_cpus=args.num_cpus)

        timestep = 1
        replay_buffer_id = ReplayMemoryWrapper.remote()
        replay_online_buffer_id = ReplayMemoryWrapper.remote()
        rollout_agent_id = RolloutWrapper012.remote(replay_online_buffer_id, replay_buffer_id,
                                                    train=False, scene_level=scene_level, batch_size=batch_size)
        learner_id = AgentWrapper012.remote(replay_online_buffer_id, replay_buffer_id, scene_level=scene_level,
                                            batch_size=batch_size)
        actor_ids = [ActorWrapper012.remote(replay_online_buffer_id, replay_buffer_id, rollout_agent_id,
                                            renders=visual, scene_level=scene_level) for _ in range(actor_num)]
        current_file_path = os.path.abspath(__file__).replace('/train_pipeline_DDQN.py', '/')
        current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M')

        checkpoint_path = os.path.join(current_file_path, 'checkpoints')
        # Create a folder for logs using the formatted datetime
        log_path = os.path.join(current_file_path, 'logs')
        npz_data_path = os.path.join(current_file_path, 'npz_data/scene_level') if scene_level else os.path.join(current_file_path, 'npz_data/object_level')
        # load model test
        if args.load_filename is not None:
            ray.get(learner_id.load.remote(checkpoint_path + "/" + args.load_filename))
            timestep = ray.get(rollout_agent_id.load.remote(checkpoint_path + "/" + args.load_filename)) + 1

        if load_memory:
            if (os.path.isfile(npz_data_path + "/" + "expert.npz") and
               os.path.isfile(npz_data_path + "/" + "on_policy.npz")):
                roll = []
                roll.extend([replay_buffer_id.load_data.remote(npz_data_path + "/" + "expert.npz")])
                roll.extend([replay_online_buffer_id.load_data.remote(npz_data_path + "/" + "on_policy.npz")])
                ray.get(roll)

        save_checkpoint_path = os.path.join(checkpoint_path, current_datetime)
        save_log_path = os.path.join(log_path, current_datetime)
        save_npz_data_path = npz_data_path
        # Create the directories if they don't exist
        os.makedirs(save_checkpoint_path, exist_ok=True)
        os.makedirs(save_log_path, exist_ok=True)
        os.makedirs(save_npz_data_path, exist_ok=True)

        # model_path = current_file_path + "checkpoints/"
        # log_path = current_file_path + "logs/"
        writer = SummaryWriter(save_log_path)

        for _ in range(args.warmup_times):
            ray.get([actor.rollout_once.remote(vis=True) for actor in actor_ids])

        for _ in range(args.warmup_times):
            ray.get([actor.rollout_once.remote(mode="onpolicy", explore_ratio=1) for actor in actor_ids])

        weight = ray.get([learner_id.get_weight.remote()])[0]
        ray.get([rollout_agent_id.load.remote(weight, dict=True)])

        for i in range(policy_train_times):
            # data_ratio = max(0., min(0.8, 1-i/policy_train_times))
            data_ratio = min(0.5, max(0., i/policy_train_times))
            explore_ratio = max(1 - (2*i)/(policy_train_times), 0.1)
            print(f"!!!!!!!!explore_ratio: {explore_ratio}")
            print(f"!!!!!!!!data_ratio: {data_ratio}")
            roll = []
            roll.extend([actor.rollout_once.remote(mode="both", explore_ratio=explore_ratio) for actor in actor_ids])
            roll.extend([learner_id.critic_train.remote(batch_size, timestep, ratio=data_ratio)])
            roll.extend([rollout_agent_id.load.remote(weight, dict=True)])
            roll.extend([learner_id.get_weight.remote()])
            result = ray.get(roll)
            critic_loss, policy_loss, bc_loss = result[-3]
            weight = result[-1]

            # get the total reward value of policy move for observation
            policy_reward_list = []
            for x in result[:actor_num]:
                if x[0] == 1:
                    policy_reward_list.append(x[1])

            writer.add_scalar("critic_loss", critic_loss, timestep)
            if len(policy_reward_list) > 0:
                writer.add_scalar("policy_reward", sum(policy_reward_list), timestep)
            if policy_loss is not None:
                writer.add_scalar("policy_loss", policy_loss, timestep)
                writer.add_scalar("bc_loss", bc_loss, timestep)
            if timestep % policy_save_frequency == 0:
                filename = save_checkpoint_path + "/policy_" + str(timestep)
                ray.get([learner_id.save.remote(filename, timestep)])
            if timestep % buffer_save_frequency == 0:
                filename = save_npz_data_path + "/expert"
                ray.get([replay_buffer_id.save_data.remote(filename)])
                filename = save_npz_data_path + "/on_policy"
                ray.get([replay_online_buffer_id.save_data.remote(filename)])
            timestep += 1

    elif args.mode == "Test":
        ray.init(num_cpus=args.num_cpus)
        actor_num = args.actor_num
        load_memory = args.load_memory
        scene_level = args.scene_level
        batch_size = args.batch_size
        replay_buffer_id = ReplayMemoryWrapper.remote(state_dim=2048, con_action_dim=64)
        replay_online_buffer_id = ReplayMemoryWrapper.remote(state_dim=2048, con_action_dim=64)
        rollout_agent_id = RolloutWrapper012.remote(replay_online_buffer_id, replay_buffer_id, train=False,
                                                    scene_level=scene_level)
        actor_ids = [ActorWrapper012.remote(replay_online_buffer_id, replay_buffer_id, rollout_agent_id,
                                            renders=True, scene_level=scene_level) for _ in range(actor_num)]
        current_file_path = os.path.abspath(__file__).replace('/train_pipeline_plain.py', '/')
        checkpoint_path = os.path.join(current_file_path, 'checkpoints')
        # npz_data_path = os.path.join(current_file_path, 'npz_data')
        npz_data_path = os.path.join(current_file_path, 'npz_data/scene_level') if scene_level else os.path.join(current_file_path, 'npz_data/object_level')

        if args.load_filename is not None:
            timestep = ray.get(rollout_agent_id.load.remote(checkpoint_path + "/" + args.load_filename))
        else:
            raise ValueError("please input load filename")

        # if load_memory:
        #     if (os.path.isfile(npz_data_path + "/" + "expert.npz") and
        #         os.path.isfile(npz_data_path + "/" + "on_policy.npz")):
        #         roll = []
        #         roll.extend([replay_buffer_id.load_data.remote(npz_data_path + "/" + "expert.npz")])
        #         roll.extend([replay_online_buffer_id.load_data.remote(npz_data_path + "/" + "on_policy.npz")])
        #         ray.get(roll)

        # # assign the median and offset of the rollout_agent
        # ray.get([rollout_agent_id.get_median_offset.remote(batch_size)])

        """
        Start to use model to move in pybullet
        """
        # # First to collect data and get the range for z(from policy) to rescale.
        # for _ in range(1):
        #     roll = []
        #     roll.extend([actor.rollout_once.remote() for actor in actor_ids])
        #     result = ray.get(roll)

        # Use the data above to get the range and then to use policy to interact with env.
        roll = []
        roll.extend([actor.rollout_once.remote(mode="onpolicy") for actor in actor_ids])
        result = ray.get(roll)

    else:
        print("please input Train or Test for mode")
