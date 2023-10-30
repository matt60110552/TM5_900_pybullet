import numpy as np
import torch
import queue
import ray
import random

class ReplayBuffer(object):
    def __init__(self, max_size=int(5e4)):
        self.max_size = max_size
        self.size = 0

        self.buffer = queue.Queue(maxsize=max_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, goal_pos, joint_state, cur_gripper_pos, conti_action, next_joint_state, next_gripper_pos, reward, done):
        if self.size < self.max_size:
            self.buffer.put((goal_pos, joint_state, cur_gripper_pos, conti_action, next_joint_state, next_gripper_pos, reward, done))
            self.size += 1
        else:
            # Remove the oldest entry to make space for the new entry
            self.buffer.get()
            self.buffer.put((goal_pos, joint_state, cur_gripper_pos, conti_action, next_joint_state, next_gripper_pos, reward, done))

    def sample(self, batch_size):
        batch = []
        random_idx = random.sample(range(self.size), batch_size)

        for i in random_idx:
            batch.append(self.buffer.queue[i])

        goal_pos_batch, joint_state_batch, cur_gripper_pos_batch, conti_action_batch, next_joint_state_batch, next_gripper_pos_batch, reward_batch, done_batch = zip(*batch)
        return (
            np.stack(goal_pos_batch),
            np.stack(joint_state_batch),
            np.stack(cur_gripper_pos_batch),
            np.stack(conti_action_batch),
            np.stack(next_joint_state_batch),
            np.stack(next_gripper_pos_batch),
            np.stack(reward_batch),
            np.stack(done_batch)
        )

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        goal_pos, joint_state, cur_gripper_pos, conti_action, next_joint_state, next_gripper_pos, reward, done = self.buffer.queue[idx]

        data = {
            "goal_pos": np.float32(goal_pos),
            "joint_state": np.float32(joint_state),
            "cur_gripper_pos": np.float32(cur_gripper_pos),
            "conti_action": np.float32(conti_action[:6]),
            "next_joint_state": np.float32(next_joint_state),
            "next_gripper_pos": np.float32(next_gripper_pos),  # Adjust as needed
            "reward": np.float32(reward),
            "done": np.float32(done)
        }
        return data

    def is_full(self):
        return self.size == self.max_size

    def get_size(self):
        return self.size

    def save_data(self, filename):
        # Convert the contents of the queue to lists
        (goal_pos_list, joint_state_list, cur_gripper_pos_list, conti_action_list,
         next_joint_state_list, next_gripper_pos_list, reward_list, done_list) = [], [], [], [], [], [], [], []

        for item in self.buffer.queue:
            goal_pos, joint_state, cur_gripper_pos, conti_action, next_joint_state, next_gripper_pos, reward, done = item
            goal_pos_list.append(goal_pos)
            joint_state_list.append(joint_state)
            cur_gripper_pos_list.append(cur_gripper_pos)
            conti_action_list.append(conti_action)
            next_joint_state_list.append(next_joint_state)
            next_gripper_pos_list.append(next_gripper_pos)
            reward_list.append(reward)
            done_list.append(done)

        # Convert lists to NumPy arrays
        goal_pos_array = np.array(goal_pos_list)
        joint_state_array = np.array(joint_state_list)
        cur_gripper_pos_array = np.array(cur_gripper_pos_list)
        conti_action_array = np.array(conti_action_list)
        next_joint_state_array = np.array(next_joint_state_list)
        next_gripper_pos_array = np.array(next_gripper_pos_list)
        reward_array = np.array(reward_list)
        done_array = np.array(done_list)

        # Create a dictionary to store the arrays
        data_dict = {
            'goal_pos': goal_pos_array,
            'joint_state': joint_state_array,
            'cur_gripper_pos': cur_gripper_pos_array,
            'conti_action': conti_action_array,
            'next_joint_state': next_joint_state_array,
            'next_gripper_pos': next_gripper_pos_array,
            'reward': reward_array,
            'done': done_array
        }

        # Save the data dictionary to a NumPy file
        np.savez(filename, **data_dict)

    def load_data(self, filename):
        # Load the data dictionary from the NumPy file
        data_dict = np.load(filename, allow_pickle=True)

        # Extract individual arrays from the data dictionary
        arrays = ['goal_pos', 'joint_state', 'cur_gripper_pos', 'conti_action', 'next_joint_state', 'next_gripper_pos', 'reward', 'done']
        data_list = zip(*(data_dict[array] for array in arrays))

        # Clear the current buffer
        self.buffer.queue.clear()
        self.size = 0

        # Fill the buffer with the loaded data
        for data in data_list:
            self.add(*data)


@ray.remote(num_cpus=1)
class ReplayMemoryWrapper(ReplayBuffer):
    pass
