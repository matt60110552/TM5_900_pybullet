import numpy as np
import torch
import queue
import ray
import random


class ReplayBuffer(object):
    def __init__(self, state_dim, joint_state_dim=6, con_action_dim=6, dis_action_dim=1, max_size=int(5e4)):
        self.max_size = max_size
        self.size = 0

        self.state_dim = state_dim
        self.joint_state_dim = joint_state_dim
        self.con_action_dim = con_action_dim
        self.dis_action_dim = dis_action_dim

        self.buffer = queue.Queue(maxsize=max_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, pc_state, joint_state, con_action, conti_para, next_pc_state, next_joint_state, reward, done, success):
        if self.size < self.max_size:
            self.buffer.put((pc_state, joint_state, con_action, conti_para, next_pc_state, next_joint_state, reward, done, success))
            self.size += 1
        else:
            # Remove the oldest entry to make space for the new entry
            self.buffer.get()
            self.buffer.put((pc_state, joint_state, con_action, conti_para, next_pc_state, next_joint_state, reward, done, success))

    def sample(self, batch_size):
        batch = []
        random_idx = random.sample(range(self.size), batch_size)

        for i in random_idx:
            batch.append(self.buffer.queue[i])
        # self.buffer.queue.extend(batch)  # Re-extend the queue with the sampled data
        pc_state_batch, joint_state_batch, con_action_batch, conti_para_batch, next_pc_state_batch, next_joint_state_batch, reward_batch, done_batch, success_batch = zip(*batch)

        return (
            np.stack(pc_state_batch),
            np.stack(joint_state_batch),
            np.stack(con_action_batch),
            np.stack(conti_para_batch),
            np.stack(next_pc_state_batch),
            np.stack(next_joint_state_batch),
            np.stack(reward_batch),
            np.stack(done_batch),
            np.stack(success_batch)
        )

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        pc_state, joint_state, con_action, conti_para, next_pc_state, next_joint_state, reward, done, success = self.buffer.queue[idx]

        data = {
            "pc_state": np.float32(pc_state),
            "joint_state": np.float32(joint_state),
            "con_action": np.float32(con_action),
            "conti_para": np.float32(conti_para),
            "next_pc_state": np.float32(next_pc_state),
            "next_joint_state": np.float32(next_joint_state),
            "reward": np.float32(reward),
            "done": np.float32(done),
            "success": np.float(success)
        }
        return data

    def is_full(self):
        return self.size == self.max_size

    def get_size(self):
        return self.size

    def save_data(self, filename):
        # Convert the contents of the queue to lists
        (pc_state_list, joint_state_list, con_action_list, conti_para_list, next_pc_state_list, next_joint_state_list,
         reward_list, done_list, success_list) = [], [], [], [], [], [], [], [], []

        for item in self.buffer.queue:
            pc_state, joint_state, con_action, conti_para, next_pc_state, next_joint_state, reward, done, success = item
            pc_state_list.append(pc_state)
            joint_state_list.append(joint_state)
            con_action_list.append(con_action)
            conti_para_list.append(conti_para)
            next_pc_state_list.append(next_pc_state)
            next_joint_state_list.append(next_joint_state)
            reward_list.append(reward)
            done_list.append(done)
            success_list.append(success)
        # Convert lists to NumPy arrays
        pc_state_array = np.array(pc_state_list)
        joint_state_array = np.array(joint_state_list)
        con_action_array = np.array(con_action_list)
        conti_para_array = np.array(conti_para_list)
        next_pc_state_array = np.array(next_pc_state_list)
        next_joint_state_array = np.array(next_joint_state_list)
        reward_array = np.array(reward_list)
        done_array = np.array(done_list)
        success_array = np.array(success_list)

        # Create a dictionary to store the arrays
        data_dict = {
            'pc_state': pc_state_array,
            'joint_state': joint_state_array,
            'con_action': con_action_array,
            'conti_para': conti_para_array,
            'next_pc_state': next_pc_state_array,
            'next_joint_state': next_joint_state_array,
            'reward': reward_array,
            'done': done_array,
            'success': success_array
        }

        # Save the data dictionary to a NumPy file
        np.savez(filename, **data_dict)

    def load_data(self, filename):
        # Load the data dictionary from the NumPy file
        data_dict = np.load(filename, allow_pickle=True)

        # Extract individual arrays from the data dictionary
        arrays = ['pc_state', 'joint_state', 'con_action','conti_para', 'next_pc_state', 'next_joint_state', 'reward', 'done', 'success']
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
