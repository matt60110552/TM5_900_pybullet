import numpy as np
import torch
import queue
import ray


class ReplayBuffer(object):
    def __init__(self, state_dim, joint_state_dim=6, con_action_dim=6, dis_action_dim=1, max_size=int(1e5)):
        self.max_size = max_size
        self.size = 0

        self.state_dim = state_dim
        self.joint_state_dim = joint_state_dim
        self.con_action_dim = con_action_dim
        self.dis_action_dim = dis_action_dim

        self.buffer = queue.Queue(maxsize=max_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, pc_state, joint_state, con_action, dis_action, next_pc_state, next_joint_state, reward, done):
        if self.size < self.max_size:
            self.buffer.put((pc_state, joint_state, con_action, dis_action, next_pc_state, next_joint_state, reward, done))
            self.size += 1
        else:
            # Remove the oldest entry to make space for the new entry
            self.buffer.get()
            self.buffer.put((pc_state, joint_state, con_action, dis_action, next_pc_state, next_joint_state, reward, done))

    def sample(self, batch_size):
        batch = []
        for _ in range(batch_size):
            batch.append(self.buffer.get())

        pc_state_batch, joint_state_batch, con_action_batch, dis_action_batch, next_pc_state_batch, next_joint_state_batch, reward_batch, done_batch = zip(*batch)

        return (
            np.stack(pc_state_batch),
            np.stack(joint_state_batch),
            np.stack(con_action_batch),
            np.stack(dis_action_batch),
            np.stack(next_pc_state_batch),
            np.stack(next_joint_state_batch),
            np.stack(reward_batch),
            np.stack(done_batch)
        )

    def is_full(self):
        return self.size == self.max_size

    def get_size(self):
        return self.size

    def save_data(self, filename):
        # Convert the contents of the queue to lists
        pc_state_list, joint_state_list, con_action_list, dis_action_list, next_pc_state_list, next_joint_state_list, reward_list, done_list = [], [], [], [], [], [], [], []
        while not self.buffer.empty():
            pc_state, joint_state, con_action, dis_action, next_pc_state, next_joint_state, reward, done = self.buffer.get()
            pc_state_list.append(pc_state)
            joint_state_list.append(joint_state)
            con_action_list.append(con_action)
            dis_action_list.append(dis_action)
            next_pc_state_list.append(next_pc_state)
            next_joint_state_list.append(next_joint_state)
            reward_list.append(reward)
            done_list.append(done)

        # Convert lists to NumPy arrays
        pc_state_array = np.array(pc_state_list)
        joint_state_array = np.array(joint_state_list)
        con_action_array = np.array(con_action_list)
        dis_action_array = np.array(dis_action_list)
        next_pc_state_array = np.array(next_pc_state_list)
        next_joint_state_array = np.array(next_joint_state_list)
        reward_array = np.array(reward_list)
        done_array = np.array(done_list)

        # Create a dictionary to store the arrays
        data_dict = {
            'pc_state': pc_state_array,
            'joint_state': joint_state_array,
            'con_action': con_action_array,
            'dis_action': dis_action_array,
            'next_pc_state': next_pc_state_array,
            'next_joint_state': next_joint_state_array,
            'reward': reward_array,
            'done': done_array
        }

        # Save the data dictionary to a NumPy file
        np.savez(filename, **data_dict)


@ray.remote(num_cpus=1)
class ReplayMemoryWrapper(ReplayBuffer):
    pass
