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

    def add(self, obstacle_points, sampled_joint_values, valid_type):
        if self.size < self.max_size:
            self.buffer.put((obstacle_points, sampled_joint_values, valid_type))
            self.size += 1
        else:
            # Remove the oldest entry to make space for the new entry
            self.buffer.get()
            self.buffer.put((obstacle_points, sampled_joint_values, valid_type))

    def sample(self, batch_size):
        batch = []
        random_idx = random.sample(range(self.size), batch_size)

        for i in random_idx:
            batch.append(self.buffer.queue[i])

        (obstacle_points_batch, sampled_joint_values_batch, valid_type_batch) = zip(*batch)
        return (
            np.stack(obstacle_points_batch),
            np.stack(sampled_joint_values_batch),
            np.stack(valid_type_batch),
        )

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        (obstacle_points, sampled_joint_values, valid_type) = self.buffer.queue[idx]

        data = {
            "obstacle_points": np.float32(obstacle_points),
            "sampled_joint_values": np.float32(sampled_joint_values),
            "valid_type": np.float32(valid_type),
        }
        return data

    def is_full(self):
        return self.size == self.max_size

    def get_size(self):
        return self.size

    def save_data(self, filename):
        # Convert the contents of the queue to lists
        (obstacle_points_list, sampled_joint_values_list, valid_type_list) = [], [], []

        for item in self.buffer.queue:
            (obstacle_points, sampled_joint_values, valid_type) = item
            obstacle_points_list.append(obstacle_points)
            sampled_joint_values_list.append(sampled_joint_values)
            valid_type_list.append(valid_type)

        # Convert lists to NumPy arrays
        obstacle_points_array = np.array(obstacle_points_list)
        sampled_joint_values_array = np.array(sampled_joint_values_list)
        valid_type_array = np.array(valid_type_list)

        # Create a dictionary to store the arrays
        data_dict = {
            'obstacle_points': obstacle_points_array,
            'sampled_joint_values': sampled_joint_values_array,
            'valid_type': valid_type_array,
        }

        # Save the data dictionary to a NumPy file
        np.savez(filename, **data_dict)

    def load_data(self, filename):
        # Load the data dictionary from the NumPy file
        data_dict = np.load(filename, allow_pickle=True)

        # Extract individual arrays from the data dictionary
        arrays = ["obstacle_points", "sampled_joint_values", "valid_type"]
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
