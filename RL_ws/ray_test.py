import ray
import datetime
import time
import queue
import random


# Define a remote actor class
@ray.remote(num_cpus=1)
class Actor:
    def __init__(self, actor_id, buffer_id):
        self.actor_id = actor_id
        print(f"Initialized Actor with ID {actor_id} {datetime.datetime.now()}")
        self.buffer_id = buffer_id

    def save_data(self, data):
        return [self.buffer_id.add_data.remote(item=data)]


@ray.remote(num_cpus=1)
class Agent:
    def __init__(self, agent_id, buffer_id):
        self.agent_id = agent_id
        print(f"Initialized Agent with ID {agent_id} {datetime.datetime.now()}")
        self.buffer_id = buffer_id

    def sample_data(self):
        return self.buffer_id.sample_data.remote()

    def fake_train(self):
        for _ in range(6):
            data = ray.get(self.sample_data())
            print(data)
            print(f"doing some process")
            time.sleep(0.5)


@ray.remote(num_cpus=1)
class Buffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data_queue = queue.Queue(maxsize=capacity)

    def add_data(self, item):
        if self.data_queue.qsize() < self.capacity:
            self.data_queue.put(item)
        else:
            self.data_queue.get()
            self.data_queue.put(item)

    def sample_data(self, batch_size=2):
        if self.data_queue.qsize() < batch_size:
            output = list(self.data_queue.queue)
        else:
            output = random.sample(list(self.data_queue.queue), batch_size)
        return output


if __name__ == "__main__":
    # Initialize ray
    ray.init(num_cpus=7)
    # Create the TrainPipeline instance
    num_actors = 5
    # buffer_id = Buffer.remote(0)
    # Create a buffer with capacity 5
    buffer_id = Buffer.remote(10)
    actor_ids = [Actor.remote(i, buffer_id) for i in range(num_actors)]
    agent_id = Agent.remote(0, buffer_id)

    ray.get([actor.save_data.remote(idx) for idx, actor in enumerate(actor_ids)])

    roll = []
    roll.extend([agent_id.fake_train.remote()])
    roll.extend([actor.save_data.remote(idx+5) for idx, actor in enumerate(actor_ids)])

    result = ray.get(roll)

    print(f"ready to shutdown")
    # Shutdown ray at the end
    ray.shutdown()
