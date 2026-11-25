import random
import collections as cl
import numpy as np

class PrioritizedReplayBuffer:
    def __init__(self , capacity , alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.priorities = []

        self.position = 0
        self.buffer = cl.deque(maxlen=capacity)


    def add (self , expriance):
        max_priority = max(self.priorities, default=1.0)
        if self.buffer.__len__() < self.capacity:
            self.buffer.append(expriance)
            self.priorities.append(max_priority)
        else:
            self.priorities[self.position] = max_priority
            self.buffer[self.position] = expriance
        self.position = (self.position + 1) % self.capacity

    def sample(self , batch_size , beta=0.4):
        priorities = np.array(self.priorities , dtype=np.float32)
        scaled_priorities = priorities ** self.alpha
        probabilities = (
            scaled_priorities /
            scaled_priorities.sum()
        )

        indices = np.random.choice(
            self.buffer.__len__(),
            batch_size,
            p=probabilities
        )

        samples = [self.buffer[i] for i in indices]
        weights = ( self.buffer.__len__() * probabilities[indices] ) ** (-beta)

        weights /= weights.max()

        return samples, indices, weights

    def update_priorities(self , indices , td_error):
        priorities = np.abs(td_error + 1e-6)
        for idx , priority in zip(indices , priorities):
            self.priorities[idx] = priority
            
    def len(self):
        return self.buffer.__len__()


if __name__ == "__main__":
    buffer = PrioritizedReplayBuffer(5)

    buffer.push(
        "state_0",
        1,
        10,
        "state_1",
        False
    )
    print("buffer len" , buffer.len())
    print(buffer.sample(1))