import random
import collections as cl

class ReplayBuffer:
    def __init__(self , capacity):
        self.capacity = capacity
        self.buffer = cl.deque(maxlen=capacity)

    def push(self , state , action , reward , next_state , done):
        self.buffer.append((action , reward , next_state , done))

    def sample(self , batch_size):
        batch = random.sample(self.buffer , batch_size)
        return batch

    def len(self):
        return len(self.buffer)


if __name__ == "__main__":
    buffer = ReplayBuffer(5)

    buffer.push(
        "state_0",
        1,
        10,
        "state_1",
        False
    )
    print("buffer len" , buffer.len())
    print(buffer.sample(1))