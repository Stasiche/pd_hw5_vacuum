from collections import deque
import random


class Buffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, element):
        self.buffer.append(element)

    def sample(self, batch_size):
        return list(zip(*random.sample(self.buffer, batch_size)))

    def __len__(self):
        return len(self.buffer)