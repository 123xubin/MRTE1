import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros(capacity)
        self.alpha = alpha
        self.pos = 0

    def push(self, transition, priority=1.0):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = priority ** self.alpha
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        valid_size = len(self.buffer)
        probs = self.priorities[:valid_size]
        probs = probs / probs.sum()  # 归一化
        indices = np.random.choice(valid_size, batch_size, p=probs)
        weights = (valid_size * probs[indices]) ** (-beta)
        weights /= weights.max()
        return [self.buffer[i] for i in indices], indices, weights

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio ** self.alpha

    def __len__(self):
        return len(self.buffer)
