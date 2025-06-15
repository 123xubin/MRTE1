import numpy as np

class PheromoneTable:
    def __init__(self, decay=0.9):
        self.table = {}  # {(i,j,u,v): tau}
        self.decay = decay

    def update(self, key, throughput, delay):
        R = throughput / (delay + 1e-6)
        delta_tau = 1 / (1 + np.exp(-R))
        tau_old = self.table.get(key, 0.1)
        new_tau = (1 - self.decay) * tau_old + self.decay * delta_tau
        tau_clipped = max(0.01, min(new_tau, 1.0))
        self.table[key] = tau_clipped

    def get(self, key):
        return self.table.get(key, 0.1)

    def clear(self):
        self.table.clear()

    def get_state_feature(self, region_id, neighbors, edges):
        """
        组合 pheromone + weights 成为 agent2 的状态向量的一部分
        """
        feature = []
        for (j, u, v) in edges:
            tau = self.get((region_id, j, u, v))
            feature.append(tau)
        return feature