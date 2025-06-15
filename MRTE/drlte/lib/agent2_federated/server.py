import torch

from MRTE.MRTE.drlte.lib.agent2_federated.agent import FederatedClient


class FederatedServer:
    def __init__(self, num_regions, policy_cfg, device='cuda'):
        self.clients = [
            FederatedClient(region_id=i,
                            state_dim=policy_cfg['state_dim'],
                            action_dim=policy_cfg['action_dim'],
                            policy_cfg=policy_cfg,
                            device=device)
            for i in range(num_regions)
        ]

    def aggregate(self, client_weights):
        # 平均参数
        global_weights = {}
        for key in client_weights[0]:
            global_weights[key] = sum([w[key] for w in client_weights]) / len(client_weights)
        return global_weights

    def dispatch_weights(self, global_weights):
        for client in self.clients:
            client.set_model_weights(global_weights)

    def save(self, path):
        torch.save(self.clients[0].policy.state_dict(), path)