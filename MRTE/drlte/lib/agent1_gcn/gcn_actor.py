import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCNPathActor(nn.Module):
    """
    GCN-based Actor for path selection (Agent1)
    输入:
        x: 节点特征 (N, F)
        edge_index: 邻接边 (2, E)
        path_indices: List[List[int]], 每条路径的节点索引序列
    输出:
        weights: 每条路径的概率分配 (和为1)
    """

    def __init__(self, in_feats, gcn_hidden, mlp_hidden):
        super().__init__()
        self.gcn1 = GCNConv(in_feats, gcn_hidden)
        self.gcn2 = GCNConv(gcn_hidden, gcn_hidden)
        self.mlp = nn.Sequential(
            nn.Linear(gcn_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1)
        )

    def forward(self, x, edge_index, path_indices):
        h = torch.relu(self.gcn1(x, edge_index))
        h = torch.relu(self.gcn2(h, edge_index))
        max_path_len = max(len(p) for p in path_indices) if path_indices else 0
        path_mask = torch.zeros(len(path_indices), max_path_len, dtype=torch.bool)
        path_embeds = torch.zeros(len(path_indices), h.size(1), device=h.device)
        for i, path in enumerate(path_indices):
            if path:  # 空路径保护
                indices = torch.tensor(path, device=h.device)
                path_embeds[i] = h[indices].mean(dim=0)
                path_mask[i, :len(path)] = True
        scores = self.mlp(path_embeds).squeeze(-1)
        weights = torch.softmax(scores, dim=0)
        return weights

