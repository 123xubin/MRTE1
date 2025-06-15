import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class Critic(nn.Module):
    """
    Critic网络：输入GCN处理后的全局状态 + 动作，输出Q值
    输入：
        x: 节点特征 (N, F)
        edge_index: 邻接边 (2, E)
        action: 动作向量 (action_dim)
    输出：
        Q值 (标量)
    """

    def __init__(self, in_feats, action_dim, gcn_hidden, mlp_hidden):
        super().__init__()
        self.expected_action_dim = action_dim
        # GCN层处理拓扑
        self.gcn1 = GCNConv(in_feats, gcn_hidden)
        self.gcn2 = GCNConv(gcn_hidden, gcn_hidden)
        # 全连接层处理拼接后的特征
        self.fc = nn.Sequential(
            nn.Linear(gcn_hidden + action_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1)
        )

    def forward(self, x, edge_index, action):
        # 1. 通过GCN处理拓扑
        h = torch.relu(self.gcn1(x, edge_index))
        h = torch.relu(self.gcn2(h, edge_index))
        # 2. 全局平均池化得到全局状态
        h_pooled = h.mean(dim=0)  # 输出形状: [gcn_hidden]

        if action.shape[-1] < self.expected_action_dim:
            pad_size = self.expected_action_dim - action.shape[-1]
            padding = torch.zeros(pad_size, device=action.device)
            action = torch.cat([action, padding], dim=-1)

        if action.shape[-1] != self.expected_action_dim:
            action = action[..., :self.expected_action_dim]  # 截断
            if action.shape[-1] < self.expected_action_dim:
                padding = torch.zeros(*action.shape[:-1], self.expected_action_dim - action.shape[-1],
                                      device=action.device)
                action = torch.cat([action, padding], dim=-1)

        # 3. 拼接动作并输出Q值
        combined = torch.cat([h_pooled, action], dim=-1)
        return self.fc(combined)