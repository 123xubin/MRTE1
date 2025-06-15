import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler
from .gcn_actor import GCNPathActor
from .critic import Critic
from .replay_buffer import ReplayBuffer
from .gcn_explorer import Explorer



class GCNDDPGAgent:
    """
    GCN-DDPG智能体
    """
    def __init__(self, in_feats, gcn_hidden, mlp_hidden, state_dim, action_dim, gamma=0.99, tau=0.005, actor_lr=1e-4, critic_lr=1e-3, buffer_size=100000, device='cuda', max_epoch=100*5):
        self.scaler = GradScaler()
        self.device = device
        self.actor = GCNPathActor(in_feats, gcn_hidden, mlp_hidden).to(self.device)
        self.target_actor = GCNPathActor(in_feats, gcn_hidden, mlp_hidden).to(self.device)
        self.critic = Critic(in_feats=6, action_dim=action_dim, gcn_hidden=gcn_hidden, mlp_hidden=mlp_hidden).to(
            self.device)
        self.target_critic = Critic(in_feats=6, action_dim=action_dim, gcn_hidden=gcn_hidden, mlp_hidden=mlp_hidden).to(
            self.device)
        self.gamma = gamma
        self.tau = tau
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.buffer = ReplayBuffer(capacity=buffer_size)
        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic)
        self.explorer = Explorer(epsilon_begin=0.3,  epsilon_end=0.01,  max_epoch=max_epoch,  dim_act=action_dim,  num_path=[],  seed=42)

    def hard_update(self, target, source):
        target.load_state_dict(source.state_dict())

    def soft_update(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)

    def select_action(self, x, edge_index, path_indices):
        self.actor.eval()
        with torch.no_grad():
            weights = self.actor(x, edge_index, path_indices).cpu().numpy()
        self.actor.train()
        if np.any(np.isnan(weights)):
            print("[WARN] NaN in actor output")
        weights = self.explorer.get_act(weights)
        return np.array(weights)

    def store(self, state, action, reward, next_state, done):
        state['x'] = torch.as_tensor(state['x'], dtype=torch.float32, device=self.device)
        state['edge_index'] = torch.as_tensor(state['edge_index'], dtype=torch.long, device=self.device)
        next_state['x'] = torch.as_tensor(next_state['x'], dtype=torch.float32, device=self.device)
        next_state['edge_index'] = torch.as_tensor(next_state['edge_index'], dtype=torch.long, device=self.device)

        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float().to(self.device)
        if isinstance(reward, (float, int)):
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        if isinstance(done, (bool, int, float)):
            done = torch.tensor(done, dtype=torch.float32, device=self.device)
        self.buffer.push((state, action, reward, next_state, done))

    def update(self, batch_size):
        if len(self.buffer) < batch_size:
            return

        samples, indices, weights = self.buffer.sample(batch_size)

        for transition in samples:
            state, action, reward, next_state, done = transition
            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action).float().to(self.device)
            # 拆解当前状态
            x = state['x'].to(self.device)
            edge_index = state['edge_index'].to(self.device)
            path_indices = state['path_indices']

            # 拆解下一个状态
            next_x = next_state['x'].to(self.device)
            next_edge_index = next_state['edge_index'].to(self.device)
            next_path_indices = next_state['path_indices']

            # 得到状态嵌入
            with torch.no_grad():
                target_action = self.target_actor(next_x, next_edge_index, next_path_indices)
                target_value = self.target_critic(next_x, next_edge_index, target_action)
                y = reward + (1 - done) * self.gamma * target_value

            # 当前状态的 Q 值
            with torch.amp.autocast('cuda'):
                predicted_value = self.critic(x, edge_index, action)
                critic_loss = F.mse_loss(predicted_value, y)

            self.critic_optimizer.zero_grad()
            self.scaler.scale(critic_loss).backward()
            self.scaler.step(self.critic_optimizer)
            self.scaler.update()

            # Actor loss
            predicted_action = self.actor(x, edge_index, path_indices)
            actor_loss = -self.critic(x, edge_index, predicted_action).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 更新 target 网络
            self.soft_update(self.target_actor, self.actor)
            self.soft_update(self.target_critic, self.critic)

            td_error = torch.abs(predicted_value - y).detach().cpu().numpy()
            self.buffer.update_priorities(indices, td_error + 1e-5)

    # 在ddpg.py的GCNDDPGAgent类中添加以下方法
    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, path)