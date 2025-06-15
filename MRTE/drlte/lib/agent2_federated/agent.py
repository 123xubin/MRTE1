import numpy as np
import torch
from .policy import MLPPolicy
from .replaybuffer import ReplayBuffer

class FederatedClient:
        def __init__(self, region_id, state_dim, action_dim, policy_cfg, device='cuda'):
            self.region_id = region_id
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.device = device

            self.policy = MLPPolicy(state_dim, policy_cfg['hidden_dim'], action_dim).to(device)
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_cfg['lr'])
            self.buffer = ReplayBuffer(size_buffer=policy_cfg['buffer_size'])
            self.gamma = policy_cfg['gamma']
            self.batch_size = policy_cfg['batch_size']

        def select_action(self, state_np):
            if np.isnan(state_np).any():
                print(f"[ERROR] Region {self.region_id} 输入 state 含 NaN！state_np: {state_np}")
                state_np = np.nan_to_num(state_np)
                state_np = (state_np - np.mean(state_np)) / (np.std(state_np) + 1e-8)
            state = torch.tensor(state_np, dtype=torch.float32).to(self.device)
            logits = self.policy(state)
            if torch.isnan(logits).any():
                print(f"[ERROR] logits 为 NaN: {logits}")
                logits = torch.zeros_like(logits)
            probs = torch.softmax(logits, dim=-1)
            if torch.isnan(probs).any() or probs.sum().item() == 0:
                print(f"[ERROR] softmax 输出为 NaN 或全 0！probs: {probs}")
                probs = torch.ones_like(probs) / probs.shape[0]
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            return action.item()

        def local_update(self, env, pheromones):
            for (src, dst) in env.get_inter_domain_demands(self.region_id):
                state = env.get_agent2_state(self.region_id, env.get_node_region(dst), pheromones)
                action_idx = self.select_action(state)
                next_region, e_out, e_in = env.decode_action(self.region_id, action_idx)
                path = [src, e_out, e_in, dst]
                throughput = env.get_demand_volume(src, dst)
                delay = env.get_path_delay(path)
                reward, next_state, done = env.agent2_step(self.region_id, src, dst, next_region, e_out, e_in)
                pheromones.update((self.region_id, next_region, e_out, e_in), throughput, delay)

                self.buffer.push((state, action_idx, reward, next_state, done))

            if len(self.buffer) >= self.batch_size:
                self.update()

        def update(self):
            batch = self.buffer.sample_batch(self.batch_size)
            batch_s, batch_a, batch_r, batch_sn = batch
            loss_list = []
            for state, action, reward, next_state in zip(batch_s, batch_a, batch_r, batch_sn):
                s = torch.tensor(state, dtype=torch.float32, device=self.device)
                ns = torch.tensor(next_state, dtype=torch.float32, device=self.device)
                a = torch.tensor(action, dtype=torch.int64, device=self.device)
                r = torch.clamp(torch.tensor(reward, dtype=torch.float32, device=self.device), min=-10, max=10)

                logits = self.policy(s)
                logp = torch.log_softmax(logits, dim=-1)[a]
                with torch.no_grad():
                    next_logits = self.policy(ns)
                    next_value = torch.max(torch.softmax(next_logits, dim=-1))

                target = torch.clamp(r + self.gamma * next_value, min=-10, max=10)
                loss = -logp * target
                loss_list.append(loss)
                if torch.isnan(loss):
                    print(f"[ERROR] Loss 为 NaN！state={s}, reward={r}, target={target}")
            total_loss = torch.stack(loss_list).mean()
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()

        def get_model_weights(self):
            return {k: v.cpu() for k, v in self.policy.state_dict().items()}

        def set_model_weights(self, weights):
            self.policy.load_state_dict(weights)