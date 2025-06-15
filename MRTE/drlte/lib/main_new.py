"""
    Main file for federated intra/inter-domain routing
"""
import os
import numpy as np
import yaml
import torch
from datetime import datetime
from environment import Environment
from agent1_gcn.ddpg import GCNDDPGAgent
from agent2_federated.server import FederatedServer
from MRTE.MRTE.drlte.lib.agent2_federated.pheromone import PheromoneTable



# ------------------ 配置加载 ------------------
def load_config():
    with open("../../config.yaml") as f:
        return yaml.safe_load(f)

config = load_config()
COMMON_CFG = config['common']
AGENT1_CFG = config['agent1']
AGENT2_CFG = config['agent2']

# ------------------ 环境初始化 ------------------
def init_environment():
    env = Environment(
        infile_prefix=COMMON_CFG['path_pre'],
        topo_name=COMMON_CFG['topo_name'],
        episode=COMMON_CFG['max_episodes'],
        epoch=COMMON_CFG['max_ep_steps'],
        start_index=0,
        train_flag=True,
        path_type=COMMON_CFG['intra_type'],
        synthesis_type=COMMON_CFG['inter_type'],
        traffic_type=COMMON_CFG['traffic_type'],
        small_ratio=COMMON_CFG['small_ratio'],
        failure_flag=0,
        block_num=1,
        intra_type=COMMON_CFG['intra_type'],
        inter_type=COMMON_CFG['inter_type']
    )
    return env

# ------------------ 日志系统 ------------------
class Logger:
    def __init__(self, path_pre):
        self.real_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(path_pre, "outputs/log", self.real_stamp)
        os.makedirs(self.log_dir, exist_ok=True)
        self.util_log = open(os.path.join(self.log_dir, 'util.log'), 'w', 1)

    def log(self, episode, max_util, reward):
        self.util_log.write(f"Episode {episode}: MaxUtil={max_util:.4f}, Reward={reward:.2f}\n")

    def close(self):
        self.util_log.close()



# ------------------ 主训练流程 ------------------
def main():
    os.makedirs("checkpoints", exist_ok=True)
    print(f"检查点将保存在：{os.path.abspath('checkpoints')}")

    # 初始化核心组件
    env = init_environment()
    logger = Logger(COMMON_CFG['path_pre'])
    regionNum, edgeNumList, pathNumListDual, regionNodeNeibor = env.get_info()
    def build_uniform_actions(pathNumListDual):
        actions = []
        for regionId in range(regionNum):
            # ---------- intra ----------
            intra_weights = []
            for n in pathNumListDual[regionId][0]:  # n = 该路径集合的条数
                intra_weights.extend([1.0 / n] * n)
            actions.append(intra_weights)  # 下标 region*2

            # ---------- inter ----------
            inter_weights = []
            for n in pathNumListDual[regionId][1]:
                inter_weights.extend([1.0 / n] * n)
            actions.append(inter_weights)  # 下标 region*2+1
        return actions

    # 转换为 actionmatrix 格式
    def build_actionmatrix(env, actions):
        env.com_action_matrix(actions)

    # Agent1: GCN-DDPG域内路由
    agent1 = GCNDDPGAgent(
        in_feats=6,  # 节点特征维度
        gcn_hidden=AGENT1_CFG['gcn_hidden'],
        mlp_hidden=AGENT1_CFG['mlp_hidden'],
        state_dim=16,
        action_dim=env.get_max_action_dim(),
        gamma=AGENT1_CFG['gamma'],
        tau=AGENT1_CFG['tau'],
        actor_lr=AGENT1_CFG['actor_lr'],
        critic_lr=AGENT1_CFG['critic_lr'],
        buffer_size=AGENT1_CFG['buffer_size'],
        device=COMMON_CFG['device']
    )

    # Agent2: 联邦学习服务端
    fed_server = FederatedServer(
        num_regions=env.get_num_regions(),
        policy_cfg=AGENT2_CFG,
        device=COMMON_CFG['device']
    )


    pheromones = PheromoneTable()

    # 训练循环
    for episode in range(COMMON_CFG['max_episodes']):
        env.reset()
        env.dynamic_path_selection()
        episode_reward = 0.0

        # 初始化动作矩阵为均匀分布
        actions = build_uniform_actions(pathNumListDual)
        env.com_action_matrix(actions)

        # 缓存初始流量图
        initial_flowmap = env.compute_flowmap()
        env.set_flowmap(initial_flowmap)

        for step in range(COMMON_CFG['max_ep_steps']):
            all_states = {}
            all_actions = {}

            # 收集所有流的动作
            for region_id, (src, dst) in env.region_flows():
                state = env.get_gcn_input(region_id, src, dst, env.get_flowmap())
                if state is None or len(state['paths']) == 0:
                    continue
                x = state['x'].detach().clone().to(dtype=torch.float32, device=agent1.device)
                edge_index = state['edge_index'].detach().clone().to(dtype=torch.long, device=agent1.device)
                action = agent1.select_action(x, edge_index, path_indices=state['paths'])

                # 修正维度
                num_paths = len(state['paths'])
                if len(action) < num_paths:
                    action = np.pad(action, (0, num_paths - len(action)), constant_values=1e-6)
                action = action / action.sum()

                all_states[(region_id, src, dst)] = state
                all_actions[(src, dst)] = action

            # 批量更新动作矩阵
            for (src, dst), action in all_actions.items():
                env.set_single_action(src, dst, action)

            # 更新流量图
            new_flowmap = env.compute_flowmap()
            env.set_flowmap(new_flowmap)

            # 全局统一奖励：负最大链路利用率（越低越好）
            max_util = env.get_max_utilization()
            global_reward = -max_util

            # 经验存储
            for (region_id, src, dst), state in all_states.items():
                next_state = env.get_gcn_input(region_id, src, dst, env.get_flowmap())
                if next_state is None:
                    continue
                x_next = torch.tensor(next_state['x'], dtype=torch.float32, device=agent1.device)
                edge_index_next = torch.tensor(next_state['edge_index'], dtype=torch.long, device=agent1.device)

                transition = {
                    "state": {
                        'x': torch.tensor(state['x'], dtype=torch.float32, device=agent1.device),
                        'edge_index': torch.tensor(state['edge_index'], dtype=torch.long, device=agent1.device),
                        'path_indices': state['paths']
                    },
                    "action": all_actions[(src, dst)],
                    "reward": torch.tensor(global_reward, dtype=torch.float32, device=agent1.device),
                    "next_state": {
                        'x': x_next,
                        'edge_index': edge_index_next,
                        'path_indices': next_state['paths']
                    },
                    "done": torch.tensor(False, dtype=torch.float32, device=agent1.device)
                }
                agent1.store(**transition)

            # 每步更新智能体
            if len(agent1.buffer) > AGENT1_CFG['minibatch']:
                agent1.update(batch_size=AGENT1_CFG['minibatch'])

            episode_reward += global_reward
            print(f"[INFO] step {step + 1} finished. Global Reward: {global_reward:.2f}")

        print(f"[INFO] Episode {episode + 1} finished. Reward: {episode_reward:.2f}, Buffer Size: {len(agent1.buffer)}")

        # ----------- Agent2 联邦学习更新 -----------
        for client in fed_server.clients:
            client.local_update(env, pheromones)

        # ----------- Agent2 联邦聚合 -----------
        if (episode + 1) % AGENT2_CFG['agg_interval'] == 0:
            print(f"[INFO] Aggregation triggered at Episode {episode + 1}")
            # 客户端上传参数
            client_weights = [client.get_model_weights() for client in fed_server.clients]

            # 服务端聚合
            global_weights = fed_server.aggregate(client_weights)

            # 下发全局参数
            fed_server.dispatch_weights(global_weights)
            print(f"[INFO] Federated aggregation completed for Episode {episode + 1}")

        # ----------- 日志与保存 -----------
        max_util = env.get_max_utilization()
        logger.log(episode+1, max_util, episode_reward)


        if (episode + 1) % 20 == 0:
            agent1.save(f"checkpoints/agent1_ep{episode+1}.pt")
            fed_server.save("checkpoints/fed_server.pt")


    logger.close()
    print("===== 训练完成 =====")

if __name__ == "__main__":
    main()