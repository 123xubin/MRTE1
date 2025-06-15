# evaluate.py
# 评估已训练模型在多个测试 TM 上的性能（基于 main_new.py 架构）

import os
import yaml
import torch
import numpy as np
import pandas as pd
from datetime import datetime

from environment import Environment
from agent1_gcn.ddpg import GCNDDPGAgent
from agent2_federated.server import FederatedServer
from agent2_federated.pheromone import PheromoneTable


def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def evaluate():
    config = load_config()
    COMMON_CFG = config['common']
    AGENT1_CFG = config['agent1']
    AGENT2_CFG = config['agent2']
    device = COMMON_CFG['device']

    # 初始化环境
    env = Environment(
        infile_prefix=COMMON_CFG['path_pre'],
        topo_name=COMMON_CFG['topo_name'],
        episode=1,
        epoch=COMMON_CFG['max_ep_steps'],
        start_index=0,
        train_flag=False,
        path_type=COMMON_CFG['intra_type'],
        synthesis_type=COMMON_CFG['inter_type'],
        traffic_type=COMMON_CFG['traffic_type'],
        small_ratio=COMMON_CFG['small_ratio'],
        failure_flag=0,
        block_num=1,
        intra_type=COMMON_CFG['intra_type'],
        inter_type=COMMON_CFG['inter_type']
    )

    # 初始化 Agent1
    agent1 = GCNDDPGAgent(
        in_feats=6,
        gcn_hidden=AGENT1_CFG['gcn_hidden'],
        mlp_hidden=AGENT1_CFG['mlp_hidden'],
        state_dim=16,
        action_dim=env.get_max_action_dim(),
        gamma=AGENT1_CFG['gamma'],
        tau=AGENT1_CFG['tau'],
        actor_lr=AGENT1_CFG['actor_lr'],
        critic_lr=AGENT1_CFG['critic_lr'],
        buffer_size=AGENT1_CFG['buffer_size'],
        device=device
    )
    agent1.load_model(COMMON_CFG['eval_agent1_ckpt'])

    # 初始化 Agent2 联邦策略
    fed_server = FederatedServer(
        num_regions=env.get_num_regions(),
        policy_cfg=AGENT2_CFG,
        device=device
    )
    fed_weights = torch.load(COMMON_CFG['eval_agent2_ckpt'])
    fed_server.dispatch_weights(fed_weights)

    pheromones = PheromoneTable()
    regionNum, edgeNumList, pathNumListDual, _ = env.get_info()

    # 评估所有测试 TM
    test_tm_dir = COMMON_CFG['eval_traffic_dir']
    tm_files = sorted([f for f in os.listdir(test_tm_dir) if f.endswith('.csv')])
    results = []

    for tm_file in tm_files:
        print(f"[EVAL] Traffic: {tm_file}")
        env.load_traffic(os.path.join(test_tm_dir, tm_file))
        env.reset()
        env.dynamic_path_selection()

        # 构建初始化动作
        actions = []
        for regionId in range(regionNum):
            intra_weights = [1.0 / n for n in pathNumListDual[regionId][0] for _ in range(n)]
            actions.append(intra_weights)
            inter_weights = [1.0 / n for n in pathNumListDual[regionId][1] for _ in range(n)]
            actions.append(inter_weights)
        env.com_action_matrix(actions)

        episode_reward = 0.0

        for step in range(COMMON_CFG['max_ep_steps']):
            for region_id, (src, dst) in env.region_flows():
                state = env.get_gcn_input(region_id, src, dst, env.compute_flowmap())
                x = torch.tensor(state['x'], dtype=torch.float32, device=device)
                edge_index = torch.tensor(state['edge_index'], dtype=torch.long, device=device)
                action = agent1.select_action(x, edge_index, path_indices=state['paths'])

                num_paths = len(state['paths'])
                if num_paths == 0:
                    continue
                if len(action) < num_paths:
                    action = np.pad(action, (0, num_paths - len(action)), constant_values=1e-6)
                    action /= action.sum()
                elif len(action) > num_paths:
                    action = action[:num_paths]

                reward, *_ = env.step(region_id, src, dst, action)
                episode_reward += reward

        metrics = {
            'Traffic': tm_file,
            'MaxUtil': env.get_max_utilization(),
            'AvgDelay': env.get_average_delay(),
            'Throughput': env.get_total_throughput(),
            'LossRate': env.get_total_loss_rate(),
            'EpisodeReward': episode_reward
        }
        results.append(metrics)

    df = pd.DataFrame(results)
    out_path = COMMON_CFG['eval_output_path']
    df.to_csv(out_path, index=False)
    print("[EVAL] Results saved to:", out_path)


if __name__ == '__main__':
    evaluate()