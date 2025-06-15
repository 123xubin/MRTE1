"""
    Main file of the system
"""
import sys
import os
from datetime import datetime

# 不要一开始就 import drlAgent 或 FLAGS
from environment import Environment
from agent1_gcn.ddpg import GCNDDPGAgent
import torch

from ma_demo.train import device

if not hasattr(sys, 'argv'):
    sys.argv = ['']

# ----------- 先用一个轻量 flag 方式读取 AGENT_TYPE -----------
# 防止因 flag.py/tf.app.flags 导致的错误
def get_agent_type_from_sysargv():
    for arg in sys.argv:
        if arg.startswith('--agent_type='):
            return arg.split('=')[1]
    return None

AGENT_TYPE = get_agent_type_from_sysargv()
if AGENT_TYPE is None:
    AGENT_TYPE = "agent1_gcn"  # 默认

# ----------- agent1_gcn 分支：只用 PyTorch，不依赖 drlAgent/flag.py -----------
if AGENT_TYPE == "agent1_gcn":
    # 你可以自己实现读取参数的逻辑，或手动写死参数（如下面这样）
    IS_TRAIN = True
    MAX_EPISODES = 100
    MAX_EP_STEPS = 5
    MINI_BATCH = 32
    PATHPRE = './'
    TOPO_NAME = '1221'
    SYNT_TYPE = 'synthetic'
    PATH_TYPE = 'KSP'
    FAILURE_FLAG = 0
    START_INDEX = 0
    SMALL_RATIO = 1.0
    BLOCK_NUM = 1
    REAL_STAMP = str(datetime.now())
    GAMMA = 0.99
    TAU = 0.005
    ACTOR_LEARNING_RATE = 1e-4
    CRITIC_LEARNING_RATE = 1e-3
    BUFFER_SIZE = 100000

    print("\n----Information list----")
    print("agent_type: %s" % (AGENT_TYPE))
    print("stamp_type: %s" % (REAL_STAMP))
    timeRecord = []
    import time
    timeRecord.append(time.time())

    # 初始化日志输出
    def init_output_file():
        dirLog = PATHPRE + "outputs/log/" + REAL_STAMP
        dirCkpoint = PATHPRE + "outputs/ckpoint/" + REAL_STAMP
        if not os.path.exists(dirLog):
            os.makedirs(dirLog, exist_ok=True)
        fileUtilOut = open(dirLog + '/util.log', 'w', 1)
        if IS_TRAIN and AGENT_TYPE != "ECMP":
            if not os.path.exists(dirCkpoint):
                os.makedirs(dirCkpoint, exist_ok=True)
        if not IS_TRAIN and AGENT_TYPE != "ECMP" and MAX_EPISODES == 1:
            fileEdgeOut = open(dirLog + '/edge.log', 'w', 1)
        else:
            fileEdgeOut = None
        return dirLog, dirCkpoint, fileUtilOut, fileEdgeOut

    def log_time_file(timeRecord, dirLog):
        print('\n' + REAL_STAMP)
        logfile = open(dirLog + "/runtime.log", 'w')
        runtimeType = ["inital time", "training time", "running time"]
        timeRecordPair = [[timeRecord[0], timeRecord[1]]]
        for t in range(len(timeRecordPair)):
            start_time = timeRecordPair[t][0]
            end_time = timeRecordPair[t][1]
            interval = int((end_time-start_time)*1000)
            timeMs = interval%1000
            timeS = int(interval/1000)%60
            timeMin = int((interval/1000-timeS)/60)%60
            timeH = int(interval/1000)/3600
            runtime = "%dh-%dmin-%ds-%dms" % (timeH, timeMin, timeS, timeMs)
            print("%s: %s" % (runtimeType[t], runtime))
            logfile.write("%s: %s\n" % (runtimeType[t], runtime))
        logfile.close()

    dirLog, dirCkpoint, fileUtilOut, fileEdgeOut = init_output_file()

    print("==== GCN-DDPG agent1_gcn 方案 ====")
    env = Environment(PATHPRE, TOPO_NAME, MAX_EPISODES, MAX_EP_STEPS, START_INDEX, IS_TRAIN, PATH_TYPE, SYNT_TYPE,
                      SMALL_RATIO, FAILURE_FLAG, BLOCK_NUM)
    regionNum, edgeNumList, pathNumListDual, regionNodeNeibor = env.get_info()


    # 假设每个区域有 intra/inter path 结构，均分动作
    # 修正版 build_uniform_actions
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


    all_path_lengths = []
    for region in pathNumListDual:
        intra_nums, inter_nums = region
        all_path_lengths += intra_nums + inter_nums  # 所有路径数量
    max_action_dim = max(all_path_lengths)

    agent = GCNDDPGAgent(
        in_feats=6,  # 节点特征维度
        gcn_hidden=16,
        mlp_hidden=16,
        state_dim=16,  # 池化后的 gcn_hidden
        action_dim=max_action_dim,  # 路径数目
        gamma=GAMMA,
        tau=TAU,
        actor_lr=ACTOR_LEARNING_RATE,
        critic_lr=CRITIC_LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        device='cuda',
        max_epoch=MAX_EPISODES * MAX_EP_STEPS
    )

    update_count = 0
    update_freq = 5
    warmup_steps = 500
    for episode in range(MAX_EPISODES):
        env.reset()
        total_reward = 0

        actions = build_uniform_actions(pathNumListDual)
        build_actionmatrix(env, actions)
        # for r in range(regionNum):
        #     intra_len = len(actions[r * 2])
        #     inter_len = len(actions[r * 2 + 1])

            # 期望长度＝该 region 的所有 intra / inter 路径数之和
            # expect_intra = sum(pathNumListDual[r][0])
            # expect_inter = sum(pathNumListDual[r][1])
            #
            # print(f"region {r}: intra={intra_len}/{expect_intra}, "
            #       f"inter={inter_len}/{expect_inter}")


        for step in range(MAX_EP_STEPS):
            # 需根据你的 region_flows 逻辑遍历
            flowmap = env.compute_flowmap()
            for region_id, (src, dst) in env.region_flows():
                output = env.get_gcn_input(region_id, src, dst, flowmap)

                X = output['x']
                edge_index = output['edge_index']
                path_indices = output['paths']
                agent.explorer.set_num_paths(len(path_indices))
                X_torch = torch.as_tensor(X, dtype=torch.float32, device=agent.device)
                edge_index_torch = torch.as_tensor(edge_index, dtype=torch.long, device=agent.device)

                num_nodes = X_torch.size(0)
                max_index = edge_index_torch.max().item()
                if max_index >= num_nodes:
                    print(f"[ERROR] edge_index contains invalid index {max_index} >= num_nodes {num_nodes}")
                    print("edge_index:\n", edge_index_torch)
                    print("X_torch.shape:", X_torch.shape)
                    raise ValueError("Invalid edge_index: index out of bounds")

                agent.explorer._Explorer__num_paths = [len(path_indices)]
                weights = agent.select_action(X_torch, edge_index_torch, path_indices)
                reward, next_X, next_edge_index, next_path_indices, done = env.step(region_id, src, dst, weights)
                with torch.no_grad():
                    h = agent.actor.gcn1(X_torch, edge_index_torch)
                    state_dict = {
                        'x': X_torch,
                        'edge_index': edge_index_torch,
                        'path_indices': path_indices
                    }
                    next_state = {
                        'x': next_X.detach().clone().float().to(agent.device),
                        'edge_index': torch.tensor(next_edge_index, dtype=torch.long, device=agent.device),
                        'path_indices': next_path_indices
                    }
                agent.store(state_dict, weights, reward, next_state, done)
                total_reward += reward
                update_count += 1
                if len(agent.buffer) >= warmup_steps and update_count % update_freq == 0:
                    agent.update(MINI_BATCH)
                if update_count % 1000 == 0:
                    print("update_count:", update_count//update_freq, "  reward:", reward)
                if done:
                    break
        print(f"[Episode {episode + 1}/{MAX_EPISODES}] total_reward={total_reward:.3f}")
    print("==== agent1_gcn 训练结束 ====")
    checkpoint_path = os.path.join(dirCkpoint, 'gcn_ddpg_agent.pt')
    agent.save(checkpoint_path)
    print(f"模型已保存到 {checkpoint_path}")
    fileUtilOut.close()
    if fileEdgeOut is not None:
        fileEdgeOut.close()
    timeRecord.append(time.time())
    log_time_file(timeRecord, dirLog)
    sys.exit(0)



from drlAgent import *
from flag import FLAGS
IS_TRAIN = getattr(FLAGS, "is_train")
RWD_FLAG = getattr(FLAGS, "rwd_flag")
ACTOR_LEARNING_RATE = getattr(FLAGS, 'learning_rate_actor')
CRITIC_LEARNING_RATE = getattr(FLAGS, 'learning_rate_critic')

GAMMA = getattr(FLAGS, 'gamma')
TAU = getattr(FLAGS, 'tau')

EP_BEGIN = getattr(FLAGS, 'epsilon_begin')
EP_END = getattr(FLAGS, 'epsilon_end')

BUFFER_SIZE = getattr(FLAGS, 'size_buffer')
MINI_BATCH = getattr(FLAGS, 'mini_batch')

MAX_EPISODES = getattr(FLAGS, 'episodes')
MAX_EP_STEPS = getattr(FLAGS, 'epochs')

if getattr(FLAGS, 'stamp_type') == '':
    REAL_STAMP = str(datetime.datetime.now())
else:
    REAL_STAMP = getattr(FLAGS, 'stamp_type')
PATHPRE = getattr(FLAGS, 'path_pre')

AGENT_TYPE = getattr(FLAGS, "agent_type")

CKPT_PATH = getattr(FLAGS, "ckpt_path")
TOPO_NAME = getattr(FLAGS, "topo_name")
SYNT_TYPE = getattr(FLAGS, "synthesis_type")
PATH_TYPE = getattr(FLAGS, "path_type")
FAILURE_FLAG = getattr(FLAGS, "failure_flag")
START_INDEX = getattr(FLAGS, "train_start_index")
SMALL_RATIO = getattr(FLAGS, "small_ratio")
BLOCK_NUM = getattr(FLAGS, "block_num")


initActions = []
def init_action(regionNum, pathNumList):
    res = []
    for regionId in range(regionNum):
        action = []
        for item in pathNumList[regionId]:
            action += [round(1.0/item, 6) for i in range(item)]
        res.append(action)
    return res


def update_step(maxutil, maxutilList, netutilList, agents, regionNodeNeibor, actionBorderline):
    if AGENT_TYPE == "MDA":
        actions = []
        agentNum = len(agents)
        if RWD_FLAG == 0:
            for agentid in range(agentNum):
                state = netutilList[agentid//2]
                if agentid % 2 == 0:
                    reward = -1*maxutilList[agentid//2]
                    result = agents[agentid].predict(state, reward)
                else:
                    maxutil_nei = [maxutilList[nrid] for nrid in regionNodeNeibor[agentid//2]]
                    if len(maxutil_nei) == 0:
                        reward = -0.7*maxutilList[agentid//2]
                    else:
                        reward = -0.7*maxutilList[agentid//2] - 0.3*sum(maxutil_nei)/len(maxutil_nei)
                    result = agents[agentid].predict(state, reward)
                ret_c = list(result)
                actions.append(ret_c)
        else:
            print("reward flag ERROR")
            exit()
        return actions
    
    elif AGENT_TYPE == "MSA":
        actions = []
        agentNum = len(agents)
        if RWD_FLAG == 0:
            for agentid in range(agentNum):
                state = netutilList[agentid]
                reward = -1*maxutilList[agentid]
                result = agents[agentid].predict(state, reward)
                ret_c = list(result)
                actions.append(ret_c[:actionBorderline[agentid]])
                actions.append(ret_c[actionBorderline[agentid]:])
        elif RWD_FLAG == 1:
            for agentid in range(agentNum):
                state = netutilList[agentid]
                maxutil_nei = [maxutilList[nrid] for nrid in regionNodeNeibor[agentid]]
                reward = -0.7*maxutilList[agentid] - 0.3*sum(maxutil_nei)/len(maxutil_nei)
                result = agents[agentid].predict(state, reward)
                ret_c = list(result)
                actions.append(ret_c[:actionBorderline[agentid]])
                actions.append(ret_c[actionBorderline[agentid]:])
        else:
            print("reward flag ERROR")
            exit()
        return actions

    elif AGENT_TYPE == "ECMP":
        return initActions
    else:
        pass

def init_multi_agent(globalSess):
    env = Environment(PATHPRE, TOPO_NAME, MAX_EPISODES, MAX_EP_STEPS, START_INDEX, IS_TRAIN, PATH_TYPE, SYNT_TYPE, SMALL_RATIO, FAILURE_FLAG, BLOCK_NUM)

    regionNum, edgeNumList, pathNumListDual, regionNodeNeibor = env.get_info()
    if AGENT_TYPE == "MDA":
        print("\nConstructing MDA multiple agents ...")
        agents = []
        for regionId in range(regionNum):
            print("Region%d .." % regionId)
            dimState = edgeNumList[regionId]
            dimAction = sum(pathNumListDual[regionId][0])
            agent = DrlAgent(globalSess, IS_TRAIN, dimState, dimAction, pathNumListDual[regionId][0], ACTOR_LEARNING_RATE, CRITIC_LEARNING_RATE, TAU, BUFFER_SIZE, MINI_BATCH, EP_BEGIN, EP_END, GAMMA, MAX_EP_STEPS)
            agents.append(agent)

            dimState = edgeNumList[regionId]
            dimAction = sum(pathNumListDual[regionId][1])
            agent = DrlAgent(globalSess, IS_TRAIN, dimState, dimAction, pathNumListDual[regionId][1], ACTOR_LEARNING_RATE, CRITIC_LEARNING_RATE, TAU, BUFFER_SIZE, MINI_BATCH, EP_BEGIN, EP_END, GAMMA, MAX_EP_STEPS)
            agents.append(agent)
        
        # parameters init  
        print("Running global_variables initializer ...")
        globalSess.run(tf.global_variables_initializer())
        
        # build target actor and critic para
        print("Building target network ...")
        for agentid in range(len(agents)):
            agents[agentid].target_paras_init()
        
        # parameters restore
        mSaver = tf.train.Saver(tf.trainable_variables()) 
        if CKPT_PATH != None and CKPT_PATH != "":
            print("restore paramaters...")
            mSaver.restore(globalSess, CKPT_PATH)
        
        initActions = []
        for regionId in range(regionNum):
            initActions += init_action(2, pathNumListDual[regionId])
        return env, agents, initActions, regionNodeNeibor, []

    elif AGENT_TYPE == "MSA":
        print("\nConstructing MSA agents ...")
        agents = []
        actionBorderline = []
        for regionId in range(regionNum):
            print("Region%d .." % regionId)
            dimState = edgeNumList[regionId]
            dimAction = sum(pathNumListDual[regionId][0]) + sum(pathNumListDual[regionId][1])
            agent = DrlAgent(globalSess, IS_TRAIN, dimState, dimAction, pathNumListDual[regionId][0] + pathNumListDual[regionId][1], ACTOR_LEARNING_RATE, CRITIC_LEARNING_RATE, TAU, BUFFER_SIZE, MINI_BATCH, EP_BEGIN, EP_END, GAMMA, MAX_EP_STEPS)
            agents.append(agent)
            actionBorderline.append(sum(pathNumListDual[regionId][0]))
        
        # parameters init  
        print("Running global_variables initializer ...")
        globalSess.run(tf.global_variables_initializer())
        
        # build target actor and critic para
        print("Building target network ...")
        for agentid in range(len(agents)):
            agents[agentid].target_paras_init()
        
        # parameters restore
        mSaver = tf.train.Saver(tf.trainable_variables()) 
        if CKPT_PATH != None and CKPT_PATH != "":
            print("restore paramaters...")
            mSaver.restore(globalSess, CKPT_PATH)
        
        initActions = []
        for regionId in range(regionNum):
            initActions += init_action(2, pathNumListDual[regionId])
        return env, agents, initActions, regionNodeNeibor, actionBorderline
    elif AGENT_TYPE == "ECMP":
        initActions = []
        for regionId in range(regionNum):
            initActions += init_action(2, pathNumListDual[regionId])
        return env, [], initActions, regionNodeNeibor, []
    else:
        print("Scheme type error")
        exit()

def log_to_file(maxutil, fileUtilOut, netutilList, fileEdgeOut):
    print(maxutil, file=fileUtilOut)
    if fileEdgeOut != None:
        netutils = []
        for item in netutilList:
            netutils += item
        print(netutils, file=fileEdgeOut)

def init_output_file():
    dirLog = PATHPRE + "outputs/log/" + REAL_STAMP
    dirCkpoint = PATHPRE + "outputs/ckpoint/" + REAL_STAMP
    if not os.path.exists(dirLog):
        os.makedirs(dirLog, exist_ok=True)
    fileUtilOut = open(dirLog + '/util.log', 'w', 1)
    if IS_TRAIN and AGENT_TYPE != "ECMP":
        if not os.path.exists(dirCkpoint):
            os.makedirs(dirCkpoint, exist_ok=True)
    if not IS_TRAIN and AGENT_TYPE != "ECMP" and MAX_EPISODES == 1:
        fileEdgeOut = open(dirLog + '/edge.log', 'w', 1)
    else:
        fileEdgeOut = None
    return dirLog, dirCkpoint, fileUtilOut, fileEdgeOut

def log_time_file(timeRecord, dirLog):
    print('\n' + REAL_STAMP)
    logfile = open(dirLog + "/runtime.log", 'w')
    runtimeType = ["inital time", "training time", "running time"]
    timeRecordPair = [[timeRecord[0], timeRecord[1]], 
                    [timeRecord[1], timeRecord[2]], 
                    [timeRecord[0], timeRecord[3]]]
    for t in range(len(timeRecordPair)):
        start_time = timeRecordPair[t][0]
        end_time = timeRecordPair[t][1]
        interval = int((end_time-start_time)*1000)
        timeMs = interval%1000
        timeS = int(interval/1000)%60
        timeMin = int((interval/1000-timeS)/60)%60
        timeH = int(interval/1000)/3600
        runtime = "%dh-%dmin-%ds-%dms" % (timeH, timeMin, timeS, timeMs)
        print("%s: %s" % (runtimeType[t], runtime))
        logfile.write("%s: %s\n" % (runtimeType[t], runtime))
    logfile.close()

if __name__ == "__main__":
    '''initial part'''
    print("\n----Information list----")
    print("agent_type: %s" % (AGENT_TYPE))
    print("stamp_type: %s" % (REAL_STAMP))
    timeRecord = []
    timeRecord.append(time.time())
    dirLog, dirCkpoint, fileUtilOut, fileEdgeOut = init_output_file()
    config = tf.ConfigProto(intra_op_parallelism_threads = 10)
    globalSess = tf.Session(config = config)
    env, agents, initActions, regionNodeNeibor, actionBorderline = init_multi_agent(globalSess)
    env.show_info()
    timeRecord.append(time.time())

    update_count = 0
    routing = initActions
    for _ in range(MAX_EPISODES * MAX_EP_STEPS):
        maxutil, maxutilList, netutilList = env.update(routing)
        log_to_file(maxutil, fileUtilOut, netutilList, fileEdgeOut)
        routing = update_step(maxutil, maxutilList, netutilList, agents, regionNodeNeibor, actionBorderline)
        if update_count % 1000 == 0:
            print("update_count:", update_count, "  max_util:", maxutilList)
        update_count += 1
    
    timeRecord.append(time.time())
    # store global variables
    if IS_TRAIN and AGENT_TYPE != "ECMP":
        print("saving checkpoint...")
        mSaver = tf.train.Saver(tf.global_variables())        
        mSaver.save(globalSess, dirCkpoint + "/ckpt")
        print("save checkpoint over")
    
    timeRecord.append(time.time())
    fileUtilOut.close()
    if fileEdgeOut != None:
        fileEdgeOut.close()
    log_time_file(timeRecord, dirLog)
