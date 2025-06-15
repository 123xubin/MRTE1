#!/usr/bin/python3
# -*- coding: UTF-8 -*-

'''
    Environment for offline training and test. 
'''

from __future__ import division
import copy
import os
import pickle
import numpy as np
import time
import multiprocessing as mp
import torch
#from MRTE.MRTE.drlte.lib.main import region_id


class Environment:
    def __init__(self, infile_prefix, topo_name, episode, epoch, start_index, train_flag, path_type, synthesis_type, small_ratio, failure_flag, block_num,
                 intra_type='default', inter_type='default', traffic_type='default'):
        # k-shortest path
        self.intra_paths = []  # 域内动态路径
        self.inter_paths = []  # 域间静态路径
        self.K = 5  # KSP参数
        self.M = 3  # 保留路径数
        self.path_update_interval = 10  # 路径更新间隔
        self.last_update_step = 0  # 最后更新时间步
        self.last_utilization = None
        self.__current_flowmap = None

        # store topo info
        self.__toponame = topo_name
        self.__nodenum = 0
        self.__linknum = 0
        self.__linkset = []
        self.__wMatrix = []
        self.__cMatrix = []
        self.__edgemap = []
        self.__regionwMatrix = [] # region-level
        self.__regionrMatrix = [] # region-level
        self.__regionnodeneibor = [] # region-level

        self.__regionedgenum = []
        self.__regionnum = 0
        self.__noderegionid = []
        self.__bordernodes = []

        # store demands
        self.__demands = []
        self.__smalldemidmap = []

        # store paths
        self.__oripathmaxtrix = []
        self.__interpathedgemap = []
        self.__actionrangemap = []

        # store rates
        self.__totalTMnum = 0
        self.__demrates = []
        self.__demrate = []
        self.__TM = []

        # train
        self.__start_index = start_index
        self.__epoch = epoch
        self.__episode = -1
        self.__maxepoch = episode * epoch
        self.__updatenum = 0
        self.__actionmatrix = []
        self.__smallratio = small_ratio
        self.__failure_flag = failure_flag
        self.__failurelink = None

        # multi process
        self.__procnum = 6
        self.__partitions = []

        # for grid topo
        self.__blockflag = False

        # file path
        # 获取当前文件（environment.py）的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 修正路径构建逻辑
        self.__topofile = os.path.join(current_dir, "inputs", "region", f"{topo_name}.txt")
        self.__pathfileintra = os.path.join(current_dir, "inputs", "pathset", "intra",
                                            f"{topo_name}_{intra_type}.pickle")
        self.__pathfileinter = os.path.join(current_dir, "inputs", "pathset", "inter",
                                            f"{topo_name}_{inter_type}.pickle")
        self.__ratefile = os.path.join(current_dir, "inputs", "traffic", f"{topo_name}_TMset_{traffic_type}.txt")

        # initial functions
        self.get_regions()
        # self.get_demands()
        self._build_node_idx_mapping()

        self.__oripathmaxtrix = [[[] for _ in range(self.__nodenum)] for _ in range(self.__nodenum)]

        # 判断是否为你自定义的域内KSP路径类型
        # environment.py 的 __init__ 方法中修改如下：
        if path_type == 'KSP':
            # 尝试加载预生成的静态路径
            with open(self.__pathfileintra, 'rb') as f:
                print(f"[INFO] 从静态文件加载域内路径")
                all_paths = pickle.load(f)
            self.intra_paths = [[[] for _ in range(self.__nodenum)] for _ in range(self.__nodenum)]
            for src in range(self.__nodenum):
                 for dst in range(self.__nodenum):
                    if self.__noderegionid[src] == self.__noderegionid[dst] and src != dst:
                        self.intra_paths[src][dst] = all_paths[src][dst]
            # 筛选 Top-M 路径
            self.dynamic_path_selection()
            self.load_static_inter_paths()  # 加载域间路径
        self.get_TMset()
        if self.__failure_flag == 1:
            self.__brokenlinkfile = infile_prefix + "inputs/brokenlink/" + topo_name + "_%dlinks.txt" % 100
            self.get_broken_link()

        self.dynamic_refresh_paths(force=True)



    def _build_node_idx_mapping(self):
        """构建所有节点编号映射表（节点号→索引，索引→节点号），便于TM/邻接矩阵等索引安全"""
        self.all_node_ids = list(range(self.__nodenum))
        self.node_id2idx = {nid: i for i, nid in enumerate(self.all_node_ids)}
        self.idx2node_id = {i: nid for i, nid in enumerate(self.all_node_ids)}
        # print("节点号→索引映射：", self.node_id2idx)

    def get_broken_link(self):
        filein = open(self.__brokenlinkfile, 'r')
        lines = filein.readlines()
        self.__brokenlinklist = []
        for line in lines:
            lineList = line.strip().split()
            self.__brokenlinklist.append(list(map(int, lineList[1:])))
        filein.close()

    def get_regions(self):
        file = open(self.__topofile)
        lines = file.readlines()
        file.close()
        lineList = lines[0].strip().split()
        self.__nodenum = int(lineList[0])
        self.__linknum = int(lineList[1])

        for i in range(self.__nodenum):
            self.__wMatrix.append([])
            self.__cMatrix.append([0.0] * self.__nodenum)
            self.__edgemap.append([-1] * self.__nodenum)
            for j in range(self.__nodenum):
                if i == j:
                    self.__wMatrix[i].append(0)
                else:
                    self.__wMatrix[i].append(999999)
        lineList = lines[self.__linknum + 1].strip().split()
        self.__noderegionid = list(map(int, lineList))
        self.__regionnum = max(self.__noderegionid) + 1
        self.__regionedgenum = [0] * self.__regionnum
        self.__bordernodes = [[] for _ in range(self.__regionnum)]
        for i in range(1, self.__linknum + 1):
            lineList = lines[i].strip().split()
            left = int(lineList[0]) - 1
            right = int(lineList[1]) - 1
            weight = int(lineList[2])
            capa = float(lineList[3])
            regionId = int(lineList[4])
            if regionId == -1:
                lRegion = self.__noderegionid[left]
                rRegion = self.__noderegionid[right]
                if left not in self.__bordernodes[lRegion]:
                    self.__bordernodes[lRegion].append(left)
                if right not in self.__bordernodes[rRegion]:
                    self.__bordernodes[rRegion].append(right)
            self.__linkset.append([left, right, weight, capa, regionId])
            self.__wMatrix[left][right] = weight
            self.__wMatrix[right][left] = weight
            self.__cMatrix[left][right] = capa
            self.__cMatrix[right][left] = capa
            self.__edgemap[left][right] = self.__noderegionid[left]
            self.__edgemap[right][left] = self.__noderegionid[right]
            self.__regionedgenum[self.__noderegionid[left]] += 1
            self.__regionedgenum[self.__noderegionid[right]] += 1
        for i in range(self.__regionnum):
            self.__regionwMatrix.append([])
            for j in range(self.__regionnum):
                if i == j:
                    self.__regionwMatrix[i].append(0)
                else:
                    self.__regionwMatrix[i].append(999999)
        self.__regionnodeneibor = [[] for _ in range(self.__regionnum)]
        for i in range(self.__linknum + 2, len(lines)):  # region-level
            lineList = lines[i].strip().split()
            left = int(lineList[0])
            right = int(lineList[1])
            self.__regionnodeneibor[left].append(right)
            self.__regionnodeneibor[right].append(left)
            self.__regionwMatrix[left][right] = 1
            self.__regionwMatrix[right][left] = 1
        self.__regionrMatrix = self.com_shr_path(self.__regionnum, self.__regionwMatrix)

    def com_shr_path(self, regionNum, wMatrix_ori): # region-level
        rMatrix = []
        for i in range(regionNum):
            rMatrix.append([j for j in range(regionNum)])
        wMatrix = copy.deepcopy(wMatrix_ori)
        for k in range(regionNum):
            for i in range(regionNum):
                for j in range(regionNum):
                    if wMatrix[i][j] > wMatrix[i][k] + wMatrix[k][j]:
                        wMatrix[i][j] = wMatrix[i][k] + wMatrix[k][j]
                        rMatrix[i][j] = rMatrix[i][k]
        return rMatrix

    def load_static_inter_paths(self):
        """从pickle文件加载域间路径"""
        with open(self.__pathfileinter, 'rb') as f:
            all_paths = pickle.load(f)
        self.inter_paths = [[[] for _ in range(self.__nodenum)] for _ in range(self.__nodenum)]
        for src in range(self.__nodenum):
            for dst in range(self.__nodenum):
                if self.__noderegionid[src] != self.__noderegionid[dst]:
                    self.inter_paths[src][dst] = all_paths[src][dst]

    def get_paths(self):
        # 生成域内KSP路径
        self.intra_paths = [[[] for _ in range(self.__nodenum)] for _ in range(self.__nodenum)]
        for src in range(self.__nodenum):
            for dst in range(self.__nodenum):
                if src != dst and self.__noderegionid[src] == self.__noderegionid[dst]:
                    self.intra_paths[src][dst] = self.generate_ksp(src, dst)
                    self.dynamic_path_selection()  # 动态筛选
        # 合并路径：动态域内 + 静态域间
        for src in range(self.__nodenum):
            for dst in range(self.__nodenum):
                if self.__noderegionid[src] == self.__noderegionid[dst]:
                    self.__oripathmaxtrix[src][dst] = self.intra_paths[src][dst]
                else:
                    self.__oripathmaxtrix[src][dst] = self.inter_paths[src][dst]

    def get_demands(self):
        for src in range(self.__nodenum):
            for dst in range(self.__nodenum):
                if src == dst:
                    continue
                self.__demands.append([src,dst])

    def com_path_flow(self, flowmap, pathSet, action, size):
        if len(action) != len(pathSet):
            if len(action) < len(pathSet):
                action = list(action) + [1.0 / max(1, len(pathSet))] * (len(pathSet) - len(action))
            else:
                action = list(action)[:len(pathSet)]
        subsizes = []
        gates = []
        pathNum = len(pathSet)
        tmp = 0
        src = pathSet[0][0]
        dst = pathSet[0][-1]
        gates.append(dst)
        for i in range(pathNum):
            length = len(pathSet[i])
            subsize = action[i]*size
            if dst != pathSet[i][-1]:
                dst = pathSet[i][-1]
                gates.append(dst)
                subsizes.append(tmp)
                tmp = 0
            tmp += subsize
            for j in range(length-1):
                node1 = pathSet[i][j]
                node2 = pathSet[i][j+1]
                flowmap[node1][node2] += subsize
        subsizes.append(tmp)
        return subsizes, gates

    def com_action_matrix(self, actionList):
        self.__actionmatrix = [[[] for _ in range(self.__nodenum)] for _ in range(self.__nodenum)]

        # 按区域处理动作
        act_counters = [0] * self.__regionnum  # 每个区域的动作计数器

        dem_id = 0
        for src in range(self.__nodenum):
            for dst in range(self.__nodenum):
                if src == dst:
                    continue

                s_region = self.__noderegionid[src]
                t_region = self.__noderegionid[dst]
                paths = self.__oripathmaxtrix[src][dst]
                path_count = len(paths)

                # 域内流量
                if s_region == t_region:
                    if self.__smalldemidmap[dem_id] == 0:  # 仅大流量使用智能体动作
                        # 确保有足够的动作
                        if act_counters[s_region] + path_count <= len(actionList[s_region * 2]):
                            action = actionList[s_region * 2][
                                     act_counters[s_region]:act_counters[s_region] + path_count]
                            act_counters[s_region] += path_count
                        else:  # 动作不足时使用回退策略
                            action = [1.0 / path_count] * path_count
                    else:
                        action = [1.0 / path_count] * path_count

                # 域间流量
                else:
                    action_range = self.__actionrangemap[src][t_region]
                    # 确保索引有效
                    start_idx = action_range[0]
                    end_idx = min(action_range[1], len(actionList[s_region * 2 + 1]))

                    if start_idx < end_idx:
                        action = actionList[s_region * 2 + 1][start_idx:end_idx]
                        # 确保动作长度匹配路径数
                        if len(action) < path_count:
                            action += [0.0] * (path_count - len(action))
                        action = action[:path_count]  # 截断多余动作
                    else:
                        action = [1.0 / path_count] * path_count

                # 归一化动作
                action_sum = sum(action)
                if action_sum > 0:
                    action = [a / action_sum for a in action]
                else:
                    action = [1.0 / path_count] * path_count

                self.__actionmatrix[src][dst] = action
                dem_id += 1

    def compute_flowmap(self):
        if self.__current_flowmap is None:
            self.__current_flowmap = np.zeros((self.__nodenum, self.__nodenum), dtype=np.float32)
        new_flowmap = np.copy(self.__current_flowmap)
        for src in range(self.__nodenum):
            for dst in range(self.__nodenum):
                if src == dst:
                    continue
                src_idx = self.node_id2idx[src]
                dst_idx = self.node_id2idx[dst]
                sources = [src]
                sizes = [self.__TM[src_idx][dst_idx]]
                while True:
                    if len(sources) == 0:
                        break
                    pathSet = self.__oripathmaxtrix[sources[0]][dst]
                    action = self.__actionmatrix[sources[0]][dst]
                    subsizes, gates = self.com_path_flow(new_flowmap, pathSet, action, sizes[0])
                    sources.pop(0)
                    sizes.pop(0)
                    for gwid in range(len(gates)):
                        if gates[gwid] == dst:
                            continue
                        if subsizes[gwid] <= 0.001:
                            continue
                        sources.append(gates[gwid])
                        sizes.append(subsizes[gwid])
        self.__current_flowmap = new_flowmap
        return new_flowmap

    def process_com_flowmap_pipe(self, start, end, conn):
        TM = self.__TM
        while True:
            try:
                msg = conn.recv()
                if msg[0] == 0:
                    actionmatrix = msg[1]
                elif msg[0] == 1:
                    TM = msg[1]
                    continue
                else:
                    pass
                flowmap = []
                for _ in range(self.__nodenum):
                    flowmap.append([0.0]*self.__nodenum)
                for src in range(self.__nodenum):
                    for dst in range(start, end):
                        if src == dst:
                            continue
                        sources = [src]
                        sizes = [TM[src][dst]]
                        while True:
                            if len(sources) == 0:
                                break
                            pathSet = self.__oripathmaxtrix[sources[0]][dst]
                            action = actionmatrix[sources[0]][dst]
                            subsizes, gates = self.com_path_flow(flowmap, pathSet, action, sizes[0])
                            sources.pop(0)
                            sizes.pop(0)
                            for gwid in range(len(gates)):
                                if gates[gwid] == dst:
                                    continue
                                if subsizes[gwid] <= 0.001:
                                    continue
                                sources.append(gates[gwid])
                                sizes.append(subsizes[gwid])
                conn.send(flowmap)
            except EOFError:
                break

    def compute_flowmap_paralell(self): # for large topo
        if self.__updatenum == 0:
            step = self.__nodenum//self.__procnum
            self.__partitions = [procid*step for procid in range(self.__procnum)] + [self.__nodenum]
            pool = mp.Pool(self.__procnum)
            self.__connList = []
            for procid in range(self.__procnum):
                parent_conn, child_conn = mp.Pipe()
                self.__connList.append(parent_conn)
                pool.apply_async(self.process_com_flowmap_pipe, args = (self.__partitions[procid], self.__partitions[procid+1], child_conn, ))

        if self.__updatenum%self.__epoch == 0:
            for parent_conn in self.__connList:
                parent_conn.send((1, self.__TM))

        for parent_conn in self.__connList:
            parent_conn.send((0, self.__actionmatrix))

        flowmapList = []
        for parent_conn in self.__connList:
            flowmapList.append(parent_conn.recv())

        if self.__updatenum == self.__maxepoch - 1:
            for parent_conn in self.__connList:
                parent_conn.close()

        flowmap = []
        for _ in range(self.__nodenum):
            flowmap.append([0.0]*self.__nodenum)
        for src in range(self.__nodenum):
            for dst in range(self.__nodenum):
                for flowmap_sub in flowmapList:
                    flowmap[src][dst] += flowmap_sub[src][dst]

        return flowmap

    def compute_util(self, flowmap):
        maxutilList = []
        netutilList = [[] for _ in range(self.__regionnum)]
        if self.__failure_flag == 0:
            for i in range(self.__nodenum):
                for j in range(self.__nodenum):
                    if self.__edgemap[i][j] >= 0:
                        util = round(flowmap[i][j]/self.__cMatrix[i][j], 4)
                        regionId = self.__edgemap[i][j]
                        netutilList[regionId].append(util)
            for netutil in netutilList:
                maxutilList.append(max(netutil))
            return maxutilList, netutilList
        else: # failure
            netutilList2 = [[] for _ in range(self.__regionnum)]
            for i in range(self.__nodenum):
                for j in range(self.__nodenum):
                    if self.__edgemap[i][j] >= 0:
                        regionId = self.__edgemap[i][j]
                        if (i,j) == self.__failurelink or (j,i) == self.__failurelink:
                            util = 1.0
                        else:
                            util = round(flowmap[i][j]/self.__cMatrix[i][j], 4)
                            netutilList2[regionId].append(util)
                        netutilList[regionId].append(util)
            for netutil in netutilList2:
                maxutilList.append(max(netutil))
            return maxutilList, netutilList

    def convert_block_action(self, actions):
        actionsSplit = [[] for _ in range(2*self.__regionnum)]
        for bid in range(self.__blocknum):
            for index in range(len(self.__blockrule[bid])):
                for agentType in range(2):
                    start = self.__actionBorderInBlock[bid][agentType][index]
                    end = self.__actionBorderInBlock[bid][agentType][index + 1]
                    actionsSplit[self.__blockrule[bid][index]*2+agentType] = actions[bid*2+agentType][start:end]
        return actionsSplit

    def convert_block_util(self, maxutilList, netutilList):
        maxutilListMerge = [0]*self.__blocknum
        netutilListMerge = [[] for _ in range(self.__blocknum)]
        for bid in range(self.__blocknum):
            for rid in self.__blockrule[bid]:
                maxutilListMerge[bid] = max([maxutilListMerge[bid], maxutilList[rid]])
                netutilListMerge[bid] += netutilList[rid]
        return maxutilListMerge, netutilListMerge

    def update(self, actions):
        if self.__updatenum % self.__epoch == 0:
            self.__episode += 1
            self.change_TM()
        if self.__blockflag:
            actions = self.convert_block_action(actions)
        self.com_action_matrix(actions)
        if self.__toponame == "briten12r16grid": # for large topo
            flowmap = self.compute_flowmap_paralell()
        else:
            flowmap = self.compute_flowmap()

        maxutilList, netutilList = self.compute_util(flowmap)
        # 事件驱动路径更新
        if self.check_network_state_change(maxutilList):
            print(f"[Update] Regenerating paths at step {self.__updatenum}")
            self.get_paths()  # 重新生成KSP并筛选
            self.last_utilization = maxutilList

        if self.__blockflag:
            maxutilList, netutilList = self.convert_block_util(maxutilList, netutilList)
        self.__updatenum += 1
        return max(maxutilList), maxutilList, netutilList

    def action_failure(self, actionList):
        failureLinkIndex = self.__episode*10 + (self.__updatenum%self.__epoch)//5
        left = self.__brokenlinklist[failureLinkIndex][0]
        right = self.__brokenlinklist[failureLinkIndex][1]
        self.__failurelink = (left, right)

        # rescale, self.__act2edgepath
        newActionList = []
        for agentId in range(len(actionList)):
            action = actionList[agentId]
            newAction = []
            count = 0
            for i in range(len(self.__act2edgepath[agentId])):
                pathNum = len(self.__act2edgepath[agentId][i])
                edgepaths = self.__act2edgepath[agentId][i]
                newAction += self.rescale_action(pathNum, edgepaths, action[count:count+pathNum])
                count += pathNum
            newActionList.append(newAction)
        return newActionList

    def rescale_action(self, pathNum, edgepaths, subaction):
        action_tmp = []
        action_flag = []
        split_more = 0.0
        (left, right) = self.__failurelink
        for j in range(pathNum):
            if (left, right) in edgepaths[j] or (right, left) in edgepaths[j]:
                action_tmp.append(0.0)
                action_flag.append(0)
                split_more += subaction[j]
            else:
                action_flag.append(1)
                action_tmp.append(subaction[j])

        sums = 0.0
        for i in range(len(action_flag)):
            if action_flag[i] == 1:
                sums += action_tmp[i]

        res = []
        if sum(action_flag) == 0:
            print(self.__updatenum, self.__failurelink, edgepaths)
            exit()
        if sums <= 0.0001:
            w = 1.0/sum(action_flag)
            for i in range(len(action_flag)):
                if action_flag[i] == 1:
                    res.append(w)
                else:
                    res.append(0.0)
        else:
            for i in range(len(action_flag)):
                if action_flag[i] == 1:
                    res.append(action_tmp[i] + (action_tmp[i]/sums)*split_more)
                else:
                    res.append(0.0)
        return res

    def validate_correctness(self, flowmap):
        for row in self.__TM:
            for item in row:
                print("%5s  " % str(round(item, 1)), end='')
            print('\n')
        for src in range(self.__nodenum):
            if src == 0:
                print("%5s  " % str(0), end = '')
                for dst in range(self.__nodenum):
                    print("%5s  " % str(dst), end='')
                print('\n')
            print("%5s  " % str(src), end = '')
            for dst in range(self.__nodenum):
                print("%5s  " % str(round(flowmap[src][dst], 0)), end='')
            print('\n')

    def get_TMset(self):
        file = open(self.__ratefile)
        lines = file.readlines()
        file.close()
        self.__totalTMnum = len(lines)
        for i in range(self.__totalTMnum):
            lineList = lines[i].strip().split(',')
            rates = list(map(float, lineList))
            self.__demrates.append(rates)

    def change_TM(self):
        self.__demrate = self.__demrates[(self.__episode + self.__start_index) % self.__totalTMnum]
        self.__TM = []
        demId = 0
        for i in range(self.__nodenum):
            self.__TM.append([0.0]*self.__nodenum)
            for j in range(self.__nodenum):
                if i == j:
                    continue
                self.__TM[i][j] = self.__demrate[demId]
                demId += 1

    def set_TM(self, s, t, size):
        self.__TM = []
        for i in range(self.__nodenum):
            self.__TM.append([0.0]*self.__nodenum)
        self.__TM[s][t] = size

    def show_info(self):
        print("--------------------------")
        print("----detail information----")
        print("topology:%s(%d,%d) with %d region(s)" % (self.__toponame, self.__nodenum, self.__linknum, self.__regionnum))
        print("--------------------------")

    def cal_terminal_demands(self, TM):
        ternimalTM = copy.deepcopy(TM)

        actionmatrix = []
        for src in range(self.__nodenum):
            actionmatrix.append([])
            for dst in range(self.__nodenum):
                if src == dst:
                    actionmatrix[src].append([])
                else:
                    action = [1.0/len(self.__oripathmaxtrix[src][dst])]*len(self.__oripathmaxtrix[src][dst])
                    actionmatrix[src].append([round(item, 6) for item in action])

        flowmap = []
        for _ in range(self.__nodenum):
            flowmap.append([0.0]*self.__nodenum)
        for src in range(self.__nodenum):
            for dst in range(self.__nodenum):
                srid = self.__noderegionid[src]
                trid = self.__noderegionid[dst]
                if srid == trid:
                    continue
                sources = [src]
                sizes = [TM[src][dst]]
                ingressNodes = {}
                while True:
                    if len(sources) == 0:
                        break
                    pathSet = self.__oripathmaxtrix[sources[0]][dst]
                    action = actionmatrix[sources[0]][dst]
                    subsizes, gates = self.com_path_flow(flowmap, pathSet, action, sizes[0])
                    sources.pop(0)
                    sizes.pop(0)
                    for gwid in range(len(gates)):
                        if gates[gwid] == dst:
                            continue
                        sources.append(gates[gwid])
                        sizes.append(subsizes[gwid])
                        if self.__noderegionid[gates[gwid]] == trid:
                            if gates[gwid] not in ingressNodes:
                                ingressNodes[gates[gwid]] = subsizes[gwid]
                            else:
                                ingressNodes[gates[gwid]] += subsizes[gwid]
                for ingress in ingressNodes.keys():
                    ternimalTM[ingress][dst] += ingressNodes[ingress]

        return ternimalTM

    def sort_intra_demand(self, aveNum = 40):
        # 1. get average demand rates
        demandNum = len(self.__demrates[0])
        demrate = np.array([0]*demandNum)
        for i in range(aveNum):
            rate = np.array(self.__demrates[i])
            demrate = demrate + rate
        demrate /= aveNum
        TM = []
        demId = 0
        for i in range(self.__nodenum):
            TM.append([0.0]*self.__nodenum)
            for j in range(self.__nodenum):
                if i == j:
                    continue
                TM[i][j] = demrate[demId]
                demId += 1
        ternimalTM = self.cal_terminal_demands(TM)

        # 2. get region demand rates
        regionRates = [[] for _ in range(self.__regionnum)]
        regionDemIds = [[] for _ in range(self.__regionnum)]
        totalTraffic_tmp = [[0, 0] for _ in range(self.__regionnum)]
        demId = 0
        for src in range(self.__nodenum):
            for dst in range(self.__nodenum):
                if src == dst:
                    continue
                sRegion = self.__noderegionid[src]
                tRegion = self.__noderegionid[dst]
                if sRegion == tRegion:
                    regionRates[sRegion].append(ternimalTM[src][dst])
                    regionDemIds[sRegion].append(demId)
                    totalTraffic_tmp[sRegion][0] += ternimalTM[src][dst]
                else:
                    totalTraffic_tmp[sRegion][1] += ternimalTM[src][dst]
                demId += 1
        # 3. sort region's demands
        smallDemIdMap = [0]*demandNum
        for rid in range(self.__regionnum):
            index = np.argsort(regionRates[rid])
            res = [round(regionRates[rid][i], 0) for i in index]
            index = index[:int(len(regionRates[rid])*self.__smallratio)]
            # print("small demand num", int(len(regionRates[rid])), len(index))
            for i in index:
                smallDemIdMap[regionDemIds[rid][i]] = 1

        return smallDemIdMap

    def convert_edge_paths(self, paths):
        edgepaths = []
        for path in paths:
            pathLen = len(path)
            epath = []
            for l in range(pathLen-1):
                epath.append((path[l], path[l+1]))
            edgepaths.append(epath)
        return edgepaths

    def get_info(self):
        self.__smalldemidmap = self.sort_intra_demand()

        pathNumListDual = [[[],[]] for _ in range(self.__regionnum)]
        pathNumMapRegion = []
        for _ in range(self.__nodenum):
            pathNumMapRegion.append([0]*self.__regionnum)

        # failure
        self.__act2edgepath = [[] for _ in range(self.__regionnum*2)]
        edgepathsMapRegion = []
        for i in range(self.__nodenum):
            edgepathsMapRegion.append([])
            for _ in range(self.__regionnum):
                edgepathsMapRegion[i].append([])

        demId = 0
        for src in range(self.__nodenum):
            for dst in range(self.__nodenum):
                if src == dst:
                    continue
                sRegion = self.__noderegionid[src]
                tRegion = self.__noderegionid[dst]
                if sRegion == tRegion:
                    if self.__smalldemidmap[demId] == 0:
                        if len(pathNumListDual[sRegion]) == 0:
                            pathNumListDual[sRegion].append([])
                            pathNumListDual[sRegion].append([])
                        pathNumListDual[sRegion][0].append(len(self.__oripathmaxtrix[src][dst]))
                        edgepaths = self.convert_edge_paths(self.__oripathmaxtrix[src][dst])
                        self.__act2edgepath[sRegion*2].append(edgepaths)
                else:
                    pathNumMapRegion[src][tRegion] = len(self.__oripathmaxtrix[src][dst])
                    edgepaths = self.convert_edge_paths(self.__oripathmaxtrix[src][dst])
                    edgepathsMapRegion[src][tRegion] = edgepaths
                demId += 1

        actionRangeMap = []
        for src in range(self.__nodenum):
            actionRangeMap.append([])
            for rid in range(self.__regionnum):
                actionRangeMap[src].append([])

        actCountList = [0]*self.__regionnum
        for src in range(self.__nodenum):
            sRegion = self.__noderegionid[src]
            for tRegion in range(self.__regionnum):
                if sRegion == tRegion:
                    continue
                if self.__regionrMatrix[sRegion][tRegion] != tRegion:  # region-level
                    continue
                pathNumListDual[sRegion][1].append(pathNumMapRegion[src][tRegion])
                self.__act2edgepath[sRegion*2+1].append(edgepathsMapRegion[src][tRegion]) # failure
                actionRangeMap[src][tRegion] = [actCountList[sRegion], actCountList[sRegion]+pathNumMapRegion[src][tRegion]]
                actCountList[sRegion] += pathNumMapRegion[src][tRegion]

        for src in range(self.__nodenum):
            sRegion = self.__noderegionid[src]
            for tRegion in range(self.__regionnum):
                if sRegion == tRegion:
                    continue
                if self.__regionrMatrix[sRegion][tRegion] != tRegion:
                    nextRegionHop = self.__regionrMatrix[sRegion][tRegion]
                    actionRangeMap[src][tRegion] = actionRangeMap[src][nextRegionHop]
        self.__actionrangemap = actionRangeMap

        print("regionedgenum:", self.__regionedgenum)
        print("actionDim:", [(sum(item[0]), sum(item[1])) for item in pathNumListDual])
        # print("pathNumListDual:", pathNumListDual)
        # print("regionNodeNeibor:", self.__regionnodeneibor)
        if not self.__blockflag:
            return self.__regionnum, self.__regionedgenum, pathNumListDual, self.__regionnodeneibor

        print("\nBlock Block Block")
        blockNum = len(self.__blockrule)
        self.__blocknum = blockNum
        regionEdgeNum = [0]*blockNum
        pathNumListDualBlock = []
        regionNodeNeibor = [[] for _ in range(blockNum)]
        ridMap = [0]*self.__regionnum
        self.__actionBorderInBlock = []
        for bid in range(blockNum):
            pathNumListDualBlock.append([[], []])
            self.__actionBorderInBlock.append([[0], [0]])
            for rid in self.__blockrule[bid]:
                regionEdgeNum[bid] += self.__regionedgenum[rid]
                pathNumListDualBlock[bid][0] += pathNumListDual[rid][0]
                pathNumListDualBlock[bid][1] += pathNumListDual[rid][1]
                self.__actionBorderInBlock[bid][0].append(sum(pathNumListDualBlock[bid][0]))
                self.__actionBorderInBlock[bid][1].append(sum(pathNumListDualBlock[bid][1]))
                ridMap[rid] = bid

        for bid in range(blockNum):
            for rid in self.__blockrule[bid]:
                for nrid in self.__regionnodeneibor[rid]:
                    if ridMap[nrid] not in regionNodeNeibor[bid] and ridMap[nrid] != bid:
                        regionNodeNeibor[bid].append(ridMap[nrid])
        print("regionEdgeNum:", regionEdgeNum)
        print("regionNodeNeibor:", regionNodeNeibor)
        print("self.__actionBorderInBlock", self.__actionBorderInBlock)
        return blockNum, regionEdgeNum, pathNumListDualBlock, regionNodeNeibor

    def encode_node_features(self, region_nodes, src, dst, current_flowmap):
        """编码节点特征（需在调用前预先计算flowmap）"""
        features = []
        for v in region_nodes:
            # 基础拓扑特征
            degree = sum(1 for u in region_nodes if self.__wMatrix[v][u] < 999999 and u != v)
            is_border = int(v in self.__bordernodes[self.__noderegionid[v]])
            is_source = int(v == src)
            is_dest = int(v == dst)

            # 流量特征（从当前flowmap获取）
            out_util = [current_flowmap[v][u] / self.__cMatrix[v][u] for u in region_nodes if
                        self.__edgemap[v][u] != -1]
            in_util = [current_flowmap[u][v] / self.__cMatrix[u][v] for u in region_nodes if self.__edgemap[u][v] != -1]
            avg_out_util = np.mean(out_util) if out_util else 0.0
            avg_in_util = np.mean(in_util) if in_util else 0.0

            # 特征向量
            features.append([
                degree / 10.0,  # 归一化节点度
                is_source,  # 是否为流量起点
                is_dest,  # 是否为流量终点
                is_border,  # 是否为边界节点
                avg_out_util,  # 流出链路平均利用率
                avg_in_util  # 流入链路平均利用率
            ])
        return np.array(features, dtype=np.float32)

    def encode_edge_features(self, region_edges, current_flowmap):
        """编码边特征（含方向性）"""
        edge_features = []
        edge_index = [[], []]
        for (u, v) in region_edges:
            # 正向边特征
            capacity = self.__cMatrix[u][v]
            used = current_flowmap[u][v]
            util = used / capacity if capacity > 0 else 0.0
            delay = self.__wMatrix[u][v]  # 假设weight代表时延

            edge_features.append([
                capacity / 1e6,  # 归一化带宽（假设单位是Mbps）
                util,  # 当前利用率
                delay / 100.0  # 归一化时延
            ])
            edge_index[0].append(u)
            edge_index[1].append(v)

            # 反向边特征（如果存在）
            if self.__wMatrix[v][u] < 999999:
                reverse_used = current_flowmap[v][u]
                reverse_util = reverse_used / self.__cMatrix[v][u] if self.__cMatrix[v][u] > 0 else 0.0
                edge_features.append([
                    self.__cMatrix[v][u] / 1e6,
                    reverse_util,
                    self.__wMatrix[v][u] / 100.0
                ])
                edge_index[0].append(v)
                edge_index[1].append(u)

        return np.array(edge_features, dtype=np.float32), np.array(edge_index, dtype=np.int64)

    def get_gcn_input(self, region_id, src, dst, current_flowmap):
        current_flowmap = self.__current_flowmap if self.__current_flowmap is not None else np.zeros((self.__nodenum, self.__nodenum))
        # 获取区域内全局节点列表
        region_nodes = [i for i in range(self.__nodenum) if self.__noderegionid[i] == region_id]
        if not region_nodes or src not in region_nodes or dst not in region_nodes:
            return None
        node_global_to_local = {gid: lid for lid, gid in enumerate(region_nodes)}  # 全局ID到局部ID的映射

        # 生成节点和边特征
        X = self.encode_node_features(region_nodes, src, dst, current_flowmap)
        region_edges = [(u, v) for u in region_nodes for v in region_nodes
                        if u != v and self.__wMatrix[u][v] < 999999]
        E, edge_index = self.encode_edge_features(region_edges, current_flowmap)

        # 将全局 edge_index 转换为局部索引
        local_edge_index = []
        for i in range(edge_index.shape[1]):
            u, v = edge_index[:, i]
            if u in node_global_to_local and v in node_global_to_local:
                local_edge_index.append([node_global_to_local[u], node_global_to_local[v]])

        edge_index = np.array(local_edge_index, dtype=np.int64).T if local_edge_index else np.empty((2, 0), dtype=np.int64)
        # 路径处理：转换全局节点ID为局部ID
        path_indices = []
        for path in self.intra_paths[src][dst]:
            if not self.validate_intra_path(path):
                continue

                # 转换全局节点ID为局部ID
            try:
                local_path = [node_global_to_local[n] for n in path]
                path_indices.append(local_path)
            except KeyError as e:
                print(f"路径节点 {e} 不在当前区域 {region_id} 中，已过滤")
                continue
        return {
            'x': torch.tensor(X, dtype=torch.float),
            'edge_index': torch.tensor(edge_index, dtype=torch.long),
            'paths': path_indices
        }

    def get_current_flow_util(self, u, v):
        """获取指定链路的实时利用率"""
        if self.__current_flowmap is None:
            return 0.0
        return self.__current_flowmap[u][v] / self.__cMatrix[u][v] if self.__cMatrix[u][v] > 0 else 0.0

    def get_region_nodes(self, region_id):
        """获取指定区域的全局节点ID列表"""
        return [i for i in range(self.__nodenum) if self.__noderegionid[i] == region_id]

    def reset(self):
        """
        重置环境到初始状态，用于新一轮 episode 的训练。
        建议每次 episode 都调用 reset 以确保环境独立性。
        """
        # 1. 重置仿真时钟
        self.time_step = 0

        # 2. 重新加载或抽样流量矩阵
        if hasattr(self, "TMset") and self.TMset is not None:
            # 选择新的一组 TM（如有多帧流量，则按 episode 切换或随机）
            if isinstance(self.TMset, list) and len(self.TMset) > 0:
                self.current_TM = self.TMset[0]  # 或随机抽样
            else:
                self.current_TM = self.TMset
        # 3. 重置链路利用率/负载
        if hasattr(self, "edge_utils"):
            self.edge_utils = [0.0 for _ in self.edge_utils]  # 全0，或按初始状态

        # 4. 恢复初始拓扑、断链、故障等（如有断链机制）
        if hasattr(self, "link_failures"):
            self.link_failures = [False for _ in self.link_failures]

        self.__actionmatrix = []
        for src in range(self.__nodenum):
            self.__actionmatrix.append([])
            for dst in range(self.__nodenum):
                self.__actionmatrix[src].append([])
        # 5. 其它自定义变量（如区域流、调度策略等）
        self.change_TM()
        return

    def region_flows(self):
        """
        遍历每个 region 下的 (src, dst) 对。
        这里举例只返回每个 region 的所有 border node 间的流（你可以根据自己的业务逻辑调整）。
        """
        for region_id in range(self.__regionnum):
            # 找到本 region 的所有 border node 作为 src
            region_nodes = self.get_region_nodes(region_id)
            # 示例：所有不同的 node 对
            for src in region_nodes:
                for dst in region_nodes:
                    if src != dst:
                        yield region_id, (src, dst)

    def step(self, region_id, src, dst, weights):
        """
        轻量化 step 方法：基于当前 action (路径权重)，构造流量图，计算 reward 和下一个 GCN 输入。
        """
        # 将 action 应用于当前 (src, dst)
        path_set = self.__oripathmaxtrix[src][dst]
        if len(path_set) != len(weights):
            raise ValueError(f"路径数量与权重不匹配: {len(path_set)} vs {len(weights)}")

        # 1. 构造最小 action matrix，仅更新当前 src-dst 对
        self.__actionmatrix[src][dst] = weights

        # 2. 生成新的 flowmap
        flowmap = self.compute_flowmap()

        # 3. 计算最大链路利用率作为 reward
        max_util = 0.0
        for i in range(self.__nodenum):
            for j in range(self.__nodenum):
                if self.__edgemap[i][j] >= 0:
                    util = flowmap[i][j] / self.__cMatrix[i][j] if self.__cMatrix[i][j] > 0 else 0.0
                    if util > max_util:
                        max_util = util

        reward = -max_util  # 越小越好

        # 4. 生成下一个 GCN 输入
        next_obs = self.get_gcn_input(region_id, src, dst, flowmap)
        next_X = next_obs['x']
        next_edge_index = next_obs['edge_index']
        next_path_indices = next_obs['paths']

        # 5. 是否结束（可忽略）
        done = False
        return reward, next_X, next_edge_index, next_path_indices, done


    def generate_ksp(self, src, dst, weight_attr='weight'):
        if self.__noderegionid[src] != self.__noderegionid[dst]:
            return []
        """使用networkx生成K条最短路径"""
        import networkx as nx
        region_id = self.__noderegionid[src]
        region_nodes = self.get_region_nodes(region_id)

        G = nx.DiGraph()
        for u, v, w, capa, _ in self.__linkset:
            if capa>0 and u in region_nodes and v in region_nodes:
                G.add_edge(u, v, weight=self.__wMatrix[u][v])
                G.add_edge(v, u, weight=self.__wMatrix[v][u])
        try:
            return list(nx.shortest_simple_paths(G, src, dst, weight=weight_attr))[:self.K]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def score_path(self, path):
        """路径综合评分：时延 + 最大利用率 + 丢包率"""
        total_delay = sum(self.__wMatrix[u][v] for u, v in zip(path, path[1:]))
        max_util = max(self.get_current_flow_util(u, v) for u, v in zip(path, path[1:]))
        # 动态调整权重
        # util_weight = 0.7 if max_util > 0.8 else 0.3
        util_weight = max_util * 0.8
        delay_weight = 1 - util_weight

        return (util_weight * max_util) + (delay_weight * total_delay / 100)

    def dynamic_path_selection(self):
        """动态筛选Top-M路径"""
        for src in range(self.__nodenum):
            for dst in range(self.__nodenum):
                if src == dst or self.__noderegionid[src] != self.__noderegionid[dst]:
                    continue
                valid_paths = [p for p in self.intra_paths[src][dst] if self.validate_intra_path(p)]
                if not valid_paths:
                    default_path = self.generate_ksp(src, dst)
                    valid_paths = default_path[:1]
                scored_paths = sorted(
                    [(self.score_path(p), p) for p in valid_paths],
                    key=lambda x: x[0]
                )
                selected_paths = [p for score, p in scored_paths[:self.M] if score > 0 and len(p) >= 2]
                # 确保至少有一条路径
                if not selected_paths:
                    selected_paths = [valid_paths[0]]  # 使用第一条有效路径
                self.intra_paths[src][dst] = selected_paths

    def check_network_state_change(self, current_util):
        """检测网络状态是否显著变化"""
        if self.last_utilization is None:
            self.last_utilization = current_util
            return False
        # 1. 利用率变化检测
        delta_util = np.abs(np.array(current_util) - np.array(self.last_utilization)).max()

        # 2. 路径有效性检测
        active_paths = sum(len(p) for src in self.__nodenum for dst in self.__nodenum
                               for p in self.intra_paths[src][dst])
        total_possible = self.__nodenum ** 2 * self.M
        path_health_ratio = active_paths / total_possible

        return delta_util > self.path_update_threshold or path_health_ratio < 0.6

    def dynamic_refresh_paths(self, force=False):
        """智能路径更新策略"""
        # 强制更新条件
        if force or self.check_network_state_change():
            print(f"[事件触发] 重新生成域内 KSP 路径...")
            for src in range(self.__nodenum):
                for dst in range(self.__nodenum):
                    if self.__noderegionid[src] == self.__noderegionid[dst] and src != dst:
                        self.intra_paths[src][dst] = self.generate_ksp(src, dst)
            self.dynamic_path_selection()
            self.get_paths()
            self.last_update_step = self.__updatenum
            return

            # 基于链路状态的条件更新
        high_util_links = sum(
            1 for u, v in self.__linkset
            if self.get_current_flow_util(u, v) > 0.8
        )
        if high_util_links / len(self.__linkset) > 0.3:
            for src in range(self.__nodenum):
                for dst in range(self.__nodenum):
                    if self.__noderegionid[src] == self.__noderegionid[dst] and src != dst:
                        self.intra_paths[src][dst] = self.generate_ksp(src, dst)
            self.dynamic_path_selection()
        self.get_paths()

    def validate_intra_path(self, path):
        """严格验证路径完全在域内"""
        if not path:
            return False
        region_id = self.__noderegionid[path[0]]
        return all(self.__noderegionid[node] == region_id for node in path)

    def get_agent2_state(self, region_id, dst_region, pheromones, target_dim=20):
        state = []
        # 1. 本域出口链路利用率平均值
        border_edges = []
        for u in self.__bordernodes[region_id]:
            for v in range(self.__nodenum):
                if self.__noderegionid[v] != region_id and self.__wMatrix[u][v] < 999999:
                    border_edges.append((u, v))
        utilizations = [self.get_current_flow_util(u, v) for u, v in border_edges]
        w_i = np.mean(utilizations) if utilizations else 0.0
        state.append(w_i)

        # 2. 添加邻接区域信息素和边权
        for j in self.__regionnodeneibor[region_id]:
            for u in self.__bordernodes[region_id]:
                for v in self.__bordernodes[j]:
                    if self.__wMatrix[u][v] >= 999999:
                        continue
                    w_uv = self.__wMatrix[u][v] / 100.0
                    τ_uv = pheromones.get((region_id, j, u, v))
                    state.extend([j, u, v, w_uv, τ_uv])

        state.extend([region_id, dst_region])
        state_np = np.array(state, dtype=np.float32)
        if len(state_np) < target_dim:
            state_np = np.pad(state_np, (0, target_dim - len(state_np)))
        elif len(state_np) > target_dim:
            state_np = state_np[:target_dim]
        return state_np

    def decode_action(self, region_id, action_idx):
        candidate_actions = []
        for j in self.__regionnodeneibor[region_id]:
            for u in self.__bordernodes[region_id]:
                for v in self.__bordernodes[j]:
                    if self.__wMatrix[u][v] < 999999:
                        candidate_actions.append((j, u, v))
        if action_idx >= len(candidate_actions):
            return candidate_actions[0]  # 默认动作
        return candidate_actions[action_idx]

    def agent2_step(self, region_id, src, dst, next_region, e_out, e_in):
        path = [src, e_out, e_in, dst]
        for i in range(len(path) - 1):
            if self.__wMatrix[path[i]][path[i + 1]] >= 999999:
                return -1.0, self.get_agent2_state(region_id, self.__noderegionid[dst], {}), True

        self.__actionmatrix[src][dst] = [1.0]  # TODO: 适配你真实路径替换逻辑
        flowmap = self.compute_flowmap()

        max_util = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            capacity = self.__cMatrix[u][v]
            if capacity <= 0:
                util = 1.0
            else:
                util = flowmap[u][v] / capacity
            max_util = max(max_util, util)

        reward = -max_util

        next_state = self.get_agent2_state(region_id, self.__noderegionid[dst], {})
        return reward, next_state, False

    def get_inter_domain_demands(self, region_id):
        results = []
        for src in range(self.__nodenum):
            if self.__noderegionid[src] != region_id:
                continue
            for dst in range(self.__nodenum):
                if self.__noderegionid[dst] != region_id and src != dst:
                    results.append((src, dst))
        return results

    def get_num_regions(self):
        return self.__regionnum

    def get_node_region(self, node_id):
        return self.__noderegionid[node_id]

    def get_max_action_dim(self):
        total = 0
        for i in range(self.__regionnum):
            for j in self.__regionnodeneibor[i]:
                for u in self.__bordernodes[i]:
                    for v in self.__bordernodes[j]:
                        if self.__wMatrix[u][v] < 999999:
                            total += 1
        return total

    def get_max_utilization(self):
        flowmap = self.compute_flowmap()
        return max([
            flowmap[i][j] / self.__cMatrix[i][j]
            for i in range(self.__nodenum)
            for j in range(self.__nodenum)
            if self.__edgemap[i][j] >= 0 and self.__cMatrix[i][j] > 0
        ])

    def get_demand_volume(self, src, dst):
        return self.__TM[src][dst]

    def get_path_delay(self, path):
        return sum(self.__wMatrix[path[i]][path[i+1]] for i in range(len(path)-1))

    # 在 Environment 类中添加以下方法
    def set_single_action(self, src, dst, action):
        """设置单个 src-dst 对的 action"""
        self.__actionmatrix[src][dst] = action

    def get_action_matrix(self):
        """获取整个 action 矩阵"""
        return self.__actionmatrix

    def set_flowmap(self, flowmap):
        """设置当前流量图"""
        self.__current_flowmap = flowmap

    def get_flowmap(self):
        """获取当前流量图"""
        return self.__current_flowmap