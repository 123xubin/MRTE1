import pickle
import networkx as nx
from time import time, sleep

# 配置参数
topo_path = '../inputs/region/1221.txt'
output_path = '../inputs/pathset/intra/1221_KSP.pickle'  # 建议放入 intra 子目录
K = 5

# 1. 读取拓扑和区域划分
with open(topo_path, 'r') as f:
    lines = f.readlines()

# 第一行是节点数和链路数
nodenum, linknum = map(int, lines[0].strip().split())

# 构建图
G = nx.DiGraph()
region_ids = list(map(int, lines[linknum + 1].strip().split()))
region_num = max(region_ids) + 1

for line in lines[1:1 + linknum]:
    u, v, w, c, region = line.strip().split()
    u, v, w = int(u) - 1, int(v) - 1, float(w)
    G.add_edge(u, v, weight=w)
    G.add_edge(v, u, weight=w)  # 假设为无向图（如 MRTE 默认）

# 2. 生成每个域的域内路径
paths = [[[] for _ in range(nodenum)] for _ in range(nodenum)]

start_time = time()
print(f"[INFO] 开始路径生成，共 {region_num} 个域...")

for region in range(region_num):
    nodes_in_region = [i for i in range(nodenum) if region_ids[i] == region]
    total_pairs = len(nodes_in_region) * (len(nodes_in_region) - 1)
    pair_count = 0

    print(f"\n[Region {region}] 包含 {len(nodes_in_region)} 个节点，处理 {total_pairs} 个 src-dst 对...")
    for src in nodes_in_region:
        for dst in nodes_in_region:
            if src != dst:
                pair_count += 1
                try:
                    ksp = []
                    for path in nx.shortest_simple_paths(G, source=src, target=dst, weight='weight'):
                        if len(ksp) >= K:
                            break
                        ksp.append(path)
                    paths[src][dst] = ksp
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    paths[src][dst] = []

                # 简单进度提示
                if pair_count % 20 == 0 or pair_count == total_pairs:
                    percent = 100 * pair_count / total_pairs
                    print(f"  [{pair_count}/{total_pairs}] ({percent:.1f}%) src={src} → dst={dst}", end='\r')

end_time = time()
duration = end_time - start_time
print(f"\n\n✅ 路径生成完成，用时 {duration:.2f} 秒")

# 3. 保存为 pickle 文件
with open(output_path, 'wb') as f:
    pickle.dump(paths, f)

# 在 generate_intra_only_pickle.py 中添加验证代码
with open(output_path, 'rb') as f:
    paths = pickle.load(f)
for src in range(nodenum):
    for dst in range(nodenum):
        if region_ids[src] == region_ids[dst] and src != dst:
            assert len(paths[src][dst]) > 0, f"空路径: {src}→{dst}"

print(f"[DONE] 域内路径集合已保存为: {output_path}")
