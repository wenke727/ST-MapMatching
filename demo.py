#%%
from mapmatching import build_geograph, ST_Matching
from mapmatching.setting import DATA_FOLDER

"""step 1: 获取/加载路网"""
# 通过bbox获取路网数据, 通过 xml_fn 指定存储位置
# net = build_geograph(bbox=[113.930914, 22.570536, 113.945456, 22.585613],
#                      xml_fn="./data/network/LXD.osm.xml")

# 通过读取 xml，处理后获得路网数据
# net = build_geograph(xml_fn="./data/network/LXD.osm.xml")

# 将预处理路网保存为 ckpt
# net.save_checkpoint('./data/network/LXD_graph.ckpt')

# 加载 ckpt
# net = build_geograph(ckpt='./data/network/LXD_graph.ckpt', ll=False)
net = build_geograph(ckpt='./data/network/Shenzhen_graph_pygeos.ckpt', ll=False)
net.to_proj()

"""step 2: 创建地图匹配 matcher"""
matcher = ST_Matching(net=net, ll=False)

#%%
# 0, 2, 9, 12, 13, 15
# 1, 8, 9, 13 (1 主要贡献在于 距离)
"""step 3: 加载轨迹点集合，以打石一路为例"""
idx = 13
traj = matcher.load_points(f"./data/trajs/traj_{idx}.geojson").reset_index(drop=True)
res = matcher.matching(traj, plot=True, top_k=5, dir_trans=True, 
                       details=True, simplify=True, debug_in_levels=False)
res['epath']


#%%
# """step 4: eval"""
# matcher.eval(traj, res, resample=5, eps=10)

# %%
gt = res['details']['graph']
layer = gt.query("eid_1 == 118899")
item = layer.loc[0, 118898, 118899]

print(item.step_0.coords[:])
print(item.geometry.coords[:])
print(item.step_n.coords[:])
print(item.dir_prob)

item.path.coords[:]

# %%
