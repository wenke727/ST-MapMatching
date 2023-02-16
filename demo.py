#%%
from mapmatching import build_geograph, ST_Matching
from mapmatching.setting import DATA_FOLDER

"""step 1: 获取/加载路网"""
# 通过bbox获取路网数据, 通过 xml_fn 指定存储位置
# net = build_geograph(bbox=[113.930914, 22.570536, 113.945456, 22.585613],
#                      xml_fn="./data/network/LXD.osm.xml")

# 通过读取 xml，处理后获得路网数据
# net = build_geograph(xml="./data/network/Shenzhen.osm.xml")

# 将预处理路网保存为 ckpt
# net.save_checkpoint('./data/network/Shenzhen_graph.ckpt')

# 加载 ckpt
net = build_geograph(ckpt='./data/network/Shenzhen_graph_pygeos.ckpt')

"""step 2: 创建地图匹配 matcher"""
matcher = ST_Matching(net=net)

"""step 3: 加载轨迹点集合，以打石一路为例"""
traj = matcher.load_points("./data/trajs/traj_4.geojson")
res = matcher.matching(traj, plot=True, top_k=5, dir_trans=True, details=True)

"""step 4: eval"""
matcher.eval(traj, res, resample=5, eps=10)

# %%
net.df_edges.crs
proj_crs = net.df_edges.estimate_utm_crs()

import pyproj

# pyproj.CRS.from_string(proj_crs)
# %timeit net.df_nodes.to_crs(proj_crs) # 351 ms
# %timeit net.df_edges.to_crs(proj_crs) # 282 ms

# %timeit traj.to_crs(proj_crs) # 2.59 ms
proj_crs.to_epsg()
# %%
dir(proj_crs)
# %%

net.epsg

# %%
