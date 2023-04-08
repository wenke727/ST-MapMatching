from mapmatching import build_geograph, ST_Matching

"""step 1: 获取/加载路网"""
# 方法1：
# 根据 bbox 从 OSM 下载路网，从头解析获得路网数据
# net = build_geograph(bbox=[113.930914, 22.570536, 113.945456, 22.585613],
#                      xml_fn="./data/network/LXD.osm.xml", ll=False)
# 将预处理路网保存为 ckpt
# net.save_checkpoint('./data/network/LXD_graph.ckpt')

# 方法2：
# 使用预处理路网 
net = build_geograph(ckpt='./data/network/LXD_graph.ckpt') 

"""step 2: 创建地图匹配 matcher"""
matcher = ST_Matching(net=net, ll=False)

"""step 3: 加载轨迹点集合，以打石一路为例"""
idx = 4
traj = matcher.load_points(f"./data/trajs/traj_{idx}.geojson").reset_index(drop=True)
res = matcher.matching(traj, top_k=5, dir_trans=True, details=False, plot=True,
                       simplify=True, debug_in_levels=False)

# 后续步骤可按需选择
"""step 4: 将轨迹点映射到匹配道路上"""
path = matcher.transform_res_2_path(res, ori_crs=True)
proj_traj = matcher.project(traj, path)

"""step 5: eval"""
matcher.eval(traj, res, resample=5, eps=10)
