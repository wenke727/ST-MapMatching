from mapmatching import build_geograph, ST_Matching

"""step 1: 获取/加载路网"""
# 通过bbox获取路网数据, 通过 xml_fn 指定存储位置
net = build_geograph(bbox=[113.930914, 22.570536, 113.945456, 22.585613],
                     xml_fn="./data/network/LXD.osm.xml")

# 通过读取 xml，处理后获得路网数据
# net = build_geograph(xml="../cache/Shenzhen.osm.xml")

# 将预处理路网保存为 ckpt
# net.save_checkpoint('../cache/Shenzhen_graph_9.ckpt')

# 加载 ckpt
# net = build_geograph(ckpt='../cache/Shenzhen_graph_9.ckpt')

"""step 2: 创建地图匹配 matcher"""
matcher = ST_Matching(net=net)

"""step 3: 加载轨迹点集合，以打石一路为例"""
traj = matcher.load_points("./data/trajs/traj_4.geojson")
# traj = self.load_points("/home/pcl/codes/ST-MapMatching/data/tmp/trip_amp_hw_compressed.geojson")
path, info = matcher.matching(traj, plot=True, top_k=5)

