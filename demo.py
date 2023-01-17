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

#%%
vpath = net.transform_epath_to_vpath(res['epath'])
vpath
# epath = net.transform_vpath_to_epath(vpath)
# epath == res['epath']

#%%
def check_details():
    import pandas as pd
    pd.set_option('display.max_rows', 20)

    r = res['details']
    # r.keys() # 'cands', 'rList', 'graph', 'route', 'steps', 'simplified_traj'

    # %%
    r['cands']

    # %%
    r['rList']

    #%%
    r['steps']

    #%%
    r['path']

    # %%
    r['graph'][list(r['graph'])[:12]]

    # %%
    r['graph'][list(r['graph'])[12:]]

    # %%
    import geopandas as gpd
    from tilemap import plot_geodata

    i = 0
    _, ax = plot_geodata(gpd.GeoDataFrame((r['graph'].iloc[[i]])))
    gpd.GeoDataFrame((r['graph'].iloc[[i]])).set_geometry('path').plot(ax=ax, color='r')
    r['simplified_traj'].plot(ax=ax, color='black')

    # %%
