#%%
import heapq
import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame

import sys

sys.path.append('..')

from utils.timer import Timer
from graph.geograph import GeoDigraph
from geo.haversine import haversine_np, Unit
from geo.geo_helper import point_to_polyline_process, coords_pair_dist, geom_series_distance
from geo.azimuth_helper import azimuth_cos_similarity_for_linestring, azimuthAngle
from match.geometricAnalysis import project_point_to_line_segment, cal_observ_prob
from match.candidatesGraph import construct_graph

pd.set_option('display.max_columns', 500)


""" candidatesGraph """
def cal_traj_distance(points):
    coords = points.geometry.apply(lambda x: [*x.coords[0]]).values.tolist()
    coords = np.array(coords)

    dist = haversine_np(coords[:-1], coords[1:], xy=True, unit=Unit.METERS)
    idxs = points.index
    
    return pd.DataFrame({'pid_0': idxs[:-1],
                         'pid_1':idxs[1:],
                         'd_euc': dist})


""" Geometric info"""
def _move_dir_similarity(traj:GeoDigraph, graph:GeoDataFrame, geograph):
    # if 'geometry' not in list(graph):
    # TODO 获得 polyline -> combine_edges_gemos_to_polyline()
    # graph[['eid_0', 'eid_1', 'eid_list', 'path']]
    graph.loc[:, 'geometry'] = graph.path.apply(geograph.transform_node_seq_to_polyline)
                
    graph.loc[:, 'move_dir'] = graph.apply(
        lambda x: 
            azimuthAngle(*traj.iloc[x.pid_0].geometry.coords[0], 
                         *traj.iloc[x.pid_1].geometry.coords[0]),
        axis=1
    )
    
    graph.loc[:, 'f_dir'] = graph.apply(
        lambda x: 
            (azimuth_cos_similarity_for_linestring(x.geometry, x.move_dir, weight=True) + 1) / 2
                if x.geometry is not None else 1, 
        axis=1
    )

    # FIXME Manually change the `f_dir` weights of the starting and ending on the same line segment
    # same_link_mask = graph.eid_0 == graph.eid_1
    # graph.loc[same_link_mask, 'f_dir'] = 1
    
    return graph


""" topological info """
def _trans_prob(gt:GeoDataFrame, net:GeoDigraph, max_steps:int, max_dist:int):
    same_link_mask = gt.eid_0 == gt.eid_1
    ods = gt[~same_link_mask][['dst', 'src']].drop_duplicates().values

    if len(ods) > 0:
        routes = []
        for o, d in ods:
            _r = net.search(o, d, max_steps=max_steps, max_dist=max_dist)
            routes.append({'dst': o, 'src': d, **_r})
            
        df_planning = pd.DataFrame(routes)
        gt = gt.merge(df_planning, on=['dst', 'src'], how='left')
        # `w` is the shortest path from `ci-1` to `ci`
        gt.loc[:, 'w'] = gt.cost + gt.offset_0 + gt.offset_1 
        # transmission probability
        gt.loc[:, 'v'] = gt.apply(lambda x: x.d_euc / x.w if x.d_euc < x.w else x.w / x.d_euc * 1.00, axis=1 )

    gt.loc[same_link_mask, 'path'] = gt.loc[same_link_mask, 'eid_0'].apply(
        lambda x: net.get_edge(x, att=['src', 'dst']).values.tolist())
    gt.loc[same_link_mask, 'v'] = 1
    
    return gt


""" spatial_analysis """
def analyse_spatial_info(geograph:GeoDigraph, 
                         points:GeoDataFrame, 
                         cands:GeoDataFrame, 
                         dir_trans=False, 
                         max_steps:int=2e3, 
                         max_dist:int=1e5,
                         gt_keys:list = ['pid_0', 'eid_0', 'eid_1']):
    """Geometric and topological info, the product of `observation prob` and the `transmission prob`
    
    Special Case:
        a. same_link_same_point
    """
    gt = construct_graph(cands)
    
    # 存在节点没有匹配的情况，目前的策略是忽略
    dist = cal_traj_distance(points.loc[cands.pid.unique()])
    gt = gt.merge(dist, on=['pid_0', 'pid_1'])
    
    gt = _trans_prob(gt, geograph, max_steps, max_dist)
    if dir_trans:
        gt = _move_dir_similarity(points, gt, geograph)
    
    # FIXME
    # spatial analysis: observ_prob * trans_prob
    # gt.loc[:, 'f'] = gt.observ_prob * gt.v * (gt.f_dir if dir_trans else 1)
    gt.loc[:, 'f'] = gt.v * (gt.f_dir if dir_trans else 1)

    gt = gt.drop_duplicates(gt_keys).set_index(gt_keys).sort_index()

    return cands, gt


def _load_test_data():
    from shapely import wkt
    from osmnet.build_graph import build_geograph

    # 读取轨迹点
    points = gpd.read_file("../../input/traj_1.geojson")
    
    # 读取数据
    fn = '../../tmp/candidates.geojson'
    cands = gpd.read_file(fn, crs='epsg:4326')
    cands.loc[:, 'point_geom'] = cands.point_geom.apply(wkt.loads)
    
    # 读取基础路网
    fn = '../../cache/Shenzhen_graph.ckpt'
    geograph = build_geograph(ckpt=fn)

    return geograph, points, cands


if __name__ == "__main__":
    geograph, points, cands = _load_test_data()
    cands, gt = analyse_spatial_info(geograph, points, cands, True)
    
    
#%%
    # 构建优先队列, 最小堆
    def temp():
        queue = []
        factor = 5
        counter = 0

        for level, links in layers.items():
            for eid_0, nxts in links.items():
                remained_nodes = set()
                for eid_1 in nxts:
                    counter += 1
                    _dict = gt[(eid_0, eid_1)]
                    route = geograph.search(_dict['src'], _dict['dst'])
                    if 'cost' not in route:
                        continue
                    # if queue and info.get('cost', np.inf) > queue[0][0] * factor:
                        # continue
                    # remained_nodes.add(d)

                    _dict.update(route)
                    # record = (eid_0.)
                    heapq.heappush(queue, (route['cost'], (eid_0, eid_1)))
                    
            break

        print(len(remained_nodes), len(queue) / counter)
            
        queue

#%%
def tmp():
    # ! 分层分析, 通过领域知识计算一个权重
    gt = graph.copy()

    # %%
    id = 0
    lvl = gt.set_index('pid_0').loc[id]
    func = lambda x: geograph.search(x.dst, x.src)
    routes = lvl.apply(func, axis=1, result_type='expand')
    lvl = pd.concat([lvl, routes], axis=1)
    lvl.loc[:, 'w'] = lvl.cost + lvl.offset_0 + lvl.offset_1 

    # 距离都是通过 haversine 计算出来的1
    lvl['w'].min(), lvl['w'].max(), lvl['d_euc'].unique()

    # %%
    dist_factor = 2
    print(lvl.query(f" w > w.min() * {dist_factor}")['eid_1'].unique())
    print(lvl['eid_1'].unique())
    lvl
    # %%

    from db.db_process import gdf_to_postgis

    gdf_to_postgis(geograph.df_edges, 'topo_osm_shenzhen_edge')
    gdf_to_postgis(geograph.df_nodes, 'topo_osm_shenzhen_node')
    # gdf_to_postgis(geograph.df_edges.drop(columns=['geom_origin', 'waypoints']), "topo_osm_shenzhen_node")
    
    # geograph.df_edges.drop(columns=['geom_origin', 'waypoints']).to_file('./nodes.geojson', driver="GeoJSON")
    arrs = np.unique(np.array(gt.loc[0].index.levels).flatten())
    len(arrs)

# %%
