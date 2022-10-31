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
from geo.azimuth_helper import azimuth_cos_similarity_for_linestring, azimuthAngle
from match.candidatesGraph import construct_graph


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
def cal_move_dir_prob(traj:GeoDigraph, gt:GeoDataFrame):
    assert 'geometry' in gt, "Check the geometry of gt"

    """ check geometry is almost equals """
    if False:
        geoms = gt.path.apply(geograph.transform_node_seq_to_polyline)
        geoms.geom_almost_equals(graph.loc[:, 'geometry'])
        cond = gpd.GeoDataFrame(gt).geom_almost_equals(gpd.GeoSeries(geoms))
        flag = cond.mean()
        if flag != 1:
            print("check details")
        graph.loc[:, 'geometry'] = geoms
    
    cal_move_dir = lambda x: azimuthAngle(*traj.iloc[x.pid_0].geometry.coords[0], 
                                          *traj.iloc[x.pid_1].geometry.coords[0])
    # BUG x.geometry 为途径路段, 是否增加第一段和最后一段的edge
    cal_f_simil = lambda x: (azimuth_cos_similarity_for_linestring(x.geometry, x.move_dir, weight=True) + 1) / 2\
                                if x.geometry is not None else 1
    
    gt.loc[:, 'move_dir'] = gt.apply(cal_move_dir, axis=1)
    gt.loc[:, 'f_dir'] = gt.apply(cal_f_simil, axis=1)
    gt.loc[gt.flag==1, 'f_dir'] = 1

    tmp = gt[['eid_0', 'eid_1', 'pid_0', 'pid_1', 'path', 'eid_list']]
    # FIXME Manually change the `f_dir` weights of the starting and ending on the same line segment
    # same_link_mask = graph.eid_0 == graph.eid_1
    # graph.loc[same_link_mask, 'f_dir'] = 1
    
    return gt


""" topological info """
def cal_trans_prob(gt:GeoDataFrame, net:GeoDigraph, max_steps:int, max_dist:int):
    ods = gt[['dst', 'src']].drop_duplicates().values

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

    # BUG (o, d) 位于同一 edge 上的处理方式
    same_link_mask = gt.eid_0 == gt.eid_1
    # gt.loc[same_link_mask, 'path'] = None
    # gt.loc[same_link_mask, 'eid_list'] = None
    gt.loc[same_link_mask, 'v'] = 1
    
    # _gt = gt[same_link_mask]
    # cond = _gt.apply(lambda x: net.get_edge(x.eid_0, 'dist') - x.offset_1 < x.offset_0, axis=1)
    # revert_idxs = cond[~cond].index
    # normal_idxs = cond[cond].index
    
    # if len(normal_idxs):
    #     gt.loc[normal_idxs, 'path'] = None
    #     gt.loc[normal_idxs, 'eid_list'] = None
    #     gt.loc[normal_idxs, 'v'] = 1
    # if len(revert_idxs):
    #     pass
    
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
    
    # 存在节点没有匹配的情况，目前的策略是忽略，还有顺序的问题
    dist = cal_traj_distance(points.loc[cands.pid.unique()])
    gt = gt.merge(dist, on=['pid_0', 'pid_1'])
    
    gt = cal_trans_prob(gt, geograph, max_steps, max_dist)
    if dir_trans:
        gt = cal_move_dir_prob(points, gt)
    
    # FIXME
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
    fn = '../../cache/Shenzhen_graph_9.ckpt'
    geograph = build_geograph(ckpt=fn)

    return geograph, points, cands


if __name__ == "__main__":
    geograph, points, cands = _load_test_data()
    cands, gt = analyse_spatial_info(geograph, points, cands, True)
    
    
