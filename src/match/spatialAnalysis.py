#%%
import heapq
import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import LineString

import sys
sys.path.append('..')

from utils.timer import Timer
from graph.geograph import GeoDigraph
from geo.azimuth_helper import cal_azimuth_cos_dist_for_linestring, azimuthAngle
from match.candidatesGraph import construct_graph


def merge_steps(gt):
    # x.geometry + step_0 + step_n
    def get_coords(item):
        if item is None or len(item) == 0:
            return None
        
        return item
    
    def helper(x):
        if x.geometry is None:
            waypoints = None
        else:
            waypoints = x.geometry.coords[:]
        
        lst = []
        first = get_coords(x.first_step)
        last = get_coords(x.last_step)
        
        if first is not None:
            lst.append(first)
        if waypoints is not None:
            lst.append(waypoints)
        if last is not None:
            lst.append(last)

        if len(lst) == 0:
            return None
        
        polyine = LineString(np.concatenate(lst))
        return polyine
    
    return gt.apply(helper, axis=1)


def check_combine_steps(idx, traj, graph):
    from tilemap import plot_geodata
    fig, ax = plot_geodata(traj, color='r', reset_extent=False)

    gdf = gpd.GeoDataFrame(graph).iloc[[idx]].set_geometry('whole_path')
    gdf.plot(ax=ax, color='blue', alpha=.5)

    _gdf = gpd.GeoDataFrame(graph).iloc[[idx]].set_geometry('geometry')
    _gdf.plot(ax=ax, color='r', linestyle=':', alpha=.5)


def cal_dir_prob(gt:GeoDataFrame, geom='geometry'):
    # Add: f_dir
    assert geom in gt, "Check the geometry of gt"

    def _cal_f_similarity(x):
        if x[geom] is None:
            return None
        
        f = cal_azimuth_cos_dist_for_linestring(x[geom], x['move_dir'], weight=True)

        return f
    
    gt.loc[:, 'f_dir'] = gt.apply(_cal_f_similarity, axis=1)
    
    filtered_idxs = gt.query("flag == 1").index
    gt.loc[filtered_idxs, 'f_dir'] = 1

    return gt


def cal_dist_prob(gt:GeoDataFrame, net:GeoDigraph, max_steps:int=2000, max_dist:int=10000):
    # Add: w, v, path, geometry
    assert 'flag' in gt, "Chech the attribute `flag` in gt or not"
    ods = gt[['dst', 'src']].drop_duplicates().values

    if len(ods) > 0:
        routes = []
        for o, d in ods:
            _r = net.search(o, d, max_steps=max_steps, max_dist=max_dist)
            routes.append({'dst': o, 'src': d, **_r})
            
        df_planning = pd.DataFrame(routes)
        gt = gt.merge(df_planning, on=['dst', 'src'], how='left')
        # `w` is the shortest path from `ci-1` to `ci`
        gt.loc[:, 'w'] = gt.cost + gt.last_step_len + gt.first_step_len 
        # dist_transmission probability
        gt.loc[:, 'v'] = gt.apply(lambda x: x.d_euc / x.w if x.d_euc < x.w else x.w / x.d_euc * 1.00, axis=1 )

    # 针对 flag == 1（即 o, d 位于同一`edge`上， 且 o 的相对位置靠前）
    filtered_idxs = gt.query("flag == 1").index
    gt.loc[filtered_idxs, 'v'] = 1
    gt.loc[filtered_idxs, 'path'] = None
    
    return gt


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
    gt = construct_graph(points, cands, dir_trans=dir_trans)
    
    gt = cal_dist_prob(gt, geograph, max_steps, max_dist)
    gt.loc[:, 'whole_path'] = merge_steps(gt)

    if dir_trans:
        cal_dir_prob(gt, 'whole_path')
        # cal_dir_prob(gt, 'geometry')
        gt.loc[:, 'f'] = gt.v * gt.f_dir
    else:
        gt.loc[:, 'f'] = gt.v

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
    
    
