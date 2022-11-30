import heapq
import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import LineString

from ..graph import GeoDigraph
from .candidatesGraph import construct_graph

from ..geo.azimuth import cal_linestring_azimuth_cos_dist


def merge_steps(gt):
    # x.geometry + step_0 + step_n
    def get_coords(item):
        if item is None or len(item) == 0:
            return None
        
        return item
    
    def helper(x):
        if not x.geometry:
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
        
        polyline = LineString(np.concatenate(lst))
        return polyline
    
    return gt.apply(helper, axis=1)


def _check_combine_steps(idx, traj, graph):
    from tilemap import plot_geodata
    fig, ax = plot_geodata(traj, color='r', reset_extent=False)

    gdf = gpd.GeoDataFrame(graph).iloc[[idx]].set_geometry('whole_path')
    gdf.plot(ax=ax, color='blue', alpha=.5)

    _gdf = gpd.GeoDataFrame(graph).iloc[[idx]].set_geometry('geometry')
    _gdf.plot(ax=ax, color='r', linestyle=':', alpha=.5)


# @timeit
def cal_dir_prob(gt:GeoDataFrame, geom='geometry'):
    # Add: f_dir
    assert geom in gt, "Check the geometry of gt"

    def _cal_dir_similarity(x):
        return cal_linestring_azimuth_cos_dist(x[geom], x['move_dir'], weight=True)
    
    gt.loc[:, 'f_dir'] = gt.apply(_cal_dir_similarity, axis=1)
    
    filtered_idxs = gt.query("flag == 1").index
    gt.loc[filtered_idxs, 'f_dir'] = 1

    return gt


# @timeit
def cal_dist_prob(gt: GeoDataFrame, net: GeoDigraph, max_steps: int = 2000, max_dist: int = 10000):
    # Add: w, v, path, geometry
    assert 'flag' in gt, "Chech the attribute `flag` in gt or not"

    paths = gt.apply(lambda x:
                        net.search(x.dst, x.src, max_steps, max_dist),
                     axis=1, result_type='expand')
    gt.loc[:, list(paths)] = paths
    # `w` is the shortest path from `ci-1` to `ci`
    gt.loc[:, 'w'] = gt.cost + gt.last_step_len + gt.first_step_len
    # distance transmission probability
    gt.loc[:, 'dist_prob'] = gt.d_euc / gt.w
    mask = gt['dist_prob'] > 1
    gt.loc[mask, 'dist_prob'] = 1 / gt.loc[mask, 'dist_prob']

    # case: flag = 1
    filtered_idxs = gt.query("flag == 1").index
    gt.loc[filtered_idxs, 'dist_prob'] = 1
    gt.loc[filtered_idxs, 'path'] = None

    # case: flag = 2
    filtered_idxs = gt.query("flag == 2").index
    gt.loc[filtered_idxs, 'dist_prob'] *= .99

    return gt


def cal_trans_prob(gt, geometry, dir_trans):
    if dir_trans:
        gt.loc[:, 'whole_path'] = merge_steps(gt)
        cal_dir_prob(gt, geometry)
        gt.loc[:, 'trans_prob'] = gt.dist_prob * gt.f_dir
        return gt

    gt.loc[:, 'trans_prob'] = gt.dist_prob
    return gt


# @timeit
def analyse_spatial_info(geograph: GeoDigraph,
                         points: GeoDataFrame,
                         cands: GeoDataFrame,
                         dir_trans=False,
                         max_steps: int = 2e3,
                         max_dist: int = 1e5,
                         gt_keys: list = ['pid_0', 'eid_0', 'eid_1'],
                         geometry='whole_path'):
    """
    Geometric and topological info, the product of `observation prob` and the `transmission prob`
    """
    gt = construct_graph(points, cands, dir_trans=dir_trans, gt_keys=gt_keys)
    
    gt = cal_dist_prob(gt, geograph, max_steps, max_dist)
    cal_trans_prob(gt, geometry, dir_trans)

    return gt


def get_trans_prob_bet_layers(gt, net, dir_trans=True, geometry='whole_path'):
    ori_index = gt.index
    gt = cal_dist_prob(gt, net)
    gt.index = ori_index
    cal_trans_prob(gt, geometry, dir_trans)
    
    return gt


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
    gt = analyse_spatial_info(geograph, points, cands, True)
    
    
