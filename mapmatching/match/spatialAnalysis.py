import warnings
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import LineString

from ..graph import GeoDigraph
from .candidatesGraph import construct_graph
from .dir_similarity import cal_dir_prob


def cal_dist_prob(gt: GeoDataFrame, net: GeoDigraph, max_steps: int = 2000, max_dist: int = 10000):
    # Add: w, v, path, geometry
    assert 'flag' in gt, "Chech the attribute `flag` in gt or not"

    if gt.empty:
        warnings.warn("Empty graph layer")
        return gt

    paths = gt.apply(lambda x: 
        net.search(x.dst, x.src, max_steps, max_dist), axis=1, result_type='expand')
    gt.loc[:, list(paths)] = paths.values
    gt.loc[:, 'd_sht'] = gt.cost + gt.step_n_len + gt.step_0_len

    # FIXME 最短路返回结果 gt.loc[0, 13884, 13884]
    # od 位于同一条edge上，但起点相对终点位置偏前
    flag_1_idxs = gt.query("flag == 1").index
    if len(flag_1_idxs):
        gt.loc[flag_1_idxs, 'epath'] = None
        gt.loc[flag_1_idxs, 'geometry'] = LineString()
        gt.loc[flag_1_idxs, 'd_sht'] = gt.step_n_len + gt.step_0_len - gt.dist

    
    # distance transmission probability
    dist = gt.d_euc / gt.d_sht
    mask = dist > 1 # Penalize slightly shorter paths
    dist[mask] = 1 / dist[mask]
    gt.loc[:, 'dist_prob'] = dist

    # dist_prob, 但两个点落在同一个节点上的时候为 0
    flag_2_idxs = gt.query("flag == 2").index
    gt.loc[flag_2_idxs, 'dist_prob'] *= 1.0001
    gt.loc[flag_1_idxs, 'dist_prob'] *= 1.0002

    return gt

def cal_trans_prob(gt, geometry, dir_trans):
    if not dir_trans:
        gt.loc[:, 'trans_prob'] = gt.dist_prob
        return gt

    cal_dir_prob(gt, geometry)
    gt.loc[:, 'trans_prob'] = gt.dist_prob * gt.dir_prob
    
    return gt

def analyse_spatial_info(net: GeoDigraph,
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
    
    gt = cal_dist_prob(gt, net, max_steps, max_dist)
    cal_trans_prob(gt, geometry, dir_trans)

    return gt

def get_trans_prob_bet_layers(gt, net, dir_trans=True, geometry='path'):
    """
    For beam-search
    """
    if gt.empty:
        return gt

    gt = cal_dist_prob(gt, net)
    cal_trans_prob(gt, geometry, dir_trans)
    
    return gt


if __name__ == "__main__":
    geograph, points, cands = _load_test_data()
    gt = analyse_spatial_info(geograph, points, cands, True)
    
    
