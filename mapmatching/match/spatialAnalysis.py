import warnings
import numpy as np
from geopandas import GeoDataFrame

from ..graph import GeoDigraph
from .dir_similarity import cal_dir_prob
from .candidatesGraph import construct_graph


def cal_dist_prob(gt: GeoDataFrame, net: GeoDigraph, max_steps: int = 2000, max_dist: int = 10000):
    # Add: w, v, path, geometry
    assert 'flag' in gt, "Chech the attribute `flag` in gt or not"
    if gt.empty:
        warnings.warn("Empty graph layer")
        return gt

    sp_attrs = ['cost', "avg_speed", 'epath', 'coords']
    gt_sp_attrs = ['cost', "avg_speed", 'epath', 'step_1']
    rout_planning = lambda x: net.search(x.dst, x.src, max_steps, max_dist)
    paths = gt.apply(rout_planning, axis=1, result_type='expand')[sp_attrs]
    gt.loc[:, gt_sp_attrs] = paths.values

    cal_temporal_prob(gt)

    gt.loc[:, 'd_sht'] = gt.cost + gt.step_0_len + gt.step_n_len 

    # od 位于同一条edge上，但起点相对终点位置偏前
    flag_1_idxs = gt.query("flag == 1").index
    if len(flag_1_idxs):
        gt.loc[flag_1_idxs, ['epath', 'step_1']] = None, None
        # TODO change `cost` -> `-dist_0`
        gt.loc[flag_1_idxs, 'd_sht'] = - gt.dist_0 + gt.step_0_len + gt.step_n_len

    # distance trans prob
    dist = gt.d_euc / gt.d_sht
    mask = dist > 1 
    dist[mask] = 1 / dist[mask]
    gt.loc[:, 'dist_prob'] = dist


    return gt

def cal_temporal_prob(gt: GeoDataFrame):
    speeds = gt[['speed_0', 'speed_1', 'avg_speed']].values
    weights = gt[['step_0_len', 'step_n_len', 'cost']].values
    weights[weights == np.inf] = 0
    avg_speeds = np.average(speeds, weights = weights, axis=1)
    # gt.loc[:, 'eta'] = gt.d_sht.values / avg_speeds
    gt.loc[:, 'avg_speed'] = avg_speeds

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
