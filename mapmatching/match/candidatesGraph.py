import numpy as np
import pandas as pd
from haversine import haversine_vector, Unit

from ..utils import timeit
from ..geo.azimuth_helper import azimuthAngle_np


def _cal_traj_params(points, move_dir=True):
    coords = points.geometry.apply(lambda x: [x.x, x.y]).values.tolist()
    coords = np.array(coords)

    dist = haversine_vector(coords[:-1, ::-1], coords[1:, ::-1], unit=Unit.METERS)
    idxs = points.index
    _dict = {'pid_0': idxs[:-1],
             'pid_1':idxs[1:],
             'd_euc': dist}

    if move_dir:
        dirs = azimuthAngle_np(coords[:-1][:,0], coords[:-1][:,1], 
                               coords[1:][:,0], coords[1:][:,1])
        _dict['move_dir'] = dirs
    
    res = pd.DataFrame(_dict)

    return res


def _identify_edge_flag(gt):
    # (src, dst) on the same edge
    # flag: 0 od 不一样；1 od 位于同一条edge上，但起点相对终点位置偏前；2 相对偏后
    gt.loc[:, 'flag'] = 0

    same_edge = gt.eid_0 == gt.eid_1
    cond = (gt.dist - gt.first_step_len) < gt.last_step_len

    same_edge_normal = same_edge & cond
    gt.loc[same_edge_normal, 'flag'] = 1
    gt.loc[same_edge_normal, ['src', 'dst']] = gt.loc[same_edge_normal, ['dst', 'src']].values

    same_edge_revert = same_edge & (~cond)
    gt.loc[same_edge_revert, 'flag'] = 2

    return gt


@timeit
def construct_graph( points,
                     cands,
                     common_attrs=['pid', 'eid'],
                     left_attrs=['dst', 'len_1', 'seg_1', 'dist'],
                     right_attrs=['src', 'len_0', 'seg_0', 'observ_prob'],
                     rename_dict={
                            'seg_0': 'last_step',
                            'len_0': 'last_step_len',
                            'seg_1': 'first_step',
                            'len_1': 'first_step_len',
                            'cost': 'd_sht'},
                     dir_trans=True,
                     gt_keys=['pid_0', 'eid_0', 'eid_1']
    ):
    """
    Construct the candiadte graph (level, src, dst) for spatial and temporal analysis.
    """
    layer_ids = np.sort(cands.pid.unique())
    prev_layer_dict = {cur: layer_ids[i] for i, cur in enumerate(layer_ids[1:]) }
    prev_layer_dict[layer_ids[0]] = -1

    # left
    left = cands[common_attrs + left_attrs]
    left.loc[:, 'mgd'] = left.pid
    
    # right
    right = cands[common_attrs + right_attrs]
    right.loc[:, 'mgd'] = right.pid.apply(lambda x: prev_layer_dict[x])
    right.query("mgd >= 0", inplace=True)
    
    # Cartesian product, 50%+ speed up
    gt = left.merge(right, on='mgd', suffixes=["_0", '_1'])\
             .drop(columns='mgd')\
             .reset_index(drop=True)\
             .rename(columns=rename_dict)
    
    _identify_edge_flag(gt)

    # FIXME There is a situation where the node does not match, 
    # the current strategy is to ignore it, and there is a problem of order
    traj_info = _cal_traj_params(points.loc[cands.pid.unique()], move_dir=dir_trans)
    gt = gt.merge(traj_info, on=['pid_0', 'pid_1'])
    if gt_keys:
        gt.set_index(gt_keys, inplace=True)
    
    return gt

