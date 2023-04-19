import numpy as np
import pandas as pd

from ..utils import timeit
from .status import CANDS_EDGE_TYPE
from ..geo.azimuth import cal_coords_seq_azimuth
from ..geo.ops.distance import coords_seq_distance
from ..geo.ops.to_array import points_geoseries_2_ndarray


def cal_traj_params(points, move_dir=True, check=False):
    coords = points_geoseries_2_ndarray(points.geometry)
    dist_arr, _ = coords_seq_distance(coords)
    idxs = points.index
    
    if check:
        zero_idxs = np.where(dist_arr==0)[0]
        if len(zero_idxs):
            print(f"Exists dumplicates points: {[(i, i+1) for i in zero_idxs]}")
        
    _dict = {'pid_0': idxs[:-1],
             'pid_1': idxs[1:],
             'd_euc': dist_arr}

    if move_dir:
        dirs = cal_coords_seq_azimuth(coords)
        _dict['move_dir'] = dirs
    
    res = pd.DataFrame(_dict)

    return res

def identify_edge_flag(gt:pd.DataFrame):
    """ Identify the type that querying shortest path from candidate `src` to `dst`. 
    Refs: Fast map matching, an algorithm integrating hidden Markov model with 
    precomputation, Fig 4

    Args:
        gt (pd.DataFrame): graph

    Returns:
        pd.DataFrame: The graph appended `flag`
    """
    # (src, dst) on the same edge
    gt.loc[:, 'flag'] = CANDS_EDGE_TYPE.NORMAL

    same_edge = gt.eid_0 == gt.eid_1
    cond = (gt['dist_0'] - gt['step_0_len']) <= gt['step_n_len']

    same_edge_normal = same_edge & cond
    gt.loc[same_edge_normal, 'flag'] = CANDS_EDGE_TYPE.SAME_SRC_FIRST
    gt.loc[same_edge_normal, ['src', 'dst']] = gt.loc[same_edge_normal, ['dst', 'src']].values

    same_edge_revert = same_edge & (~cond)
    gt.loc[same_edge_revert, 'flag'] = CANDS_EDGE_TYPE.SAME_SRC_LAST

    return gt

@timeit
def construct_graph( points,
                     cands,
                     common_attrs = ['pid', 'eid', 'dist', 'speed'], # TODO 'speed' 
                     left_attrs = ['dst', 'len_1', 'seg_1'], # 'dist'
                     right_attrs = ['src', 'len_0', 'seg_0', 'observ_prob'],
                     rename_dict = {
                            'seg_0': 'step_n',
                            'len_0': 'step_n_len',
                            'seg_1': 'step_0',
                            'len_1': 'step_0_len',
                            'cost': 'd_sht'},
                     dir_trans = True,
                     gt_keys = ['pid_0', 'eid_0', 'eid_1']
    ):
    """
    Construct the candiadte graph (level, src, dst) for spatial and temporal analysis.

    Parameters:
        path = step_0 + step_1 + step_n
    """
    layer_ids = np.sort(cands.pid.unique())
    prev_layer_dict = {cur: layer_ids[i]
                          for i, cur in enumerate(layer_ids[1:])}
    prev_layer_dict[layer_ids[0]] = -1

    # left
    left = cands[common_attrs + left_attrs]
    left.loc[:, 'mgd'] = left.pid

    # right
    right = cands[common_attrs + right_attrs]
    right.loc[:, 'mgd'] = right.pid.apply(lambda x: prev_layer_dict[x])
    right.query("mgd >= 0", inplace=True)

    # Cartesian product
    gt = left.merge(right, on='mgd', suffixes=["_0", '_1'])\
             .drop(columns='mgd')\
             .reset_index(drop=True)\
             .rename(columns=rename_dict)

    identify_edge_flag(gt)
    traj_info = cal_traj_params(points.loc[cands.pid.unique()], move_dir=dir_trans)
    
    gt = gt.merge(traj_info, on=['pid_0', 'pid_1'])
    gt.loc[:, ['src', 'dst']] = gt.loc[:, ['src', 'dst']].astype(np.int64)

    if gt_keys:
        gt.set_index(gt_keys, inplace=True)
    
    return gt

