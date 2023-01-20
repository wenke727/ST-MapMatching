import numpy as np
import pandas as pd
from haversine import haversine_vector, Unit

from ..utils import timeit
from .status import CANDS_EDGE_TYPE
from ..geo.azimuth import azimuthAngle_vector


def _cal_traj_params(points, move_dir=True, check=False):
    # from ..geo.misc import cal_points_geom_seq_distacne
    coords = points.geometry.apply(lambda x: [x.y, x.x]).values.tolist()
    coords = np.array(coords)

    dist = haversine_vector(coords[:-1], coords[1:], unit=Unit.METERS)
    idxs = points.index
    
    if check:
        zero_idxs = np.where(dist==0)[0]
        if len(zero_idxs):
            print(f"Exists dumplicates points: {[(i, i+1) for i in zero_idxs]}")
        
    _dict = {'pid_0': idxs[:-1],
             'pid_1':idxs[1:],
             'd_euc': dist}

    if move_dir:
        dirs = azimuthAngle_vector(coords[:-1, 1], coords[:-1, 0], 
                                   coords[1: , 1], coords[1:, 0])
        _dict['move_dir'] = dirs
    
    res = pd.DataFrame(_dict)

    return res


def _identify_edge_flag(gt):
    # (src, dst) on the same edge
    gt.loc[:, 'flag'] = CANDS_EDGE_TYPE.NORMAL

    same_edge = gt.eid_0 == gt.eid_1
    # FIXME `<` or `<=`
    cond = (gt['dist'] - gt['first_step_len']) <= gt['last_step_len']

    same_edge_normal = same_edge & cond
    gt.loc[same_edge_normal, 'flag'] = CANDS_EDGE_TYPE.SAME_SRC_FIRST
    gt.loc[same_edge_normal, ['src', 'dst']] = gt.loc[same_edge_normal, ['dst', 'src']].values

    same_edge_revert = same_edge & (~cond)
    gt.loc[same_edge_revert, 'flag'] = CANDS_EDGE_TYPE.SAME_SRC_LAST

    return gt


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

    Parameters:
        geometry = 除去 first step 和 last step 后，剩下的中间段
        path = first_step + geometry + last_step

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
    right.loc[:, 'mgd'] = right.pid.apply(
        lambda x: prev_layer_dict[x])
    right.query("mgd >= 0", inplace=True)

    # Cartesian product
    gt = left.merge(right, on='mgd', suffixes=["_0", '_1'])\
             .drop(columns='mgd')\
             .reset_index(drop=True)\
             .rename(columns=rename_dict)

    _identify_edge_flag(gt)

    # There is a situation where the node does not match,             
    # the current strategy is to ignore it, and maybe it has a problem of the order
    traj_info = _cal_traj_params(
        points.loc[cands.pid.unique()], move_dir=dir_trans)
    gt = gt.merge(traj_info, on=['pid_0', 'pid_1'])
    gt.loc[:, ['src', 'dst']] = gt.loc[:, ['src', 'dst']].astype(int)
    if gt_keys:
        gt.set_index(gt_keys, inplace=True)
    
    return gt

