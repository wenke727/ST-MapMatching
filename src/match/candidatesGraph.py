import numpy as np
import pandas as pd

from utils.timer import timeit
from geo.azimuth_helper import azimuthAngle, azimuthAngle_np
from geo.haversine import haversine_np, Unit


def _cal_traj_params(points, move_dir=True):
    coords = points.geometry.apply(lambda x: [x.x, x.y]).values.tolist()
    coords = np.array(coords)

    dist = haversine_np(coords[:-1], coords[1:], xy=True, unit=Unit.METERS)
    idxs = points.index
    _dict = {'pid_0': idxs[:-1],
             'pid_1':idxs[1:],
             'd_euc': dist}

    if move_dir:
        dirs = azimuthAngle_np(coords[:-1][:,0], coords[:-1][:,1], 
                               coords[1:][:,0], coords[1:][:,1])
        _dict['move_dir'] = dirs
        # dirs = []
        # for i in range(len(coords)-1):
        #     dirs.append(
        #         azimuthAngle(*coords[i], *coords[i+1])
        #     )
        # assert (res.move_dir.values == azimuthAngle_np(coords[:-1][:,0], coords[:-1][:,1], 
        #                                    coords[1:][:,0], coords[1:][:,1])).all(), "Check azimuth"

    
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
                     common_attrs=['pid', 'eid', 'mgd'],
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
    cands.loc[:, 'mgd'] = 1
    tList = [layer for _, layer in cands.groupby('pid')]
    cands.drop(columns=['mgd'], inplace=True)
    
    # Cartesian product
    gt = []
    for i in range(len(tList)-1):
        a = tList[i][common_attrs + left_attrs]
        b = tList[i+1][common_attrs + right_attrs]
        gt.append(a.merge(b, on='mgd', suffixes=["_0", '_1']).drop(columns='mgd'))
    gt = pd.concat(gt).reset_index(drop=True).rename(columns=rename_dict)
    gt.loc[:, ['dst', 'src']] = gt.loc[:, ['dst', 'src']].astype(int)
    
    _identify_edge_flag(gt)

    # FIXME There is a situation where the node does not match, 
    # the current strategy is to ignore it, and there is a problem of order
    traj_info = _cal_traj_params(points.loc[cands.pid.unique()], move_dir=dir_trans)
    gt = gt.merge(traj_info, on=['pid_0', 'pid_1'])
    if gt_keys:
        gt.set_index(gt_keys, inplace=True)
    
    return gt

