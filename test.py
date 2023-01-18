#%%
from loguru import logger
from tilemap import plot_geodata
from mapmatching import build_geograph, ST_Matching
from mapmatching.setting import DATA_FOLDER
from mapmatching.utils.serialization import save_checkpoint, load_checkpoint

import numpy as np
import pandas as pd

from mapmatching.utils import Timer, timeit
from mapmatching.match.viterbi import cal_prob_func
from mapmatching.match.spatialAnalysis import get_trans_prob_bet_layers


def save_params():
    params = {
        "cands": cands,
        'gt': gt,
        "dir_trans": False
    }

    save_checkpoint(params, "./data/debug/Bug_breakdown.pkl")

net = build_geograph(ckpt='./data/network/Shenzhen_graph_pygeos.ckpt')
matcher = ST_Matching(net=net)
matcher.logger = logger

#%%
def debug_traj_12():
    # FIXME
    """最后的输出和 dir_trans 是否开启有密切关系，当关闭的时候是正确的，当时开启结果就有点诡异，
    但这里还有一个问题就是，底层的地图数据，该路段是单向的。

    Returns:
        _type_: _description_
    """
    tmp = matcher.load_points("./data/trajs/traj_12.geojson")

    revert = False
    if revert:
        tmp = tmp[::-1].reset_index(drop=True)
    res = matcher.matching(tmp, simplify=True, details=True, dir_trans=False, debug_in_levels=False, top_k=5)
    fig, ax = matcher.plot_result(tmp, res)
    path = matcher.transform_res_2_path(res)

    return res['epath']

def debug_case_5():
    tmp = matcher.load_points("./data/case_5.geojson")

    revert = False
    if revert:
        tmp = tmp[::-1].reset_index(drop=True)

    res = matcher.matching(tmp, simplify=False, details=True, dir_trans=False, debug_in_levels=False, top_k=3)
    fig, ax = matcher.plot_result(tmp, res)
    path = matcher.transform_res_2_path(res)

    print(res['epath'])
    print('lcss: ', matcher.eval(tmp, res, metric='lcss', resample=5, eps=10))

    # res['details']['rList']
    # res['details']['simplified_traj']

    return res['details']['steps']

def debug_case_9():
    points = matcher.load_points('./data/trajs/traj_9.geojson', simplify=True)
    tmp = points.reset_index(drop=True)

    res = matcher.matching(tmp, plot=True, details=True, dir_trans=False, debug_in_levels=False, metric='lcss')
    print(res['epath'])

def debug_case_5_all():
    points = matcher.load_points('./data/case_5_all.geojson')
    # tmp = points[50:].reset_index()
    tmp = points[0:].reset_index()

    # tmp = matcher.load_points("./data/case_6.geojson")

    revert = False
    if revert:
        tmp = tmp[::-1].reset_index(drop=True)
    res = matcher.matching(tmp, simplify=False, plot=True, details=True, dir_trans=False, debug_in_levels=False, top_k=5, metric='lcss')
    print(res['epath'])

debug_case_5_all()

#%%
