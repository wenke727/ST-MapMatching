#%%
import sys
sys.path.append('../')
from utils.serialization import load_checkpoint
from match.viterbi import *
from tilemap import plot_geodata

import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from geo.azimuth_helper import cal_azimuth_cos_dist_for_linestring, azimuthAngle
from match.spatialAnalysis import cal_dist_prob, cal_dir_prob, merge_steps

from graph.geograph import GeoDigraph
from osmnet.build_graph import build_geograph

net = build_geograph(ckpt='../../cache/Shenzhen_graph_9.ckpt')

#%%
# 加载数据
fn = "../debug/traj_0_data_for_viterbi.pkl"
fn = "../../debug/traj_1_data_for_viterbi.pkl"
# fn = Path(__file__).parent / fn
data = load_checkpoint(fn)

cands = data['cands']
gt = data['graph']
rList = data['rList']
traj = data['traj']

gt.loc[:, 'dst'] = gt.loc[:, 'dst'].astype(int)
gt.loc[:, 'src'] = gt.loc[:, 'src'].astype(int)

#%%
def check_combine_steps(idx, traj, graph):
    fig, ax = plot_geodata(traj, color='r', reset_extent=False)

    gdf = gpd.GeoDataFrame(graph).iloc[[idx]].set_geometry('whole_path')
    gdf.plot(ax=ax, color='blue', alpha=.5)

    _gdf = gpd.GeoDataFrame(graph).iloc[[idx]].set_geometry('geometry')
    _gdf.plot(ax=ax, color='r', linestyle=':', alpha=.5)

check_combine_steps(15, traj, gt)


#%%
# ! 通过 (src, dst) 获得 trans_prob

attrs = ['dst', 'src', 'flag', 'first_step_len', 'last_step_len', 'first_step', 'last_step', 'move_dir', 'd_euc']
_gt = gt.loc[0][attrs]


def get_trans_prob_bet_layers(_gt, net, dir_trans = True):
    ori_index = _gt.index
    _gt = cal_dist_prob(_gt, net)
    _gt.loc[:, 'whole_path'] = merge_steps(_gt)

    if dir_trans:
        cal_dir_prob(_gt, 'whole_path')
        # cal_dir_prob(_gt, 'geometry')
        _gt.loc[:, 'f'] = _gt.v * _gt.f_dir
    else:
        _gt.loc[:, 'f'] = _gt.v

    _gt = _gt[[i for i in list(_gt) if i not in attrs]]
    
    _gt.index = ori_index
    
    return _gt

_gt = get_trans_prob_bet_layers(gt.loc[0][attrs], net)

traj.plot(ax=ax, color='r')

# %%
