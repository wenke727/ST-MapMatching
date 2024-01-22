import os
import shapely
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import box
from copy import deepcopy
import matplotlib.pyplot as plt

from .graph import GeoDigraph
from .update_network import check_steps
from .geo.metric import lcss, edr, erp
from .geo.ops import check_duplicate_points
from .geo.ops.simplify import simplify_trajetory_points
from .geo.ops.resample import resample_polyline_seq_to_point_seq, resample_point_seq

from .osmnet.build_graph import build_geograph

from .match.status import STATUS
from .match.io import load_points
from .match.spatialAnalysis import analyse_spatial_info
from .match.geometricAnalysis import analyse_geometric_info
from .match.postprocess import get_path, project, transform_mathching_res_2_path
from .match.candidatesGraph import construct_graph, get_shortest_geometry
from .match.viterbi import process_viterbi_pipeline, find_matched_sequence
from .match.visualization import plot_matching_result, debug_traj_matching

from .utils.timer import timeit
from .utils.logger_helper import make_logger
from .utils.misc import SET_PANDAS_LOG_FORMET
from .setting import DATA_FOLDER, DEBUG_FOLDER

from typing import List

class ST_Matching():
    def __init__(self,
                 net: GeoDigraph,
                 max_search_steps=2000,
                 max_search_dist=10000,
                 top_k_candidates=5,
                 cand_search_radius=50,
                 crs_wgs=4326,
                 crs_prj=None,
                 prob_thres=.8,
                 log_folder='./log',
                 console=True,
                 ll=False,
                 loc_bias=0,
                 loc_deviaction=100,
                 ):
        self.net = net
        edge_attrs = ['eid', 'src', 'dst', 'way_id', 'dir', 'dist', 'speed', 'geometry']
        # Avoid waste time on created new objects by slicing
        self.base_edges = self.net.df_edges[edge_attrs]
        self.base_edges.sindex

        self.crs_wgs = crs_wgs
        self.crs_prj = crs_prj
        self.ll = ll
        
        self.loc_bias = loc_bias
        self.loc_deviaction = loc_deviaction
        
        self.debug_folder = DEBUG_FOLDER
        self.logger = make_logger(log_folder, console=console, level="INFO")
        if not os.path.exists(self.debug_folder):
            os.makedirs(self.debug_folder)

        # hyper parameters
        self.prob_thres = prob_thres
        self.top_k_candidates = top_k_candidates
        self.cand_search_radius = cand_search_radius
        self.route_planning_max_search_steps = max_search_steps
        self.route_planning_max_search_dist = max_search_dist

    @timeit
    def matching(self, traj, top_k=None, dir_trans=False, beam_search=True,
                 simplify=True, tolerance=5, plot=False, save_fn=None,
                 debug_in_levels=False, details=False, metric=None, 
                 check_duplicate=False, check_topo=False, search_radius=None, 
                 bias=0, deviation=500):
        self.logger.trace("start")
        res = {'status': STATUS.UNKNOWN, 'ori_crs': deepcopy(traj.crs.to_epsg())}

        # simplify trajectory
        _traj = self.align_crs(traj.copy())
        if simplify:
            _traj = self.simplify(_traj, tolerance = tolerance) # tolerance, 5 meters
        elif check_duplicate:
            _traj = check_duplicate_points(_traj)
            
        # geometric analysis
        top_k = top_k if top_k is not None else self.top_k_candidates
        if search_radius is None:
            search_radius = self.cand_search_radius
        cands = analyse_geometric_info(_traj, self.base_edges, top_k, search_radius, 
                                       bias=self.loc_bias, deviation=self.loc_deviaction)
        
        # is_valid
        s, _ = self._is_valid_cands(_traj, cands, res)
        if not s:
            return res

        # spatial analysis
        res['probs'] = {}
        rList, graph = self.spatial_analysis(_traj, cands, dir_trans, beam_search, metric=res['probs'])
        match_res, steps = get_path(rList, graph, cands, metric=res['probs'], prob_thres=.6)
        
        if 'status' in res['probs']:
            res['status'] = res['probs']['status']
            del res['probs']['status']
        res.update(match_res)

        # add details
        if details or check_topo:
            attrs = ['pid_1', 'step_0_len', 'step_n_len', 'cost', 'sp_dist', 'euc_dist', 'dist_prob', 
                     'trans_prob', 'observ_prob', 'prob', 'flag', 'status', 'dst', 'src','step_0', 
                     'geometry', 'step_n', 'path', 'epath', 'vpath', 'dist', 'dist_0', 'step_1']
            attrs = [i for i in attrs if i in list(graph)]
            if 'move_dir' in graph:
                attrs += ['move_dir']

            _dict = {
                "simplified_traj": _traj,
                'cands': cands, 
                'rList': rList, 
                "steps": steps, 
                'graph': graph[attrs], 
                'path': self.transform_res_2_path(res), 
            }
            res['details'] = _dict

        # metric
        if metric is not None:
            res['metric'] = self.eval(_traj, res, metric=metric)
            self.logger.info(f"{metric}: {res['metric']}")

        # plot
        if plot or save_fn:
            fig, ax = self.plot_result(traj, res)
            if plot:
                plt.show()
            else:
                plt.close()
            if save_fn:
                fig.savefig(save_fn, dpi=300, bbox_inches='tight', pad_inches=0.02)

        # debug helper
        if debug_in_levels:
            self.debug_matching_among_levels(_traj, graph)

        # check topo        
        if check_topo:
            flag = check_steps(self, res, prob_thred=.75, factor=1.2)
            if flag:
                res = self.matching(_traj, top_k, dir_trans, beam_search,
                 False, tolerance, plot, save_fn, debug_in_levels, details, metric, 
                 check_duplicate, False)

        return res

    def _is_valid_cands(self, traj, cands, info, eps = 1e-7):
        # -> status, route
        if cands is None:
            info['status'] = STATUS.NO_CANDIDATES
            info['probs'] = {}
            edges_box = box(*self.base_edges.total_bounds)
            traj_box = box(*traj.total_bounds)
            flag = edges_box.contains(traj_box)
            if not flag:
                details = "Please adjust the bbox to contain the trajectory."
                info['detail'] = details
                self.logger.error(details)
            assert flag, "check the bbox contained the trajectory or not"

            return False, None
        
        # Only one single point matched
        if traj.shape[0] == 1 or cands.pid.nunique() == 1: 
            eid = cands.sort_values('dist_p2c').head(1).eid.values
            coord = cands.iloc[0]['proj_point']
            res = {'epath': eid, 'step_0': [coord, [coord[0] + eps, coord[1] + eps]]}
            info.update(res)
            info['status'] = STATUS.ONE_POINT
            info['probs'] = {}
            
            return False, res
        
        return True, None

    def spatial_analysis(self, traj, cands, dir_trans, beam_search, metric={}):
        if beam_search:
            graph = construct_graph(traj, cands, dir_trans=dir_trans)
            prob, rList, graph = find_matched_sequence(cands, graph, self.net, dir_trans)
        else:
            graph = analyse_spatial_info(self.net, traj, cands, dir_trans)
            prob, rList = process_viterbi_pipeline(cands, graph[['pid_1', 'dist_prob']])

        metric['prob'] = prob

        return rList, graph

    def eval(self, traj, res=None, path=None, resample=5, eps=10, metric='lcss', g=None):
        """
        lcss 的 dp 数组 循环部分，使用numba 加速，这个环节可以降低 10% 的时间消耗（20 ms） 
        """
        # FIXME 
        assert res is not None or path is not None
        assert metric in ['lcss', 'edr', 'erp']
        
        if path is None:
            path = self.transform_res_2_path(res)

        if traj.crs.to_epsg() != path.crs.to_epsg():
            traj = traj.to_crs(path.crs.to_epsg())

        if resample:
            _, path_coords_np = resample_polyline_seq_to_point_seq(path.geometry, step=resample,)
            _, traj_coords_np = resample_point_seq(traj.geometry, step=resample)
        else:
            path_coords_np = np.concatenate(path.geometry.apply(lambda x: x.coords[:]).values)
            traj_coords_np = np.concatenate(traj.geometry.apply(lambda x: x.coords[:]).values)
            
        eval_funs = {
            'lcss': [lcss, (traj_coords_np, path_coords_np, eps, self.ll)], 
            'edr': [edr, (traj_coords_np, path_coords_np, eps)], 
            'edp': [erp, (traj_coords_np, path_coords_np, g)]
        }
        _eval = eval_funs[metric]

        return _eval[0](*_eval[1])

    def project(self, points, path, keep_attrs=['eid', 'proj_point'], normalized=True, reset_geom=True):
        return project(points, path, keep_attrs, normalized, reset_geom)

    def load_points(self, fn, simplify=False, tolerance: int = 10,
                    crs: int = None, in_sys: str = 'wgs', out_sys: str = 'wgs'):
        traj, _ = load_points(fn, simplify, tolerance, crs, in_sys, out_sys)
        
        return traj

    def simplify(self, points:gpd.GeoDataFrame, tolerance:int=None, inplace=False):        
        return simplify_trajetory_points(points, tolerance, inplace=True, logger=self.logger)

    def debug_matching_among_levels(self, traj: gpd.GeoDataFrame, graph: gpd.GeoDataFrame, 
                                    level: list=None, debug_folder: str='./debug'):

        return debug_traj_matching(traj, graph, self.net, level, debug_folder)

    def plot_result(self, traj, info):
        info = deepcopy(info)
        if info['status'] == 3:
            path = None
        elif info.get('details', {}).get('path', None) is not None:
            path = info['details']['path']
        else:
            path = self.transform_res_2_path(info)
        
        fig, ax = plot_matching_result(traj, path, self.net)

        return fig, ax

    def transform_res_2_path(self, res, ori_crs=True, attrs=None):
        return transform_mathching_res_2_path(res, self.net, ori_crs, attrs)

    def update(self):
        return NotImplementedError
    
    def align_crs(self, traj):
        return self.net.align_crs(traj)


if __name__ == "__main__":
    net = build_geograph(ckpt = DATA_FOLDER / 'network/Shenzhen_graph_pygeos.ckpt')
    self = ST_Matching(net=net)
    
    traj = self.load_points(DATA_FOLDER / "trajs/traj_0.geojson")
    path, info = self.matching(traj, plot=True, top_k=5)
    
