import os
# os.environ["USE_PYGEOS"] = "0"

import numpy as np
from copy import deepcopy
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString

from .graph import GeoDigraph, GeoDigraphLL
from .update_network import check_steps
from .geo.metric import lcss, edr, erp
from .geo.ops import check_duplicate_points
from .geo.ops.point2line import project_points_2_linestrings
from .geo.douglasPeucker import simplify_trajetory_points
from .geo.ops.resample import resample_polyline_seq_to_point_seq, resample_point_seq

from .osmnet.build_graph import build_geograph

from .match.status import STATUS
from .match.io import load_points
from .match.postprocess import get_path
from .match.candidatesGraph import construct_graph
from .match.spatialAnalysis import analyse_spatial_info
from .match.geometricAnalysis import analyse_geometric_info
from .match.viterbi import process_viterbi_pipeline, find_matched_sequence
from .match.visualization import debug_gt_level, plot_matching_result, debug_gt_level_parallel

from .utils.timer import timeit
from .utils.logger_helper import make_logger
from .utils.misc import SET_PANDAS_LOG_FORMET
from .setting import DATA_FOLDER, DEBUG_FOLDER

SET_PANDAS_LOG_FORMET()


class ST_Matching():
    def __init__(self,
                 net: GeoDigraphLL,
                 max_search_steps=2000,
                 max_search_dist=10000,
                 top_k_candidates=5,
                 cand_search_radius=50,
                 crs_wgs=4326,
                 crs_prj=None,
                 prob_thres=.8, 
                 log_folder='./log',
                 console=True,
                 ):
        self.net = net
        self.crs_wgs = crs_wgs
        self.crs_prj = crs_prj
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

    def matching(self, traj, top_k=None, dir_trans=False, beam_search=True,
                 simplify=True, tolerance=5, plot=False, save_fn=None,
                 debug_in_levels=False, details=False, metric=None, 
                 check_duplicate=False, check_topo=False):
        res = {'status': STATUS.UNKNOWN}
        
        # simplify trajectory
        if simplify:
            ori_traj = traj
            traj = traj.copy()
            traj = self._simplify(traj, tolerance=tolerance) # tolerance, 5 meters
        if check_duplicate:
            traj = check_duplicate_points(traj)
            
        # geometric analysis
        top_k = top_k if top_k is not None else self.top_k_candidates
        cands = analyse_geometric_info(
            points=traj, edges=self.net.df_edges, top_k=top_k, radius=self.cand_search_radius)
        
        # is_valid
        s, route = self._is_valid(traj, cands, res)
        if not s:
            return res

        # spatial analysis
        res['probs'] = {}
        rList, graph = self._spatial_analysis(traj, cands, dir_trans, beam_search, metric=res['probs'])
        match_res, steps = get_path(self.net, traj, rList, graph, cands, metric=res['probs'])
        if 'status' in res['probs']:
            res['status'] = res['probs']['status']
            del res['probs']['status']
        res.update(match_res)

        if details or check_topo:
            attrs = ['pid_1', 'first_step_len', 'last_step_len', 'cost', 'w', 'd_euc', 'dist_prob', 'trans_prob', 'observ_prob', 'prob', 
                     'flag', 'status', 'dst', 'src','first_step', 'geometry', 'last_step', 'path', 'epath', 'vpath','dist']

            # print(f"drop_atts: {[i for i in attrs if i not in list(graph) ]}")
            attrs = [i for i in attrs if i in list(graph)]
            _dict = {
                "simplified_traj": traj,
                'cands': cands, 
                'rList': rList, 
                "steps": steps, 
                'graph': graph[attrs], 
                'path': self.transform_res_2_path(res), 
            }
            res['details'] = _dict

        if metric is not None:
            res['metric'] = self.eval(traj, res, metric=metric)
            print(f"{metric}: {res['metric']}")

        if plot or save_fn:
            fig, ax = self.plot_result(traj, res)
            if simplify:
                ori_traj.plot(ax=ax, color='gray', alpha=.5)
                traj.plot(ax=ax, color='yellow', alpha=.5)
            if not plot:
               plt.close()
            if save_fn:
                fig.savefig(save_fn, dpi=300, bbox_inches='tight', pad_inches=0.02)

        if debug_in_levels:
            self.matching_debug(traj, graph)
        
        if check_topo:
            # FIXME 重复操作，直接返回结果即可
            flag = check_steps(self, res, prob_thred=.75, factor=1.2)
            if flag:
                res = self.matching(traj, top_k, dir_trans, beam_search,
                 False, tolerance, plot, save_fn, debug_in_levels, details, metric, 
                 check_duplicate, False)

        return res

    def _is_valid(self, traj, cands, info, eps = 1e-7):
        # -> status, route
        if cands is None:
            info['status'] = STATUS.NO_CANDIDATES
            return False, None
        
        # Only one single point matched
        if traj.shape[0] == 1 or cands.pid.nunique() == 1: 
            eid = cands.sort_values('dist_p2c').head(1).eid.values
            if "projection" in list(cands):
                coord = cands.iloc[0].projection
            else:
                coord = cands.iloc[0]['proj_point'].coords[0]
            res = {'epath': eid, 'step_0': [coord, [coord[0] + eps, coord[1] + eps]]}
            info.update(res)
            info['status'] = STATUS.ONE_POINT
            
            return False, res
        
        return True, None

    def _spatial_analysis(self, traj, cands, dir_trans, beam_search, metric={}):
        if beam_search:
            graph = construct_graph(traj, cands, dir_trans=dir_trans)
            graph_bak = graph.copy()
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
        assert res is not None or path is not None
        assert metric in ['lcss', 'edr', 'erp']
        
        if path is None:
            path = self.transform_res_2_path(res)
        
        if resample:
            _, path_coords_np = resample_polyline_seq_to_point_seq(path.geometry, step=resample)
            _, traj_coords_np = resample_point_seq(traj.geometry, step=resample)
        else:
            path_coords_np = np.concatenate(path.geometry.apply(lambda x: x.coords[:]).values)
            traj_coords_np = np.concatenate(traj.geometry.apply(lambda x: x.coords[:]).values)
            
        eval_funs = {
            'lcss': [lcss, (traj_coords_np, path_coords_np, eps)], 
            'edr': [edr, (traj_coords_np, path_coords_np, eps)], 
            'edp': [erp, (traj_coords_np, path_coords_np, g)]
        }
        _eval = eval_funs[metric]

        return _eval[0](*_eval[1])

    def project(self, points, path, keep_attrs=['eid', 'geometry'], normalized=True, reset_geom=True):
        ps = project_points_2_linestrings(points, path, normalized=normalized, keep_attrs=keep_attrs)

        ps = gpd.GeoDataFrame(pd.concat([points, ps], axis=1))
        if reset_geom:
            ps.loc[:, 'ori_geom'] = points.geometry.apply(lambda x: x.wkt)
            ps.set_geometry('proj_point', inplace=True)
            ps.drop(columns=['geometry'], inplace=True)
        
        return ps

    def load_points(self, fn, simplify=False, tolerance: int = 10,
                    crs: int = None, in_sys: str = 'wgs', out_sys: str = 'wgs'):
        
        traj, _ = load_points(fn, simplify, tolerance, crs, in_sys, out_sys)
        
        return traj

    def _simplify(self, points:gpd.GeoDataFrame, tolerance:int=None, inplace=False):        
        return simplify_trajetory_points(points, tolerance, inplace=True, logger=self.logger)

    def matching_debug(self, traj:gpd.GeoDataFrame, graph:gpd.GeoDataFrame, level:int=None, debug_folder:str='./debug'):
        """_summary_

        Args:
            traj (gpd.GeoDataFrame): _description_
            graph (gpd.GeoDataFrame): _description_
            level (int, optional): _description_. Defaults to None, namely output all layers.
            debug_folder (str, optional): _description_. Defaults to './debug'.
        """
        graph = gpd.GeoDataFrame(graph)
        
        if level is None:
            layer_ids = graph.index.get_level_values(0).unique().sort_values().values
        else:
            layer_ids = level if isinstance(level, list) else [level]

        for idx in layer_ids:
            img = debug_gt_level_parallel(self.net, traj, graph, idx)
            img.save(os.path.join(debug_folder, f"level_{idx}.jpg"))
        
        return img

    def plot_result(self, traj, info):
        info = deepcopy(info)
        if info['status'] == 3:
            path = None
        else:
            path = self.transform_res_2_path(info)

        fig, ax = plot_matching_result(traj, path, self.net)
        if not info:
            return fig, ax

        for att in ['epath', "step_0", "step_n", 'details']:
            if att not in info:
                continue
            info.pop(att)

        text = []
        if "probs" in info:
            probs = info.pop('probs')
            info.update(probs)
        
        for key, val in info.items():
            if isinstance(val, float):
                _str = f"{key}: {val * 100: .2f} %"
            else:
                _str = f"{key}: {val}"
            text.append(_str)

        x0, x1, y0, y1 = ax.axis()
        ax.text(x0 + (x1- x0)/50, y0 + (y1 - y0)/50, "\n".join(text))

        return fig, ax

    def transform_res_2_path(self, res):
        path = self.net.get_edge(res['epath'], reset_index=True)
        path.loc[0, 'geometry'] = LineString(res['step_0'])
        if 'step_n' in res:
            n = path.shape[0] - 1
            path.loc[n, 'geometry'] = LineString(res['step_n'])
        
        path = path[~path.geometry.is_empty]

        return path

    def update(self):
        return NotImplementedError
    


if __name__ == "__main__":
    net = build_geograph(ckpt = DATA_FOLDER / 'network/Shenzhen_graph_pygeos.ckpt')
    self = ST_Matching(net=net)
    
    traj = self.load_points(DATA_FOLDER / "trajs/traj_0.geojson")
    path, info = self.matching(traj, plot=True, top_k=5)
    
