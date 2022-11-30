import os
os.environ["USE_PYGEOS"] = "1"

from enum import Enum
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

from .graph import GeoDigraph
from .geo.metric import lcss, edr, erp
from .geo.douglasPeucker import simplify_trajetory_points

from .osmnet.build_graph import build_geograph

from .match.io import load_points
from .match.postprocess import get_path
from .match.candidatesGraph import construct_graph
from .match.spatialAnalysis import analyse_spatial_info
from .match.geometricAnalysis import analyse_geometric_info
from .match.viterbi import process_viterbi_pipeline, find_matched_sequence
from .match.visualization import matching_debug_level, plot_matching, plot_matching_result
from .match.projection import project_traj_points_to_network

from .utils.timer import Timer, timeit
from .utils.logger_helper import make_logger
from .utils.serialization import save_checkpoint
from .utils.misc import SET_PANDAS_LOG_FORMET
from .setting import DATA_FOLDER, DEBUG_FOLDER, DIS_FACTOR

SET_PANDAS_LOG_FORMET()


class CODE:
    SUCCESS       = 0 # 成功匹配
    SAME_LINK     = 1 # 所有轨迹点位于同一条线上
    ONE_POINT     = 2 # 所有轨迹点位于同一个点上
    NO_CANDIDATES = 3 # 轨迹点无法映射到候选边上
    FAILED        = 4 # 匹配结果，prob低于阈值


class ST_Matching():
    def __init__(self,
                 net: GeoDigraph,
                 dp_thres=5,
                 max_search_steps=2000,
                 max_search_dist=10000,
                 top_k_candidates=5,
                 cand_search_radius=50,
                 crs_wgs=4326,
                 crs_prj=900913,
                 prob_thres=.8
                 ):
        self.net = net
        self.dp_thres = dp_thres
        self.crs_wgs = crs_wgs
        self.crs_wgs = crs_wgs
        self.dis_factor = DIS_FACTOR
        self.debug_folder = DEBUG_FOLDER
        self.logger = make_logger('../log', console=False, level="INFO")
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
                       simplify=True, tolerance=10, plot=False, save_fn=None, 
                       debug_in_levels=False, details=False): 
        info = {'status': CODE.SUCCESS} # 0, 成功匹配；1 所有节点位于同一条线上; 2 所有节点位于同一个点上 3 没有候选节点；
        
        # simplify trajectory: (tolerance, 10 meters)
        if simplify:
            ori_traj = traj
            traj = traj.copy()
            traj = self._simplify(traj, tolerance=tolerance)

        # geometric analysis
        top_k = top_k if top_k is not None else self.top_k_candidates
        cands = analyse_geometric_info(
            points=traj, edges=self.net.df_edges, top_k=top_k, radius=self.cand_search_radius)
        
        # is_valid
        s, route = self.is_valid(traj, cands, info)
        if not s:
            return route, info

        # spatial analysis
        if not beam_search:
            graph = analyse_spatial_info(self.net, traj, cands, dir_trans)
            prob, rList = process_viterbi_pipeline(cands, graph[['pid_1', 'dist_prob']])
        else:
            graph = construct_graph(traj, cands, dir_trans=dir_trans)
            prob, rList, graph = find_matched_sequence(cands, graph, self.net, dir_trans)
        
        route, conns, steps = get_path(self.net, traj, rList, graph, cands)
    
        if steps is not None:
            coef = 1 / len(steps.dist_prob)
            dist_prob = np.prod(steps.dist_prob)
            trans_prob = np.prod(steps.trans_prob)
            prob, dist_prob, trans_prob = np.power([prob, dist_prob, trans_prob], coef)
        
            info["prob"] = prob
            info["trans_prob"] = trans_prob
            info["dist_prob"] = dist_prob
            if trans_prob < self.prob_thres:
                info['status'] = CODE.FAILED

            if dir_trans:
                info['dir_prob'] = trans_prob / dist_prob
        else:
            info["status"] = CODE.SAME_LINK

        if details:
            _dict = {
                'cands': cands,
                'rList': rList,
                'graph': graph,
                'route': route,
                "steps": steps,
            }
            info.update(_dict)

        if plot or save_fn:
            # ax = plot_matching(self.net, traj, cands, route, satellite=False, save_fn=save_fn)
            fig, ax = plot_matching_result(traj, route, self.net)
            if simplify:
                ori_traj.plot(ax=ax, color='gray', alpha=.3)
            if not plot:
               plt.close() 

        if debug_in_levels:
            self.matching_debug(traj, graph)
        
        return route, info

    def is_valid(self, traj, cands, info):
        # -> status, route
        if cands is None:
            info['status'] = CODE.NO_CANDIDATES
            return False, None
        
        # Only one single point matched
        if traj.shape[0] == 1 or cands.pid.nunique() == 1: 
            eid = cands.sort_values('dist_p2c').head(1).eid.values
            route = self.net.get_edge(eid, reset_index=True)
            route.loc[:, 'geometry'] = Point(*cands.iloc[0].projection)
            info['status'] = CODE.ONE_POINT
            
            return False, route
        
        return True, None

    def eval(self, traj, path, eps=10, metric='lcss', g=None):
        assert metric in ['lcss', 'edr', 'erp']
        path_points = np.concatenate(path.geometry.apply(lambda x: x.coords[:]).values)
        traj_points = np.concatenate(traj.geometry.apply(lambda x: x.coords[:]).values)
        
        # FIXME 是否使用 轨迹节点 和 投影节点 作比较
        # projected_points = info['rList'][['pid', 'eid']].merge(info['cands'], on=['pid', 'eid'])
        # points = np.concatenate(projected_points.point_geom.apply(lambda x: x.coords[:]).values)
        # projections = np.concatenate(projected_points.projection.apply(lambda x: x).values).reshape((-1, 2))

        eval_funs = {
            'lcss': [lcss, (traj_points, path_points, eps)], 
            'edr': [edr, (traj_points, path_points, eps)], 
            'edp': [erp, (traj_points, path_points, g)]
        }
        _eval = eval_funs[metric]

        return _eval[0](*_eval[1])

    def project(self, traj_panos, path, keep_attrs=None):
        return project_traj_points_to_network(traj_panos, path, self.net, keep_attrs)

    def load_points(self, fn, compress=False, dp_thres: int = None,
                    crs: int = None, in_sys: str = 'wgs', out_sys: str = 'wgs'):
        
        traj, _ = load_points(fn, compress, dp_thres, crs, in_sys, out_sys)
        
        return traj

    def _simplify(self, points:gpd.GeoDataFrame, tolerance:int=None, inplace=False):        
        return simplify_trajetory_points(points, tolerance, inplace=True, logger=self.logger)

    def matching_debug(self, traj, graph, debug_folder='../debug'):
        """matching debug

        Args:
            traj ([type]): Trajectory
            tList ([type]): [description]
            graph_t ([type]): [description]
            net ([Digraph_OSM]): [description]
            debug (bool, optional): [description]. Defaults to True.
        """
        graph = gpd.GeoDataFrame(graph)
        graph.geometry = graph.whole_path
        # graph.set_geometry('whole_path', inplace=True)

        layer_ids = graph.index.get_level_values(0).unique().sort_values().values
        for layer in layer_ids:
            df_layer = graph.loc[layer]
            matching_debug_level(self.net, traj, df_layer, layer, debug_folder)
        
        return

    def plot_result(self, traj, route):
        fig, ax = plot_matching_result(traj, route, self.net)

        return fig

    def get_points(self, traj, ids):
        return NotImplementedError
    
    def points_to_polyline(self, points):
        return NotImplementedError

    def polyline_to_points(self, polyline):
        return NotImplementedError


if __name__ == "__main__":
    net = build_geograph(ckpt = DATA_FOLDER / 'network/Shenzhen_graph_pygeos.ckpt')
    self = ST_Matching(net=net)
    
    traj = self.load_points(DATA_FOLDER / "trajs/traj_0.geojson")
    path, info = self.matching(traj, plot=True, top_k=5)
    
