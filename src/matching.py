#%%
import os
os.environ["USE_PYGEOS"] = "1"

import pandas as pd
import geopandas as gpd

from utils.timer import Timer
from graph import GeoDigraph
from geo.coord.coordTransfrom_shp import coord_transfer
from geo.douglasPeucker import dp_compress_for_points as dp_compress
from setting import DEBUG_FOLDER, DIS_FACTOR

from match.postprocess import get_path
from osmnet.build_graph import build_geograph
from match.candidatesGraph import construct_graph
from match.spatialAnalysis import analyse_spatial_info
from match.geometricAnalysis import analyse_geometric_info
from match.viterbi import process_viterbi_pipeline, find_matched_sequence
from match.visualization import matching_debug_level, plot_matching

from utils.timer import timeit
from utils.logger_helper import make_logger
from utils.serialization import save_checkpoint

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 25)
pd.set_option('display.width', 5000)        # 打印结果不换行方法


#%%

class Trajectory:
    def __init__(self, dp_thres=5, crs_wgs=4326, crs_prj=900913, logger=None):
        self.crs_wgs  = crs_wgs
        self.crs_prj  = crs_prj
        self.traj     = None
        self.logger   = make_logger('../log', console=True) if logger is None else logger
        self.dp_thres = dp_thres


    def load_points(self, fn, compress=True, dp_thres:int=None, crs:int=None, in_sys:str='wgs', out_sys:str='wgs'):
        traj = gpd.read_file(fn, encoding='utf-8')
        if crs is not None:
            traj.set_crs(crs, allow_override=True , inplace=True)

        if 'time' in traj.columns:
            traj.time = pd.to_datetime(traj['time'], format='%Y-%m-%d %H:%M:%S')
        
        traj = coord_transfer(traj, in_sys, out_sys)

        if compress is not None:
            self.traj_bak = traj.copy()
            self.traj = traj = self._simplify(traj, dp_thres, inplace=True)
        else:
            self.traj_bak = None
            self.traj = traj
        
        return traj


    def _simplify(self, points:gpd.GeoDataFrame, tolerance:int=None, inplace=False):
        """The algorithm (Douglas-Peucker) recursively splits the original line into smaller parts 
        and connects these parts’ endpoints by a straight line. Then, it removes all points whose 
        distance to the straight line is smaller than tolerance. It does not move any points and 
        it always preserves endpoints of the original line or polygon.

        Args:
            points (gpd.GeoDataFrame): _description_
            traj_thres (int, optional): The compression threshold(Unit: meter). Defaults to None.
            inplace (bool, optional): _description_. Defaults to False.

        Returns:
            gpd.GeoDataFrame: _description_
        """
        ori_size = points.shape[0]
        if ori_size == 1:
            return points
        
        tolerance = self.dp_thres if tolerance is None else tolerance
        
        points = points if inplace else points.copy()
        points = dp_compress(points, dis_thred=tolerance)
        
        if ori_size == 2:
            if points.iloc[0].geometry.distance(points.iloc[1].geometry) < 1e-4:
                points = points.head(1)
                self.logger.info(f"Trajectory only has one point or all the same points.")
                return points
      
        self.logger.debug(f"Trajectory compression rate: {points.shape[0]/ori_size*100:.1f}% ({ori_size} -> {points.shape[0]})")
        
        return points
    

    def get_points(self, traj, ids):
        return NotImplementedError
    
    
    def points_to_polyline(self, points):
        return NotImplementedError


    def polyline_to_points(self, polyline):
        return NotImplementedError


class ST_Matching(Trajectory):
    def __init__(self, 
                 net:GeoDigraph, 
                 dp_thres=5, 
                 max_search_steps=2000, 
                 max_search_dist=10000,
                 top_k_candidates=5,
                 cand_search_radius=50,
                 crs_wgs=4326, 
                 crs_prj=900913, 
                 ):
        self.crs_wgs        = crs_wgs
        self.crs_prj        = crs_prj
        self.logger         = make_logger('../log', console=False, level="INFO")
        self.traj_processor = Trajectory(dp_thres, crs_wgs, crs_prj, self.logger)
        self.net            = net
        self.dis_factor     = DIS_FACTOR
        self.debug_folder   = DEBUG_FOLDER
        if not os.path.exists(self.debug_folder):
            os.makedirs(self.debug_folder)

        # hyper parameters
        self.top_k_candidates = top_k_candidates
        self.cand_search_radius = cand_search_radius
        self.route_planning_max_search_steps = max_search_steps
        self.route_planning_max_search_dist = max_search_dist


    def load_points(self, fn, compress=True, dp_thres: int = None, crs: int = None, in_sys: str = 'wgs', out_sys: str = 'wgs'):
        return self.traj_processor.load_points(fn, compress, dp_thres, crs, in_sys, out_sys)

    @timeit
    def matching(self, traj, top_k=None, dir_trans=False, plot=True, plot_scale=.2, debug_in_levels=False, beam_search=True):
        cands = analyse_geometric_info(points=traj, 
                               edges=self.net.df_edges, 
                               top_k=top_k if top_k is not None else self.top_k_candidates, 
                               radius=self.cand_search_radius,
                               edge_keys=[], 
                               edge_attrs=['src', 'dst', 'way_id', 'dir', 'dist', 'geometry'],
                               point_to_line_attrs=['len_0', 'len_1', 'seg_0', 'seg_1'],
                               pid='pid',
                               eid='eid',
                               ll=True,
                               crs_wgs=self.crs_wgs,
                               crs_prj=self.crs_prj
                )
        
        # No matched
        if cands is None:
            return None
        
        # Only one single point matched
        if traj.shape[0] == 1 or cands.pid.nunique() == 1: 
            eid = cands.sort_values('dist_p2c').head(1).eid.values
            route = self.net.get_edge(eid, reset_index=True)
            return route
        
        if not beam_search:
            graph = analyse_spatial_info(self.net, traj, cands, dir_trans)
            rList  = process_viterbi_pipeline(cands, graph[['pid_1', 'f']])
        else:
            gt_keys = ['pid_0', 'eid_0', 'eid_1']
            graph = construct_graph(traj, cands, dir_trans=dir_trans)
            graph = graph.drop_duplicates(gt_keys).set_index(gt_keys).sort_index()
            
            attrs = ['pid_1', 'dst', 'src', 'flag', 'first_step_len', 'last_step_len', 'first_step', 'last_step', 'd_euc', 'observ_prob']
            if dir_trans:
                attrs += ['move_dir']
            _, rList, graph = find_matched_sequence(cands, graph[attrs], self.net, dir_trans)
        
        route, conns = get_path(self.net, traj, rList, graph, cands)
        
        if plot:
            plot_matching(self.net, traj, cands, route, plot_scale=plot_scale, satellite=False)

        if debug_in_levels:
            self.matching_debug(traj, graph)
        
        _dict = {
            'cands': cands,
            'rList': rList,
            'graph': graph,
            'route': route
        }
        
        return route, _dict

  
    """ debug helper """
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



#%%
if __name__ == "__main__":
    net = build_geograph(ckpt='../cache/Shenzhen_graph_9_pygeos.ckpt')
    # net = build_geograph(ckpt='../cache/GBA_graph_9_pygeos.ckpt')
    self = ST_Matching(net=net)
    
    # github演示数据
    traj = self.load_points("../test/data/traj_debug_case2.geojson", dp_thres=20)
    path, info = self.matching(traj, top_k=5, plot=False, dir_trans=True, debug_in_levels=False)
    
    from tilemap import plot_geodata
    plot_geodata(path, color='r')
    
    # from db.db_process import gdf_to_geojson
    # gdf_to_geojson(traj, '../test/data/trip_amp_hw_compressed.geojson')

    # # github演示数据
    traj = self.load_points("../input/traj_0.geojson")
    path, rList = self.matching(traj, top_k=5, plot=True, dir_trans=True, debug_in_levels=False)
    

    traj = self.load_points("/home/pcl/codes/ST-MapMatching/test/data/traj_debug_141.geojson")
    path = self.matching(traj, plot=True, top_k=5, dir_trans=True, debug_in_levels=True)
    
    # # github演示数据
    # traj = self.load_points("/home/pcl/codes/ST-MapMatching/test/data/traj_debug_199.geojson")
    # path = self.matching(traj, plot=True, dir_trans=True, debug_in_levels=False)

    # data = {
    #     "cands": cands, 
    #     "graph": graph,
    #     'rList': rList,
    #     "traj": traj
    # }
    # save_checkpoint(data, "../debug/traj_0_data_for_viterbi.pkl")

