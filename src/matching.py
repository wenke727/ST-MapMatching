#%%
import os
import pandas as pd
import geopandas as gpd

# from DigraphOSM import DigraphOSM

from utils.timer import Timer
from graph import GeoDigraph
from geo.coord.coordTransfrom_shp import coord_transfer
from geo.douglasPeucker import dp_compress_for_points as dp_compress
from setting import DEBUG_FOLDER, DIS_FACTOR

from match.postprocess import get_path
from osmnet.build_graph import build_geograph
from match.viterbi import process_viterbi_pipeline
from match.spatialAnalysis import analyse_spatial_info
from match.geometricAnalysis import analyse_geometric_info
from match.visualization import matching_debug_level, plot_matching

pd.set_option('display.width', 5000)        # 打印结果不换行方法
from utils.logger_helper import make_logger

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


    def matching(self, traj, top_k=None, dir_trans=False, plot=True, plot_scale=.2, debug_in_levels=False):
        timer = Timer()
        cands = analyse_geometric_info(points=traj, 
                               edges=self.net.df_edges, 
                               top_k=top_k if top_k is not None else self.top_k_candidates, 
                               radius=self.cand_search_radius,
                               edge_keys=['way_id', 'dir'], 
                               edge_attrs=['src', 'dst', 'way_id', 'dir', 'geometry'],
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
            route = self.net.get_edge(eid)
            return route
        
        cands, graph = analyse_spatial_info(self.net, traj, cands, dir_trans)
        # rList_ = find_matched_sequence(cands, graph[['pid_1', 'f']])
        rList  = process_viterbi_pipeline(cands, graph[['pid_1', 'f']])
        
        timer.start()
        route, conns = get_path(self.net, traj, rList, graph, cands)
        print(f"Get path: {timer.stop():.4f} s")
        
        if plot:
            plot_matching(self.net, traj, cands, route, plot_scale=plot_scale, satellite=False)

        if debug_in_levels:
            self.matching_debug(traj, graph, True)
        
        return route

  
    """ debug helper """
    def matching_debug(self, traj, graph, save=True):
        """matching debug

        Args:
            traj ([type]): Trajectory
            tList ([type]): [description]
            graph_t ([type]): [description]
            net ([Digraph_OSM]): [description]
            debug (bool, optional): [description]. Defaults to True.
        """
        # create geometry
        self.__graph_path_2_polyline(graph)
        graph = gpd.GeoDataFrame(graph)

        layer_ids = graph.index.get_level_values(0).unique().sort_values().values
        for layer in layer_ids:
            df_layer = graph.loc[layer]
            matching_debug_level(traj, df_layer, layer, save)
        
        return


    """ aux functions """
    def __graph_path_2_polyline(self, graph, overwrite=False):
        if 'geometry' in list(graph) and not overwrite:
            return graph

        graph.loc[:, 'geometry'] = graph.path.apply(self.net.transform_node_seq_to_polyline)

        return graph


#%%
if __name__ == "__main__":
    # FIXME 12的版本不对, 可能是因为因为 geometry 数据不同导致的
    # net = build_geograph(ckpt='../cache/Shenzhen_graph_12.0.ckpt')
    net = build_geograph(ckpt='../cache/Shenzhen_graph_9.ckpt')
    self = ST_Matching(net=net)

    # # github演示数据
    # 真实车辆轨迹
    traj = self.load_points("../input/traj_1.geojson")
    path = self.matching(traj, plot=True, dir_trans=True, debug_in_levels=False)

    # github演示数据
    traj = self.load_points("/home/pcl/codes/ST-MapMatching/test/data/traj_debug_199.geojson")
    path = self.matching(traj, plot=True, dir_trans=True, debug_in_levels=False)
    
    # github演示数据
    traj = self.load_points("../input/traj_0.geojson")
    path = self.matching(traj, plot=True, dir_trans=True, debug_in_levels=False)


#%%
    # traj = matcher.load_points("../input/traj_0.geojson")
    # path = matcher.matching(traj, plot=True, dir_trans=True, debug_in_levels=False)
 
    # from db.db_process import gdf_to_postgis
    # net.df_edges.loc[:, 'eid'] = net.df_edges.index
    # gdf_to_postgis(net.df_edges, 'topo_osm_shenzhen_edge')
    # gdf_to_postgis(net.df_nodes, 'topo_osm_shenzhen_node')

    
    # 真实车辆移动轨迹
    # traj = matcher.load_points("../input/traj_1.geojson")
    # path = matcher.matching(traj, plot=True, dir_trans=True, debug_in_levels=False)

