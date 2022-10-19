#%%
import os
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString, box

# from DigraphOSM import DigraphOSM
from utils.logger_helper import make_logger

from graph import GeoDigraph
from geo.geo_plot_helper import map_visualize
from geo.coord.coordTransfrom_shp import coord_transfer
from geo.douglasPeucker import dp_compress_for_points as dp_compress
from geo.geo_helper import coords_pair_dist, point_to_polyline_process, geom_series_distance
from geo.azimuth_helper import azimuth_cos_similarity_for_linestring, azimuthAngle

from setting import DEBUG_FOLDER, DIS_FACTOR

from osmnet.build_graph import build_geograph
from match.geometricAnalysis import get_k_neigbor_edges
from match.spatialAnalysis import analyse_spatial_info

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
        """The algorithm (Douglas-Peucker) recursively splits the original line into smaller parts and connects these parts’ endpoints by a straight line. Then, it removes all points whose distance to the straight line is smaller than tolerance. It does not move any points and it always preserves endpoints of the original line or polygon.

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
        cands = get_k_neigbor_edges(points=traj, 
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
        rList        = self.find_matched_sequence(cands, graph)
        route, conns  = self.get_path(traj, rList, graph, cands)

        if plot:
            self._plot_matching(traj, cands, route, plot_scale=plot_scale, satellite=False)

        if debug_in_levels:
            self.matching_debug(traj, graph, True)
        
        return route


    def temporal_anylysis(self,):
        # take the similar speed conditions into account
        # the cosine distance is used to measure the similarity between the actual average speed from `ci-1` to `ci`
        # and the speed constraints of the path
        
        return NotImplementedError
    
    
    def find_matched_sequence(self, cands:gpd.GeoDataFrame, graph:gpd.GeoDataFrame):
        prev_dict, f_score = {}, {}
        layer_ids = graph.index.get_level_values(0).unique().sort_values().values

        for i, item in cands.query(f"pid=={layer_ids[0]}").iterrows():
            f_score[i] = item.observ_prob

        for cur_layer_id in layer_ids:
            df_layer     = graph.loc[cur_layer_id]
            nxt_layer_id = df_layer.pid_1.unique()[0]
            cur_eids     = cands.query(f"pid == {cur_layer_id}").eid
            nxt_eids     = cands.query(f"pid == {nxt_layer_id}").eid

            for j, nxt in nxt_eids.iteritems():
                _max = -np.inf
                for i, cur in cur_eids.iteritems():
                    info = df_layer.loc[cur].loc[nxt].to_dict()
                    _f = info['f']
                    if _f > 1.001:
                        self.logger.warning(f"level {i}->{j}({cur}, {nxt}), F value {_f:.3f}, exceed 1.\n\t{info}")
                        
                    alt = f_score[i] + _f
                    if alt > _max:
                        _max = alt
                        prev_dict[j] = i
                    f_score[j] = _max
                
        rList = []
        c = max(f_score, key=lambda x: (f_score.get(x), x))
        for i in range(len(layer_ids), 0, -1):
            rList.append(c)
            c = prev_dict[c]
        rList.append(c)
        rList = cands.loc[rList[::-1]][['pid', 'eid', 'src', 'dst']]

        self.logger.info(f'max score: {c}, f_score: {f_score}\n{rList}')
        
        return rList


    def get_path(self, traj, rList, graph, cands, plot=False):
        """Get path by matched sequence node.

        Args:
            rList ([type]): [description]
            graph_t ([type]): [description]
            net ([type]): [description]

        Returns:
            [type]: [description]
        """
        if rList.shape[0] == 1:
            return self.net.merge_edge(rList, on=['src', 'dst']), None
        
        def _helper(x):
            res = graph.loc[x.pid].loc[x.eid].loc[x.nxt_eid].path
            return res if res is None else res
        
        rList.loc[:, 'nxt_eid'] = rList.eid.shift(-1).fillna(0).astype(np.int)
        steps = rList[:-1].apply(lambda x: _helper(x), axis=1)
        # Drop `Consecutive identical line segments`, and keep the first one record
        steps = steps[steps != steps.shift(1)]

        od_lst = [rList.iloc[0]['src']]
        for step in steps.values:
            if step is None or isinstance(step, np.float):
                continue
            
            if step[0] == od_lst[-1]:
                od_lst += step[1:]
            else:    
                od_lst += step
        od_lst += [rList.iloc[-1]['dst']]
        path = self.net.transform_node_seq_to_df_edge(od_lst)
        
        # update geometry of the first/last step 
        step_0 = cands.query(f'pid == {rList.iloc[0].pid} and eid == {rList.iloc[0].eid}').seg_1.values[0]
        step_n = cands.query(f'pid == {rList.iloc[-1].pid} and eid == {rList.iloc[-1].eid}').seg_0.values[0]
        n = path.shape[0] - 1
        if n == 0:
            coords = np.concatenate((step_0[0][np.newaxis, :], 
                                     step_0[[p in step_n for p in step_0]], 
                                     step_n[-1][np.newaxis, :]))
            path.loc[0, 'geometry'] = LineString(coords)
        else:
            path.loc[0, 'geometry'], path.loc[n, 'geometry'] = LineString(step_0), LineString(step_n)
            # filter empty geometry
            path = path[~path.geometry.is_empty]
            path.loc[0, 'memo'], path.loc[n, 'memo'] = 'first step', 'last step'
        

        # connector
        p_0, p_n = traj.iloc[0].geometry, traj.iloc[-1].geometry
        # BUG path 的 geometry 为空
        try:
            connector_0 = LineString([(p_0.x, p_0.y), path.loc[0, 'geometry'].coords[0]])
        except:
            connector_0 = LineString([(p_0.x, p_0.y), (p_0.x, p_0.y)])
        try:
            connector_1 = LineString([path.loc[n, 'geometry'].coords[-1], (p_n.x, p_n.y)])
        except:
            connector_1 = LineString([(p_n.x, p_n.y), (p_n.x, p_n.y)])
            
        connectors = gpd.GeoDataFrame({
            'geometry': [
                connector_0, 
                connector_1], 
            'name':['connector_0', 'connector_1']})
        
        if plot:
            fig, ax = map_visualize(path, scale=.01)
            connectors.plot(ax=ax)
        
        return path, connectors
    
    
    def _plot_matching(self, traj, cands, route, save_fn=None, satellite=True, column=None, categorical=True, plot_scale=.2):
        def _base_plot():
            if column is not None and column in traj.columns:
                ax = traj.plot(alpha=.3, column=column, categorical=categorical, legend=True)
            else:
                ax = traj.plot(alpha=.3, color='black')
            ax.axis('off')
            
            return ax
        
        # plot，trajectory point
        if satellite:
            try:
                from tilemap import plot_geodata
                _, ax = plot_geodata(traj, alpha=.3, color='black', extra_imshow_args={'alpha':.5}, reset_extent=True)
                if column is not None:
                    traj.plot(alpha=.3, column=column, categorical=categorical, legend=True, ax=ax)
            except:
                ax = _base_plot()       
        else:
            ax = _base_plot()
            
        traj.plot(ax=ax, color='blue', alpha=.5, label= 'Compressed')
        traj.head(1).plot(ax=ax, marker = '*', color='red', zorder=9, label= 'Start point')
        # network
        edge_lst = self.net.spatial_query(box(*traj.total_bounds))
        self.net.get_edge(edge_lst).plot(ax=ax, color='black', linewidth=.8, alpha=.4, label='Network' )
        # candidate
        self.net.get_edge(cands.eid.values).plot(
            ax=ax, label='Candidates', color='blue', linestyle='--', linewidth=.8, alpha=.5)
        # route
        if route is not None:
            route.plot(ax=ax, label='Path', color='red', alpha=.5)
        
        ax.axis('off')
        if column is None:
            plt.legend(loc='best')
        
        if save_fn is not None:
            plt.tight_layout()
            plt.savefig(f'{save_fn}.jpg' if '.jpg' not in save_fn else save_fn, dpi=300)
            # plt.close()
        
        return ax


    """ debug helper """
    def __matching_debug_subplot(self, traj, item, level, src, dst, ax=None, legend=True, scale=.9):
        """Plot the matching situation of one pair of od.

        Args:
            item (pandas.core.series.Series): One record in tList. The multi-index here is (src, dest).
            net ([type], optional): [description]. Defaults to net.
            ax ([type], optional): [description]. Defaults to None.
            legend (bool, optional): [description]. Defaults to True.

        Returns:
            ax: Ax.
        
        Example:
            matching_debug_subplot(graph_t.loc[1])
        """
        if ax is None:
            _, ax = map_visualize(traj, scale=scale, alpha=.6, color='white')
        else:
            map_visualize(traj, scale=scale, alpha=.6, color='white', ax=ax)

        # OD
        traj.loc[[level]].plot(ax=ax, marker="*", label=f'O ({src})', zorder=9)
        traj.loc[[item.pid_1]].plot(ax=ax, marker="s", label=f'D ({dst})', zorder=9)

        # path
        self.net.get_edge([src]).plot(ax=ax, linestyle='--', alpha=.8, label=f'first({src})', color='green')
        gpd.GeoDataFrame( item ).T.plot(ax=ax, color='red', label='path')
        self.net.get_edge([dst]).plot(ax=ax, linestyle='-.', alpha=.8, label=f'last({dst}, {item.observ_prob:.2f})', color='black')

        # aux
        if 'f_dir' in item:
            info = f"{item.f:.3f} ({item.observ_prob:.3f}, {item.v:.3f}, {item.f_dir:.3f})"
        else:
            info = f"{item.f:.3f} ({item.observ_prob:.3f}, {item.v:.3f})"

        ax.set_title(
            f"{src} -> {dst}: {info}", 
            color = 'black' if item.f < 0.7 else 'red' 
        )
        ax.set_axis_off()

        if legend: ax.legend()
        
        return ax
        

    def __matching_debug_level(self, traj, df_layer, layer_id, save=False):
        """PLot the matchings between levels (i, i+1)

        Args:
            traj ([type]): [description]
            tList ([type]): The candidate points.
            graph_t ([type]): [description]
            level ([type]): [description]
            net ([type]): [description]
            debug (bool, optional): [description]. Save or not

        Returns:
            [type]: [description]
        """

        rows = df_layer.index.get_level_values(0).unique()
        cols = df_layer.index.get_level_values(1).unique()
        n_cols, n_rows = len(rows), len(cols)

        plt.figure(figsize=(5*n_cols, 5*n_rows))
        for i, src in enumerate(rows):
            for j, dst in enumerate(cols):
                ax = plt.subplot(n_rows, n_cols, i * n_rows + j + 1) 
                self.__matching_debug_subplot(traj, df_layer.loc[src].loc[dst], layer_id, src, dst, ax=ax)

        plt.suptitle(f'Level: {layer_id} (observ, dis, dir)')
        plt.tight_layout()
        
        if save:
            t = time.strftime("%Y%m%d_%H", time.localtime()) 
            plt.savefig( os.path.join(self.debug_folder, f"{t}_level_{layer_id}.jpg"), dpi=300)
            plt.close()
            
        return True


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
            self.__matching_debug_level(traj, df_layer, layer, save)
        
        return


    """ aux functions """
    def __graph_path_2_polyline(self, graph, overwrite=False):
        if 'geometry' in list(graph) and not overwrite:
            return graph

        graph.loc[:, 'geometry'] = graph.path.apply(self.net.transform_node_seq_to_polyline)

        return graph


def cos_similarity(self, path_, v_cal=30):
    # TODO cos similarity for speed
    # path_ = [5434742616, 7346193109, 7346193114, 5434742611, 7346193115, 5434742612, 7346193183, 7346193182]
    seg = [[path_[i-1], path_[i]] for i in range(1, len(path_))]
    v_roads = pd.DataFrame(seg, columns=['src', 'dst']).merge(self.edges,  on=['src', 'dst']).v.values
    
    num = np.sum(v_roads.T * v_cal)
    denom = np.linalg.norm(v_roads) * np.linalg.norm([v_cal for x in v_roads])
    cos = num / denom  # 余弦值
    
    return cos

 
#%%
if __name__ == "__main__":
    net = build_geograph(ckpt='../cache/Shenzhen_graph.ckpt')
    matcher = ST_Matching(net=net)
    # github演示数据
    traj = matcher.load_points("../input/traj_0.geojson")
    path = matcher.matching(traj, plot=True, dir_trans=True, debug_in_levels=False)

    # 真实车辆移动轨迹
    traj = matcher.load_points("../input/traj_1.geojson")
    path = matcher.matching(traj, plot=True, dir_trans=True, debug_in_levels=False)

