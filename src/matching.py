#%%
import os
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, box

from DigraphOSM import DigraphOSM
from utils.logger_helper import make_logger
from utils.timer import Timer

from geo.geo_plot_helper import map_visualize
from geo.coord.coordTransfrom_shp import coord_transfer
from geo.douglasPeucker import dp_compress_for_points as dp_compress
from geo.geo_helper import coords_pair_dist, point_to_polyline_process, geom_series_distance
from geo.azimuth_helper import azimuth_cos_similarity_for_linestring, azimuthAngle

from setting import DEBUG_FOLDER, DIS_FACTOR


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
                 net:DigraphOSM, 
                 dp_thres=5, 
                 route_planning_max_query_size=2000, 
                 route_planning_max_dist=10000,
                 top_k_candidates=5,
                 cand_search_radius=50,
                 crs_wgs=4326, 
                 crs_prj=900913, 
                 ):
        self.crs_wgs        = crs_wgs
        self.crs_prj        = crs_prj
        self.logger         = make_logger('../log', console=True)
        self.traj_processor = Trajectory(dp_thres, crs_wgs, crs_prj, self.logger)
        self.net            = net
        self.dis_factor     = DIS_FACTOR
        self.debug_folder   = DEBUG_FOLDER
        if not os.path.exists(self.debug_folder):
            os.makedirs(self.debug_folder)

        # hyper parameters
        self.top_k_candidates = top_k_candidates
        self.cand_search_radius = cand_search_radius
        self.route_planning_max_query_size = route_planning_max_query_size
        self.route_planning_max_dist = route_planning_max_dist


    def load_points(self, fn, compress=True, dp_thres: int = None, crs: int = None, in_sys: str = 'wgs', out_sys: str = 'wgs'):
        return self.traj_processor.load_points(fn, compress, dp_thres, crs, in_sys, out_sys)


    def matching(self, traj, top_k=None, dir_trans=False, plot=True, plot_scale=.2, debug_in_levels=False):
        cands = self.get_candidates(traj, self.cand_search_radius, top_k=top_k, plot=False)
        
        # No matched
        if cands is None:
            return None
        
        # Only one single point matched
        if traj.shape[0] == 1 or cands.pid.nunique() == 1: 
            eid = cands.sort_values('dist_p2c').head(1).eid.values
            route = self.net.get_edge(eid)
            return route

        cands, graph = self.spatial_anslysis(traj, cands, dir_trans=dir_trans)
        rList        = self.find_matched_sequence(cands, graph)
        route, conns  = self.get_path(traj, rList, graph, cands)

        if plot:
            self._plot_matching(traj, cands, route, plot_scale=plot_scale, satellite=False)

        if debug_in_levels:
            self.matching_debug(traj, graph, True)
        
        return route

    
    def get_candidates(self, traj:gpd.GeoDataFrame, radius:int=None, plot:bool=False, top_k:int=None):
        """Get candidates points and its localed edge for traj, which are line segment projection of p_i to these road segs.
        This step can be efficiently perfermed with the build-in grid-based spatial index.

        Args:
            traj (gpd.GeoDataFrame): _description_
            radius (int, optional): _description_. Defaults to 25.
            plot (bool, optional): _description_. Defaults to True.
            top_k (int, optional): _description_. Defaults to None.

        Returns:
            (gpd.GeoDataFrame): a set of candidate points and the candidate road segments they lie on.
        Example:
            traj:
                            id	geometry
                0	None	POINT (114.04219 22.53083)
                1	None	POINT (114.04809 22.53141)
        """
        def _filter_candidate(df_candidates, top_k):
            df = df_candidates.copy()
            origin_size = df.shape[0]

            # group by ['pid', 'rid', 'dir'] to filter the edges which belong to the same road.
            df_new = df.merge(self.net.df_edges, right_index=True, left_on='eid')\
                       .sort_values(['pid', 'dist_p2c'], ascending=[True, True])\
                       .groupby(['pid', 'rid', 'dir'])\
                       .head(1)
            df_new = df_new.groupby('pid').head(top_k).reset_index(drop=True)
            self.logger.info(f"Top k candidate link, size: {origin_size} -> {df_new.shape[0]}")

            return df_new

        top_k   = self.top_k_candidates if top_k is None else top_k
        radius  = (self.cand_search_radius if radius is None else radius) * self.dis_factor
        boxes   = traj.geometry.apply(lambda i: box(i.x - radius, i.y - radius, i.x + radius, i.y + radius))
        df_cand = boxes.apply(lambda x: self.net.spatial_query(x)).explode().dropna()
        if df_cand.shape[0] == 0:
            return None
        
        df_cand = pd.DataFrame(df_cand).reset_index()\
                                       .rename(columns={'index': 'pid', 'geometry':'eid'})\
                                       .merge(traj['geometry'], left_on='pid', right_index=True)\
                                       .merge(self.net.df_edges['geometry'], left_on='eid', right_index=True)\
                                       .rename(columns={'geometry_x': 'point_geom', 'geometry_y': 'edge_geom'})\
                                       .sort_index()
        # df_cand.loc[:, 'dist_p2c'] = df_cand.apply(lambda x: x.point_geom.distance(x.edge_geom) / DIS_FACTOR, axis=1)
        df_cand.loc[:, 'dist_p2c'] = geom_series_distance(df_cand.point_geom, df_cand.edge_geom, self.crs_wgs, self.crs_prj)
        cands_ = _filter_candidate(df_cand, top_k)
        
        if cands_ is None:
            self.logger.warning(f"Trajectory has no matching candidates")
            return None
        
        if plot:
            ax = self.net.get_edge(df_cand.eid.values).plot()
            self.net.get_edge(cands_.eid.values).plot(ax=ax, color='red')
            traj.plot(ax=ax)
        
        keep_cols = ['pid', 'eid', 'rid', 's', 'e', 'dir' ,'dist_p2c', 'observ_prob']
        keep_cols = [ i for i in keep_cols if i in cands_ ]
        
        return cands_[keep_cols]


    def spatial_anslysis(self, traj, cands, dir_trans=False):
        """Geometric and topological info, the product of `observation prob` and the `transmission prob`
        
        Special Case:
            a. same_link_same_point
        """
        cands = self._line_segment_projection(traj, cands)
        _     = self._observ_prob(cands)
        graph = self._construct_graph(traj, cands)        
        graph = self._trans_prob(graph)
        if dir_trans:
            graph = self._move_dir_similarity(graph, traj)
        # spatial analysis: observ_prob * trans_prob
        graph.loc[:, 'f'] = graph.observ_prob * graph.v * (graph.f_dir if dir_trans else 1)

        atts = ['pid_0', 'eid_0', 'eid_1']
        graph = graph.drop_duplicates(atts).set_index(atts).sort_index()

        return cands, graph
    
    
    def temporal_anylysis(self,):
        # take the similar speed conditions into account
        # the cosine distance is used to measure the similarity between the actual average speed from `ci-1` to `ci`
        # and the speed constraints of the path
        
        return NotImplementedError
    
    
    def _construct_graph(self, traj, cands):
        """Construct the candiadte graph (level, src, dst) for spatial and temporal analysis.
        """
        graph = []
        tList = [layer for _, layer in cands.groupby('pid')]

        base_atts = ['pid', 'eid','s', 'e', 'len_0', 'len_1', 'seg_0', 'seg_1']
        cols_filter = [
            'pid_0',
            'pid_1',
            'eid_0',
            'eid_1',
            'e_0',
            's_1',
            'seg_0_1',
            'seg_1_0',
            'observ_prob',
            'len_0_1',
            'len_1_0',
        ]
        rename_dict = {
            'seg_0_1': 'step_first',
            'seg_1_0': 'step_last',
            'len_0_1': 'offset_0',
            'len_1_0': 'offset_1',
            'cost': 'd_sht',
        }
        
        # Cartesian product
        for i in range(len(tList)-1):
            a, b = tList[i][base_atts], tList[i+1][base_atts+['observ_prob']]
            a.loc[:, 'tmp'], b.loc[:, 'tmp'] = 1, 1 
            graph.append(a.merge(b, on='tmp', suffixes=["_0", '_1']).drop(columns='tmp') )
        graph = pd.concat(graph).reset_index(drop=True)
        
        graph = graph[[i for i in cols_filter if i in graph.columns]]
        graph.rename(columns=rename_dict, inplace=True)
        graph.loc[:, 'd_euc'] = graph.apply(
            lambda x: coords_pair_dist(traj.loc[x.pid_0].geometry, traj.loc[x.pid_1].geometry), axis=1)

        return graph


    def _observ_prob(self, df, bias=0, deviation=20, normal=True):
        """The obervation prob is defined as the likelihood that a GPS sampling point `p_i` mathes a candidate point `C_ij`
        computed based on the distance between the two points. 

        Args:
            df (gpd.GeoDataFrame): GPS points dataframe.
            bias (float, optional): GPS measurement error bias. Defaults to 0.
            deviation (float, optional): GPS measurement error deviation. Defaults to 20.
            normal (bool, optional): Min-Max Scaling. Defaults to False.

        Returns:
            _type_: _description_
        """

        observ_prob_factor = 1 / (np.sqrt( 2 * np.pi) * deviation)
        cal_helper = lambda x: observ_prob_factor * np.exp(-np.power(x - bias, 2)/(2 * np.power(deviation, 2)))
        df.loc[:, 'observ_prob'] = df.dist_p2c.apply(cal_helper)
        if normal:
            df.loc[:, 'observ_prob'] = df.observ_prob / df.observ_prob.max()
        
        observ_prob_dict = df.set_index(['pid', 'eid'])['observ_prob'].to_dict()
        self.logger.info(f"candidates:\n{df}")
        
        return observ_prob_dict
    
    
    def _trans_prob(self, graph):
        same_link_mask = graph.eid_0 == graph.eid_1
        ods = graph[~same_link_mask][['e_0', 's_1']].drop_duplicates().values
        if len(ods) > 0:
            df_planning = pd.DataFrame([ {'e_0':o, 
                                          's_1':d, 
                                          **self.net.a_star(
                                              o, 
                                              d, 
                                              max_query_size=self.route_planning_max_query_size,
                                              max_dist=self.route_planning_max_dist) 
                                          } for o, d in ods ]
                                       )

            graph = graph.merge(df_planning, on=['e_0', 's_1'], how='left')
            # `w` is the shortest path from `ci-1` to `ci`
            graph.loc[:, 'w'] = graph.cost + graph.offset_0 + graph.offset_1 
            # transmission probability
            graph.loc[:, 'v'] = graph.apply(lambda x: x.d_euc / x.w if x.d_euc < x.w else x.w / x.d_euc * 1.00, axis=1 )

        graph.loc[same_link_mask, 'path'] = graph.loc[same_link_mask, 'eid_0'].apply(
            lambda x: self.net.get_edge(x, att=['s', 'e']).values.tolist())
        graph.loc[same_link_mask, 'v'] = 1
        
        return graph


    def _move_dir_similarity(self, graph, traj):
        self.__graph_path_2_polyline(graph)
        
        graph.loc[:, 'move_dir'] = graph.apply(
            lambda x: 
                azimuthAngle(*traj.iloc[x.pid_0].geometry.coords[0], 
                             *traj.iloc[x.pid_1].geometry.coords[0]),
            axis=1
        )
        
        graph.loc[:, 'f_dir'] = graph.apply(
            lambda x: 
                (azimuth_cos_similarity_for_linestring(x.geometry, x.move_dir, weight=True) + 1) / 2
                    if x.geometry is not None else 1, 
            axis=1
        )

        # FIXME Manually change the `f_dir` weights of the starting and ending on the same line segment
        # same_link_mask = graph.eid_0 == graph.eid_1
        # graph.loc[same_link_mask, 'f_dir'] = 1
        
        return graph
        

    def _line_segment_projection(self, traj_points, cands, keep_cols=['len_0', 'len_1', 'seg_0', 'seg_1']):
        # TODO: Reduce unnecessary caculations
        # `len` was an required attribute in graph, while `seg` only use in the first/final step
        cands[keep_cols] = cands.apply(
            lambda x: 
                point_to_polyline_process(
                    traj_points.loc[x.pid].geometry, 
                    self.net.get_edge(x.eid, 'geom_origin'), 
                    coord_sys=True
                ), 
            axis=1, 
            result_type='expand'
        )[keep_cols]
        
        return cands
        

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
        rList = cands.loc[rList[::-1]][['pid', 'eid', 's', 'e']]

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
            return self.net.merge_edge(rList, on=['s', 'e']), None
        
        def _helper(x):
            res = graph.loc[x.pid].loc[x.eid].loc[x.nxt_eid].path
            return res if res is None else res
        
        rList.loc[:, 'nxt_eid'] = rList.eid.shift(-1).fillna(0).astype(np.int)
        steps = rList[:-1].apply(lambda x: _helper(x), axis=1)
        # Drop `Consecutive identical line segments`, and keep the first one record
        steps = steps[steps != steps.shift(1)]

        od_lst = [rList.iloc[0].s]
        for step in steps.values:
            if step is None or isinstance(step, np.float):
                continue
            
            if step[0] == od_lst[-1]:
                od_lst += step[1:]
            else:    
                od_lst += step
        od_lst += [rList.iloc[-1].e]
        path = self.net.node_seq_to_df_edge(od_lst)
        
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

        graph.loc[:, 'geometry'] = graph.path.apply(self.net.node_seq_to_polyline)

        return graph


def cos_similarity(self, path_, v_cal=30):
    # TODO cos similarity for speed
    # path_ = [5434742616, 7346193109, 7346193114, 5434742611, 7346193115, 5434742612, 7346193183, 7346193182]
    seg = [[path_[i-1], path_[i]] for i in range(1, len(path_))]
    v_roads = pd.DataFrame(seg, columns=['s', 'e']).merge(self.edges,  on=['s', 'e']).v.values
    
    num = np.sum(v_roads.T * v_cal)
    denom = np.linalg.norm(v_roads) * np.linalg.norm([v_cal for x in v_roads])
    cos = num / denom  # 余弦值
    
    return cos


#%%
if __name__ == "__main__":
    NET = DigraphOSM("Shenzhen", resume='../input/Shenzhen.pkl')
    # path = net.route_planning(o=7959990710, d=499265789, plot=True)

    self = ST_Matching(net=NET)

    # github演示数据
    traj = self.load_points("../input/traj_0.geojson")
    path = self.matching(traj, plot=True, dir_trans=True, debug_in_levels=False)
    
    # 真实车辆移动轨迹
    traj = self.load_points("../input/traj_1.geojson")
    path = self.matching(traj, plot=True, dir_trans=True, debug_in_levels=False)
    
    # 测试：压缩算法
    traj = self.load_points("../input/test/traj_debug_199.geojson")
    path = self.matching(traj, plot=True)

    # 测试：起点和终点同个路段
    traj = self.load_points("../input/test/traj_debug_200.geojson")
    path = self.matching(traj, plot=True)

    # 测试，仅两个点，中兴公寓 
    traj = self.load_points("../input/test/traj_debug_7.geojson")
    path = self.matching(traj, plot=True, top_k=3, dir_trans=False, plot_scale=5)

    # 测试, 打石一路车道右转专用道拐弯
    traj = self.load_points("../input/test/traj_debug_20.geojson")
    path = self.matching(traj, plot=True, top_k=3, dir_trans=True, plot_scale=5)

    # 测试, 打石一路反向车道测试
    traj = self.load_points("../input/test/traj_debug_141.geojson")
    path = self.matching(traj, plot=True, top_k=5, dir_trans=True, plot_scale=5)  
  
    # 测试, 深南大道
    traj = self.load_points("../input/test/traj_debug_case1.geojson")
    path = self.matching(traj, plot=True, top_k=5, dir_trans=True, plot_scale=.1)
    
    # 测试, 小支路测试
    traj = self.load_points("../input/test/traj_debug_case2.geojson")
    path = self.matching(traj, plot=True, top_k=5, dir_trans=True, plot_scale=.1)

    # 测试, 深南大道市民中心段测试
    traj = self.load_points("../input/test/traj_debug_rid.geojson")
    path = self.matching(traj, plot=True, top_k=5, dir_trans=True, plot_scale=.1)

    # 测试，打石一路
    traj = self.load_points("../input/test/traj_debug_dashiyilu_0.geojson")
    path = self.matching(traj, plot=True, top_k=3, dir_trans=True, plot_scale=.01)

    # 测试，本身畸形的数据
    traj = self.load_points("../input/test/traj_debug.geojson")
    path = self.matching(traj, plot=True, top_k=5, dir_trans=True, plot_scale=.1)


