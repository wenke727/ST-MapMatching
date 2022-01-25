#%%
import os
import sys
import time
import math
import copy
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from shapely import wkt
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, box

from douglasPeucker import dp_compress_for_points
from DigraphOSM import Digraph_OSM, load_net_helper
from utils.geo_plot_helper import map_visualize
from utils.azimuth_helper import cal_polyline_azimuth, azimuthAngle, azimuth_cos_similarity_for_linestring
from utils.geo_helper import coords_pair_dist, cal_foot_point_on_polyline, gdf_to_geojson, gdf_to_postgis, get_foot_point
from utils.log_helper import LogHelper, logbook

from setting import DIS_FACTOR, DEBUG_FOLDER, SZ_BBOX, FT_BBOX

warnings.filterwarnings('ignore')

#%%
"""" matching plot debug helper """
def linestring_combine_helper(path, net):
    """Create Linestring by coords id sequence.

    Args:
        path (list): The id sequence of Coordinations.
        net (Digraph_OSM): The Digraph_OSM object.

    Returns:
        Linestring: The linstring of the speicla sequence.
    """
    if path is None or len(path) <= 1:
        return None
    
    lst = gpd.GeoDataFrame([{'s': path[i], 'e': path[i+1]} for i in range(len(path)-1) ])
    lines = lst.merge(net.df_edges, on=['s', 'e']).geometry.values
    points = [ l.coords[:] for l in lines ]
    
    res = []
    for lst in points:
        res += lst
    
    return LineString(res)


def matching_debug_subplot(traj, item, net, ax=None, legend=True, scale=.9):
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
    i, j = item.rindex_0, item.rindex_1
    if isinstance(item.name, tuple):
        src, dest = item.name
    else:
        src, dest = item.e_0, item.s_1
        
    if ax is None:
        _, ax = map_visualize(traj, scale=scale, alpha=.6, color='white')
    else:
        map_visualize(traj, scale=scale, alpha=.6, color='white', ax=ax)

    # OD
    traj.loc[[item.pid_0]].plot(ax=ax, marker="*", label=f'O ({src})', zorder=9)
    traj.loc[[item.pid_1]].plot(ax=ax, marker="s", label=f'D ({dest})', zorder=9)

    # path
    net.df_edges.loc[[i]].plot(ax=ax, linestyle='--', alpha=.8, label=f'first({i})', color='green')
    gpd.GeoDataFrame( item ).T.plot(ax=ax, color='red', label='path')
    net.df_edges.loc[[j]].plot(ax=ax, linestyle='-.', alpha=.8, label=f'last({j}, {item.observ_prob:.2f})', color='black')

    # aux
    if 'f_dir' in item:
        info = f"{item.f:.3f} ({item.observ_prob:.3f}, {item.v:.3f}, {item.f_dir:.3f})"
    else:
        info = f"{item.f:.3f} ({item.observ_prob:.3f}, {item.v:.3f})"
    ax.set_title(
        f"{i} -> {j}: {info}", 
        color = 'black' if item.v < 0.7 else 'red' 
    )
    ax.set_axis_off()
    if legend: ax.legend()
    
    return ax
    

def matching_debug_level(traj, tList, graph_t, level, net, debug=False, debug_folder=DEBUG_FOLDER):
    """PLot the matchings between levels (i, i+1)

    Args:
        traj ([type]): [description]
        tList ([type]): The candidate points.
        graph_t ([type]): [description]
        level ([type]): [description]
        net ([type]): [description]
        debug (bool, optional): [description]. Save or not
        debug_folder ([type], optional): [description]. Defaults to DEBUG_FOLDER.

    Returns:
        [type]: [description]
    """

    n_rows = tList[level].shape[0]
    n_cols = tList[level+1].shape[0]

    graph_tmp = graph_t.query(f'pid_0 == {level}')

    plt.figure(figsize=(5*n_cols, 5*n_rows))
    for i in range(n_rows*n_cols):
        ax = plt.subplot(n_rows, n_cols, i + 1) 
        matching_debug_subplot(traj, graph_tmp.iloc[i], net=net, ax=ax)

    plt.suptitle(f'Level: {level} (observ, dis, dir)')
    plt.tight_layout()
    
    if debug:
        t = time.strftime("%Y%m%d_%H", time.localtime()) 
        plt.savefig( os.path.join(debug_folder, f"{t}_level_{level}.jpg"), dpi=300)
        # plt.close()
        
    return True


def matching_debug(traj, tList, graph_t, net, debug=True):
    """matching debug

    Args:
        traj ([type]): Trajectory
        tList ([type]): [description]
        graph_t ([type]): [description]
        net ([Digraph_OSM]): [description]
        debug (bool, optional): [description]. Defaults to True.
    """
    # create geometry
    graph_t.loc[:, 'geometry'] = graph_t.shortest_path.apply(lambda x: linestring_combine_helper(x['path'], net) if x is not None else None)
    graph_t = gpd.GeoDataFrame(graph_t)

    levels = len(tList) - 1
    for level in range(levels):
        matching_debug_level(traj, tList, graph_t, level, net=net, debug=debug)
    
    return


def cos_similarity(self, path_, v_cal=30):
    # TODO cos similarity for speed
    # path_ = [5434742616, 7346193109, 7346193114, 5434742611, 7346193115, 5434742612, 7346193183, 7346193182]
    seg = [[path_[i-1], path_[i]] for i in range(1, len(path_))]
    v_roads = pd.DataFrame(seg, columns=['s', 'e']).merge(self.edges,  on=['s', 'e']).v.values
    
    num = np.sum(v_roads.T * v_cal)
    denom = np.linalg.norm(v_roads) * np.linalg.norm([v_cal for x in v_roads])
    cos = num / denom  # 余弦值
    
    return cos


""" functions """
def load_trajectory(fn = '../input/tra.shp'):
    tra = gpd.read_file(fn, encoding='utf-8')
    tra.set_crs('EPSG:4326', inplace=True)
    if 'time' in tra.columns:
        tra.time = pd.to_datetime(tra['time'], format='%Y-%m-%d %H:%M:%S')
    # tra = coord_transfer( tra, in_sys = 'wgs', out_sys = 'gcj' )

    return tra


def get_candidates(traj, edges, georadius=50, top_k=5, dis_factor=DIS_FACTOR, plot=False, logger=None):
    """Get candidates edges for traj

    Args:
        traj (geodataframe): Trajectory T = p1 -> p2 -> ... -> pn
        edges (geodataframe): The graph edges. In this model, it is the same as `net.df_edges`.
        georadius (int, optional): The max radius for slect candicates. Defaults to 20.
        dis_factor (float, optional): Factor of convertor from lonlat to meter. Defaults to 1/110/1000.
        shrink(bool, optional): check the candidates is the same and delete the dulplicate level.
        top_k(int, optional): The maximun number of candicates.
        verbose (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    def _filter_candidate(df_candidates, top_k):
        df = copy.deepcopy(df_candidates)
        origin_size = df.shape[0]

        df_new = df.merge(edges, right_index=True, left_on='rindex')\
                        .sort_values(['pid', 'dist_to_line'], ascending=[True, True])\
                        .groupby(['pid', 'rid', 'dir'])\
                        .head(1)[['pid', 'rindex', 'rid', 's', 'e', 'dir' ,'dist_to_line']]
        df_new = df_new.groupby('pid').head(top_k).reset_index(drop=True)
        
        if logger is not None:
            logger.debug(f"Shrink candidate link {origin_size} -> {df_new.shape[0]} by choose the closest link in a road")

        return df_new

    edges = gpd.GeoDataFrame(edges)

    radius        = georadius*dis_factor
    boxes         = traj.geometry.apply(lambda i: box(i.x-radius, i.y-radius, i.x+radius, i.y+radius))
    df_candidates = boxes.apply(lambda x: edges.sindex.query(x, predicate='intersects')).explode().dropna()
    
    if df_candidates.shape[0] == 0:
        return None
    
    df_candidates = pd.DataFrame(df_candidates).reset_index().rename(columns={'index': 'pid', 'geometry':'rindex'})
    df_candidates = df_candidates.merge(traj['geometry'], left_on='pid', right_index=True)\
                                 .merge(edges['geometry'], left_on='rindex', right_index=True)\
                                 .rename(columns={'geometry_x': 'point_geom', 'geometry_y': 'edge_geom'})\
                                 .sort_index()
    df_candidates.loc[:, 'dist_to_line'] = df_candidates.apply(lambda x: x.point_geom.distance(x.edge_geom) / DIS_FACTOR, axis=1)

    candidates_filtered = _filter_candidate(df_candidates, top_k)
        
    if plot:
        ax = edges.loc[df_candidates.rindex.values].plot()
        edges.loc[candidates_filtered.rindex.values].plot(ax=ax, color='red')
        traj.plot(ax=ax)

    return candidates_filtered


def cal_observ_prob(df, std_deviation=20, standardization=False, logger=None):
    observ_prob_factor = 1 / (np.sqrt(2*np.pi) * std_deviation)
    def helper(x):
        return observ_prob_factor * np.exp(-np.power(x, 2)/(2*np.power(std_deviation, 2)))
    
    df.loc[:, 'observ_prob'] = df.dist_to_line.apply(helper)
    if standardization:
        df.loc[:, 'observ_prob'] = df.observ_prob / df.observ_prob.max()
    
    observ_prob_dict = df.set_index(['pid', 'rindex'])['observ_prob'].to_dict()
    
    if logger is not None:
        logger.info(f"candidates\n{df}")

    return observ_prob_dict


def cal_relative_offset(node:Point, polyline:LineString, with_connector=True, verbose=False):
    """Calculate the relative offset between the node's foot and the polyline.

    Args:
        node (Point): [description]
        polyline (LineString): [description]
        with_connector (bool, optional): Connect to the shortest path.
        verbose (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
        
    Example: 
        ```
        from shapely import wkt
        node = wkt.loads('POINT (113.934144 22.577979)')
        # case 0, 创科路/打石二路路口
        polyline = wkt.loads("LINESTRING (113.934186 22.57795, 113.934227 22.577982, 113.934274 22.578013, 113.934321 22.578035, 113.934373 22.578052, 113.934421 22.57806, 113.93448 22.578067)")
        # case 1
        linestring = wkt.loads('LINESTRING (113.93407 22.577737, 113.934079 22.577783, 113.934093 22.577824, 113.934116 22.577866, 113.934144 22.577905, 113.934186 22.57795, 113.934227 22.577982, 113.934274 22.578013, 113.934321 22.578035, 113.934373 22.578052, 113.934421 22.57806, 113.93448 22.578067)')
        # case 2
        polyline = wkt.loads("LINESTRING (113.93407 22.577737, 113.934079 22.577783, 113.934093 22.577824, 113.934116 22.577866, 113.934144 22.577905)")
        cal_relative_offset(node, linestring)
        ```
    """
    # The case continuous repeating point
    coords = [polyline.coords[0]]
    for p in polyline.coords[1:]:
        if p == coords[-1]:
            continue
        coords.append(p)
    
    lines = [LineString((coords[i], coords[i+1])) for i in range(len(coords)-1) if coords[i] != coords[i+1] ]
    lines = gpd.GeoDataFrame( {'geometry': lines} )

    # Don't use the predefined variables
    lines.loc[:, '_len'] = lines.geometry.apply(lambda x: coords_pair_dist(x.coords[0], x.coords[-1], xy=True))
    lines.loc[:, '_dist'] = lines.geometry.apply(lambda x: x.distance(node)/DIS_FACTOR)
    nearest_line_id = lines['_dist'].idxmin()
    
    if with_connector:
        foot = get_foot_point(node.coords[0], coords[nearest_line_id], coords[nearest_line_id+1])
        seg_0 = coords[:nearest_line_id+1] + [foot]
        seg_1 = [foot] + coords[nearest_line_id+1:]

    dist_prev_lines = np.sum(lines.loc[:nearest_line_id]._len.values[:-1])

    line = lines.loc[nearest_line_id].geometry
    foot_dict = cal_foot_point_on_polyline(node, line)
    _dist = coords_pair_dist(foot_dict['foot'], line.coords[0], xy=True)

    _dist = dist_prev_lines + _dist if foot_dict['flag'] > 0 else dist_prev_lines - _dist

    if with_connector:
        return _dist, seg_0, seg_1

    return _dist, None, None


def combine_link_and_path(item, net):
    # FIXME the case: src and dst is located at the same link. the path along the link
    # if item.e_0 == item.s_1:
        # print(f"({item.e_0}, {item.s_1}) are the same")
        # return [item.seg_last_0[0], item.seg_first_1[-1]]
    
    mid_lst = []
    if item.shortest_path is not None:
        if 'path' in item.shortest_path and item.shortest_path['path'] is not None:
            mid_lst = [ (net.node[i]['x'], net.node[i]['y']) for i in item.shortest_path['path']] 
    
    coords = [ item.seg_last_0, mid_lst, item.seg_first_1 ]

    path = [coords[0][0]]
    for seg in coords:
        for node in seg:
            if node == path[-1]:
                continue
            path.append(node)

    return path


def cal_trans_prob(df_candidates, traj, net, dir_trans_prob=True, logger=None):
    def _od_in_same_link(gt):
        # The case: `src` and `dst` all on the same link
        con = gt.rindex_0 == gt.rindex_1
        
        # case 1: The track goes in the same direction as the road
        case1 = con & (gt.offset_0 < gt.offset_1)
        if case1.sum()>0:
            gt.loc[:, 'forward'] = False
            gt.loc[case1, 'forward'] = True
            if logger is not None :
                logger.debug(f"Adjust the (src, dst) for the case the trajectory moving direction is the same with the road:\n\t {gt.loc[case1].to_dict(orient='record')}")
        
        # case 2: The track goes in the reverse direction of the road, and the process logic is the default one
        # case2 = con & (gt.offset_0 > gt.offset_1)
        
        # case 3: Continuous points of the trajectory have the same coordinates
        case3 = con & (gt.offset_0 == gt.offset_1)
        if case3.sum()>0:
            gt.loc[:, 'same_link_same_point'] = False
            gt.loc[case1, 'same_link_same_point'] = True
            if logger is not None:
                # FIXME Since the path is already compressed, this should not happen
                logger.warning(f"Continuous points of the trajectory have the same coordinates:\n\t {gt.loc[case3].to_dict(orient='record')}")
        
        gt.e_0 = gt.e_0.astype(np.int)
        gt.s_1 = gt.s_1.astype(np.int)
        
        return gt    

    df_candidates[['offset', 'seg_first', 'seg_last']] = df_candidates.apply(
        lambda x: 
            # cal_relative_offset(traj.loc[x.pid].geometry, net.df_edges.loc[x.rindex].geometry), # FIXME df_edges.edge是变形后的geometry，会对较低造成影响
            cal_relative_offset(traj.loc[x.pid].geometry, net.df_edges.loc[x.rindex].geom_origin), 
        axis=1, 
        result_type='expand'
    )

    tList, graph_t = [], []
    for _, sub in df_candidates.groupby('pid'):
        tList.append(sub)

    for i in range(len(tList)-1):
        base_atts = ['pid', 'rindex','s', 'e', 'offset', 'seg_first', 'seg_last']
        a, b = tList[i][base_atts], tList[i+1][base_atts+['observ_prob']]
        # Cartesian product
        a.loc[:, 'tmp'], b.loc[:, 'tmp'] = 1, 1
        graph_t.append(a.merge(b, on='tmp', suffixes=["_0", '_1']).drop(columns='tmp') )

    ordered_cols = ['pid_0', 'pid_1', 'rindex_0', 'rindex_1', 's_0', 'e_0', 's_1', 'e_1', 'seg_last_0', 'seg_first_1', 'offset_0', 'offset_1', 'observ_prob']
    graph_t = pd.concat(graph_t)[ordered_cols].reset_index(drop=True)
    graph_t = _od_in_same_link(graph_t)

    graph_t.loc[:, 'shortest_path'] = graph_t.apply(lambda x: net.a_star(x.e_0, x.s_1, max_layer=500, max_dist=10**4, plot=False), axis=1)
    graph_t.loc[:, 'd_sht']         = graph_t.shortest_path.apply(lambda x: x['cost'] if x is not None else np.inf )
    graph_t.loc[:, 'd_euc']         = graph_t.apply(lambda x: coords_pair_dist(traj.loc[x.pid_0].geometry, traj.loc[x.pid_1].geometry), axis=1)
    graph_t.loc[:, 'd_step0']       = graph_t.apply(lambda x: net.edge[x.s_0, x.e_0], axis=1)

    if 'forward' in graph_t.columns:
        graph_t.loc[graph_t.forward, 'd_sht'] = -graph_t.loc[graph_t.forward, 'd_step0']
        graph_t.loc[graph_t.forward, 'shortest_path' ] = None

    graph_t.loc[:, 'w'] = graph_t.d_sht + np.abs((graph_t.d_step0 - graph_t.offset_0)) + np.abs(graph_t.offset_1) 
    # FIXME: Why is the path between two points shorter than the distance. Here its weight is increased temporarily.
    graph_t.loc[:, 'v'] = graph_t.apply(lambda x:  x.d_euc/x.w if x.d_euc < x.w else x.w/x.d_euc*1.00, axis=1 )  
    
    # The case: o and d all on the same link
    if 'same_link_same_point' in graph_t.columns:
        graph_t.loc[graph_t.same_link_same_point, ['v', 'shortest_path']] = [1, None]

    graph_t.loc[:, 'path'] =  graph_t.apply(lambda x: combine_link_and_path(x, net), axis=1)
    if dir_trans_prob:
        graph_t.loc[:, 'move_dir'] = graph_t.apply(
            lambda x: 
                azimuthAngle(*traj.iloc[x.pid_0].geometry.coords[0], *traj.iloc[x.pid_1].geometry.coords[0]),
            axis=1
        )
        graph_t.loc[:, 'f_dir'] = graph_t.apply(lambda x: azimuth_cos_similarity_for_linestring(x.path, x.move_dir, True), axis=1)

    # Manually change the `f_dir` weights of the starting and ending on the same line segment
    graph_t.loc[graph_t.rindex_0 == graph_t.rindex_1, 'f_dir'] = 1
    
    graph_t.loc[:, 'f'] = graph_t.observ_prob * graph_t.v * (graph_t.f_dir if dir_trans_prob else 1)
    
    atts = ['pid_0', 'rindex_0', 'rindex_1']
    gt = graph_t.drop_duplicates(atts).set_index(atts)

    if logger is not None:
        atts = ['pid_1', 'forward', 'observ_prob', 'd_sht', 'd_step0', 'offset_0', 'offset_1', 'w', 'd_euc', 'v', 'move_dir', 'f_dir', 'f'] # 'path'
        if 'forward' not in graph_t.columns:
            atts.remove('forward')
            
        levels = gt.index.get_level_values(0).unique().sort_values()
        info = "\n".join( gt[atts].loc[x].__str__() for x in levels )
        logger.info(f"gt\n{info}")

    return tList, gt, graph_t


def find_matched_sequence(gt, df_candidates, tList, logger=None):
    prev_dict, f_score = {}, {}

    for i, item in tList[0].iterrows():
        f_score[i] = item.observ_prob

    ids = sorted(df_candidates.pid.unique())
    for i in range(1, len(tList)):
        for j, nxt in tList[i].iterrows():
            _max = -np.inf
            for k, cur in tList[i-1].iterrows():
                _f = gt.loc[ids[i-1]].loc[cur.rindex].loc[nxt.rindex].f
                if _f > 1.001 and logger is not None:
                    logger.warning(f"level {i-1}->{i}({cur.rindex}, {nxt.rindex}), F value {_f:.3f}, exceed 1.\n\t{gt.loc[i-1].loc[cur.rindex].loc[nxt.rindex].to_dict()}")
                    
                alt = f_score[k] + _f
                if alt > _max:
                    _max = alt
                    prev_dict[j] = k
                f_score[j] = _max
    
    rList = []
    c = max(f_score, key= lambda x: (f_score.get(x), x))
    
    for i in range(len(tList)-1, 0, -1):
        rList.append(c)
        c = prev_dict[c]
    rList.append(c)
    rList = df_candidates.loc[rList[::-1]][['pid', 'rindex', 's', 'e']]

    if logger is not None:
        logger.info(f'max score: {c}, f_score: {f_score}\n{rList}')
    
    return rList


def get_path(rList, gt, net):
    """Get path by matched sequence node.

    Args:
        rList ([type]): [description]
        graph_t ([type]): [description]
        net ([type]): [description]

    Returns:
        [type]: [description]
    """
    if rList.shape[0] == 1:
        return net.df_edges.merge(rList, on=['s', 'e'])
    
    def _helper(x):
        res = gt.loc[x.pid].loc[x.rindex].loc[x.nxt_rindex].shortest_path
        
        return res if res is None else res['path']
    
    rList.loc[:, 'nxt_rindex'] = rList.rindex.shift(-1).fillna(0).astype(np.int)
    rList.loc[:, 'pre_rindex'] = rList.rindex.shift(1).fillna(0).astype(np.int)
    steps = rList[:-1].apply(lambda x: _helper(x), axis=1)

    coords = []
    for step in steps.values:
        if step is None:
            continue
        coords += step
    path = net.node_sequence_to_edge(coords, attrs=list(net.df_edges)) if len(coords) > 1 else None
    
    first_step = net.df_edges.loc[[rList.iloc[0].rindex]]
    last_step  = net.df_edges.loc[[rList.iloc[-1].rindex]]
    
    # add interval of the edge portion
    first =  gt.loc[rList.iloc[0].pid].loc[rList.iloc[0].rindex].loc[rList.iloc[0].nxt_rindex]
    first_step.loc[:, 'breakpoint'] = first.offset_0 / first.d_step0
    
    last  =  gt.loc[rList.iloc[-1].pid-1].loc[rList.iloc[-1].pre_rindex].loc[rList.iloc[-1].rindex]
    last_step.loc[:, 'breakpoint'] = last.offset_1 / net.edge[last.s_1, last.e_1]
    
        
    if path is not None: 
        path.loc[:, 'step'] = 1

    first_step.loc[:, 'step'] = 0
    last_step.loc[:, 'step']  = -1
    
    return gpd.GeoDataFrame( pd.concat([first_step, path, last_step]) ).reset_index(drop=True)


def st_matching(traj, 
                net, 
                name = None,
                std_deviation=20, 
                georadius=50, 
                top_k=5, 
                traj_compress=True,
                traj_thres=5,
                plot=True, 
                satellite=False,
                save_fn=None, 
                debug_in_levels=False, 
                plot_candidate=False,
                dir_trans_prob=True, 
                observ_prob_standardization=True,
                logger=None,
                symbology_columns=None,
                categorical=True
                ):
    """[summary]

    Args:
        traj ([type]): [description]
        net ([type]): [description]
        std_deviation (int, optional): [description]. Defaults to 20.
        georadius (int, optional): [description]. Defaults to 50.
        top_k (int, optional): [description]. Defaults to 5.
        plot (bool, optional): [description]. Defaults to True.
        satellite (bool, optional): [description]. Defaults to False.
        save_fn ([type], optional): [description]. Defaults to None.
        debug_in_levels (bool, optional): [description]. Defaults to False.
        plot_candidate (bool, optional): [description]. Defaults to False.
        dir_trans_prob (bool, optional): The transmission probalility analysis take the azimuth angle into account. Defaults to True.

    Returns:
        [type]: [description]
    """
    if traj.shape[0] == 0:
        return None

    origin_traj = traj.copy()
    if traj_compress:
        traj = dp_compress_for_points(traj, dis_thred=traj_thres)
        if logger is not None:
            # logger.info(f"compressed traj: \n{traj}")
            logger.info(f"Trajectory compression rate: {traj.shape[0]/origin_traj.shape[0]*100:.1f}% ({origin_traj.shape[0]} -> {traj.shape[0]})")

    # step 1: candidate prepararation
    df_candidates = get_candidates(traj, net.df_edges, georadius=georadius, top_k=top_k, plot=plot_candidate, logger=logger)
    if df_candidates is None:
        logger.info(f"Trajectory {name} has no matching candidates")
        return None
    
    if df_candidates.pid.nunique() <= 1:
        logger.info(f'Trajectory {name} one level has candidates.')
        return None
    
    # step 2.1: Spatial analysis, obervation prob
    cal_observ_prob(df_candidates, std_deviation, standardization=observ_prob_standardization, logger=logger)
    
    # step 2.2: Spatial analysis, transmission prob
    tList, gt, graph_t = cal_trans_prob(df_candidates, traj, net, dir_trans_prob=dir_trans_prob, logger=logger)

    # TODO step 3: temporal analysis
    
    # step 4: find matched sequence
    rList = find_matched_sequence(gt, df_candidates, tList, logger=logger)
    path = get_path(rList, gt, net)

    if debug_in_levels:
        matching_debug(traj, tList, graph_t, net, debug_in_levels)

    def _matching_plot():
        def _base_plot():
            if symbology_columns is not None and symbology_columns in origin_traj.columns:
                ax = origin_traj.plot(alpha=.3, column=symbology_columns, categorical=categorical, legend=True)
            else:
                ax = origin_traj.plot(alpha=.3, color='black')
            ax.axis('off')
            
            return ax
        
        # plot，trajectory point
        if satellite:
            try:
                _, ax = map_visualize(origin_traj, alpha=.3, scale=.2, color='black')
                if symbology_columns is not None:
                    origin_traj.plot(alpha=.3, column=symbology_columns, categorical=categorical, legend=True, ax=ax)
            except:
                ax = _base_plot()       
        else:
            ax = _base_plot()
            
        traj.plot(ax=ax, color='blue', alpha=.5, label= 'Compresed')
        traj.head(1).plot(ax=ax, marker = '*', color='red', zorder=9, label= 'Start point')
        # network
        edge_lst = net.df_edges.sindex.query(box(*traj.total_bounds), predicate='intersects')
        net.df_edges.loc[edge_lst].plot(ax=ax, color='black', linewidth=.8, alpha=.4, label='Network' )
        # candidate
        net.df_edges.loc[df_candidates.rindex.values].plot(
            ax=ax, label='Candidates', color='blue', linestyle='--', linewidth=.8, alpha=.5)
        # path
        if path is not None:
            path.plot(ax=ax, label='Path', color='red', alpha=.5)
        
        # FIXME
        if symbology_columns is None:
            plt.legend(loc='best')
        
        if save_fn is not None:
            plt.tight_layout()
            plt.savefig(f'{save_fn}.jpg' if '.jpg' not in save_fn else save_fn, dpi=300)
            plt.close()
        
        return ax

    if plot:
        _matching_plot()

    return {'path':path, 'rList': rList}


def check(fn, logger):
    traj = load_trajectory(fn).reset_index()
    traj = dp_compress_for_points(traj, dis_thred=5)
    
    # step 1: candidate prepararation
    df_candidates = get_candidates(traj, net.df_edges, georadius=50, plot=True, top_k=3)
    # step 2.1: Spatial analysis, obervation prob
    cal_observ_prob(df_candidates, std_deviation=20, standardization=True)

    # step 2.2: Spatial analysis, transmission prob
    tList, gt, graph_t = cal_trans_prob(df_candidates, traj, net, dir_trans_prob=True, logger=logger)

    # step 4: find matched sequence
    rList = find_matched_sequence(gt, df_candidates, tList)
    path = get_path(rList, gt, net)
    
    path.plot()
    rList
    matching_debug(traj, tList, graph_t, net, debug=True)


    """ check the azimuth """
    # x = gt.loc[0].loc[23136].loc[23137]
    # cal_polyline_azimuth(x.path)
    # azimuth_cos_similarity(cal_polyline_azimuth(x.path), x.move_dir )

    # df_candidates_ = get_candidates(traj, net.df_edges, georadius=50, plot=True, top_k=5, shrink=False)
    # cal_observ_prob(df_candidates_, std_deviation=20, standardization=True)
    # tList_, gt_, graph_t_ = cal_trans_prob(df_candidates, traj, net, dir_trans_prob=True)

    # df_candidates_.to_csv('df_candidates_.csv')
    # df_candidates.to_csv('df_candidates.csv')

   
    return  rList


#%%
if __name__ == '__main__':
    """ initiale """
    logger = LogHelper(log_name='matching.log', stdOutFlag=True).make_logger(level=logbook.INFO)
    net = load_net_helper(bbox=SZ_BBOX, combine_link=True, convert_to_geojson=True)

    # 福田测试案例(手动绘制)
    traj = load_trajectory("../input/traj_0.geojson")
    path = st_matching(traj, net, plot=True, satellite=True, logger=logger)

    traj = load_trajectory("../input/traj_1.geojson")
    path = st_matching(traj, net, plot=True, logger=logger, satellite=True)


    # BUG 节点1：没有匹配的记录
    fn = '../input/traj_debug_199.geojson'
    traj = load_trajectory(fn)
    path = st_matching(traj, net, plot=True, debug_in_levels=False, logger=logger, satellite=True, top_k=5)
    
    # BUG 节点1：没有匹配的记录
    fn = '../input/traj_debug_200.geojson'
    traj = load_trajectory(fn)
    path = st_matching(traj, net, plot=True, debug_in_levels=False, logger=logger, satellite=True, top_k=5)
    
        
    """ matching test 0 """
    fn = "../input/traj_debug_case2.geojson"
    traj = load_trajectory(fn)
    path = st_matching(traj, net, plot=True, debug_in_levels=False, logger=logger, satellite=True, top_k=5)


    """ matching test 1 """
    id = 1
    fn = f"../input/traj_debug_{id}.geojson"
    traj = load_trajectory(fn).reset_index().drop(columns='level_0')
    path = st_matching(traj, net, plot=True, top_k=5, debug_in_levels=False, logger=logger)
    # check(fn)

    # 真实pano数据
    fn = '../input/traj_debug_case1.geojson'
    traj = load_trajectory(fn)
    path = st_matching(traj, net, plot=True, top_k=5, debug_in_levels=False, logger=logger)

