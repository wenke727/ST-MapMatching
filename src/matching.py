#%%
import os
import sys
import time
import copy
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, box

from utils.classes import Digraph
from utils.geo_plot_helper import map_visualize
from utils.interval_helper import merge_intervals
from utils.pickle_helper import PickleSaver
from load_step import df_path, split_line_to_points
from DigraphOSM import Digraph_OSM, load_net_helper
from utils.geo_helper import coords_pair_dist, cal_foot_point_on_polyline, gdf_to_geojson, gdf_to_postgis
from utils.log_helper import LogHelper, logbook

from setting import filters as way_filters
from setting import DIS_FACTOR, DEBUG_FOLDER, SZ_BBOX

g_log_helper = LogHelper(log_name='log.log', stdOutFlag=True)
logger       = g_log_helper.make_logger(level=logbook.INFO)
warnings.filterwarnings('ignore')

net = load_net_helper(bbox=SZ_BBOX, combine_link=True, convert_to_geojson=True)

#%%

def draw_observ_prob_distribution():

    def data_prepare(std_deviation = 20):
        observ_prob_factor = 1 / (np.sqrt(2*np.pi) * std_deviation)

        def helper(x):
            return observ_prob_factor * np.exp(-np.power(x, 2)/(2*np.power(std_deviation, 2)))

        df = pd.DataFrame({'x': np.arange(0, 100, .1)})
        df.loc[:, 'y'] = df.x.apply( helper)
        df.loc[:, '_std'] = ""+ str(std_deviation)
        
        return df


    df = pd.concat( [ data_prepare(i) for i in range(5, 30, 5) ] )

    ax = sns.lineplot(x=df.x, y= df.y, hue=df._std)


"""" matching plot debug helper """
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
    net.df_edges.loc[[i]].plot(ax=ax, linestyle='--', alpha=.8, label=f'first({i}, {item.observ_prob:.2f})', color='green')
    gpd.GeoDataFrame( item ).T.plot(ax=ax, color='red', label='path')
    net.df_edges.loc[[j]].plot(ax=ax, linestyle='-.', alpha=.8, label=f'last({j}, {item.observ_prob:.2f})', color='black')

    # aux
    ax.set_title( f"{i} -> {j}, V: {item.v:.3f}, f: {item.f:.3f}", color = 'black' if item.v < 0.7 else 'red' )
    ax.set_axis_off()
    if legend: ax.legend()
    
    return ax
    

def matching_debug_level(traj, tList, graph_t, level, net, debug=False, debug_folder=DEBUG_FOLDER):
    """PLot the matchings between levels (i, i+1)

    Args:
        tList ([type]): The candidate points.
        graph_t ([type]): [description]
        level ([type]): [description]

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

    plt.suptitle(f'Level: {level}')
    plt.tight_layout()
    
    if debug:
        t = time.strftime("%Y%m%d_%H", time.localtime()) 
        plt.savefig( os.path.join(debug_folder, f"{t}_level_{level}.jpg"), dpi=300)
        # plt.close()
        
    return True


def matching_debug(traj, tList, graph_t, net, debug=True):
    # create geometry
    graph_t.loc[:, 'geometry'] = graph_t.shortest_path.apply(lambda x: linestring_combine_helper(x['path'], net) if x is not None else None)
    graph_t = gpd.GeoDataFrame(graph_t)

    levels = len(tList) - 1
    for level in range(levels):
        matching_debug_level(traj, tList, graph_t, level, net=net, debug=debug)
    
    return


def cos_similarity(self, path_, v_cal=30):
    # TODO cos_similarity
    # path_ = [5434742616, 7346193109, 7346193114, 5434742611, 7346193115, 5434742612, 7346193183, 7346193182]
    seg = [[path_[i-1], path_[i]] for i in range(1, len(path_))]
    v_roads = pd.DataFrame(seg, columns=['s', 'e']).merge(self.edges,  on=['s', 'e']).v.values
    
    num = np.sum(v_roads.T * v_cal)
    denom = np.linalg.norm(v_roads) * np.linalg.norm([v_cal for x in v_roads])
    cos = num / denom  # 余弦值
    
    return cos


"""" For test """
def code_testing_single(id, net, debuf_with_step=False):
    traj = split_line_to_points(df_path.iloc[id].geometry, compress=True, config={'dist_max': 8, 'verbose': True})

    if not debuf_with_step:
        st_matching(traj, net, plot=True, save_fn=id)
    else:
        # step 1: candidate prepararation
        df_candidates = get_candidates(traj, net.df_edges, georadius=50, plot=True, verbose=False)

        # step 2.1: Spatial analysis, obervation prob
        observ_prob_dict = cal_observ_prob(df_candidates)

        # step 2.2: Spatial analysis, transmission prob
        tList, graph_t = cal_trans_prob(df_candidates, net)


        # step 4: find matched sequence
        rList = find_matched_sequence(graph_t, df_candidates, tList)
        path = get_path(rList, graph_t, net, True)

        graph_t

    return 


def code_testing(start=0, end=100, debuf_with_step=False):
    for id in tqdm( range(start, end) ):
        traj = split_line_to_points(df_path.iloc[id].geometry, compress=True, config={'dist_max': 8, 'verbose': True})
        _ = st_matching(traj, net, plot=True, save_fn=f"{id:03d}", satellite=True)


""" functions """
def load_trajectory(fn = '../input/tra.shp'):
    tra = gpd.read_file(fn, encoding='utf-8')
    tra.set_crs('EPSG:4326', inplace=True)
    if 'time' in tra.columns:
        tra.time = pd.to_datetime(tra['time'], format='%Y-%m-%d %H:%M:%S')
    # tra = coord_transfer( tra, in_sys = 'wgs', out_sys = 'gcj' )

    return tra


def get_candidates(traj, edges, georadius=20, top_k=5, dis_factor=DIS_FACTOR, plot=False, logger=logger):
    """Get candidates edges for traj

    Args:
        traj (geodataframe): Trajectory T = p1 -> p2 -> ... -> pn
        edges (geodataframe): The graph edges. In this model, it is the same as `net.df_edges`.
        georadius (int, optional): The max radius for slect candicates. Defaults to 20.
        dis_factor (float, optional): Factor of convertor from lonlat to meter. Defaults to 1/110/1000.
        top_k(int, optional): The maximun number of candicates.
        verbose (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    def _filter_candidate(df_candidates, top_k, verbose=True):
        df = copy.deepcopy(df_candidates)
        origin_size = df.shape[0]

        # add direction
        df_new = df.merge(edges, right_index=True, left_on='rindex')\
                        .sort_values(['pid', 'dist_to_line'], ascending=[True, True])\
                        .groupby(['pid', 'rid', 'dir'])\
                        .head(1)[['pid', 'rindex', 'rid', 's', 'e', 'dir' ,'dist_to_line']]
        df_new[['pid', 'rindex', 'rid', 's', 'e','dist_to_line']].reset_index(drop=True)
        df_new = df_new.groupby('pid').head(top_k).reset_index(drop=True)
        
        if logger is not None:
            logger.debug(f"Shrink candidate link {origin_size} -> {df_new.shape[0]} by choose the closest link in a road")

        return df_new

    radius = georadius*dis_factor
    boxes = traj.geometry.apply(lambda i: box(i.x-radius, i.y-radius,i.x+radius, i.y+radius))
    df_candidates = boxes.apply(lambda x: edges.sindex.query(x, predicate='intersects')).explode().dropna()
    
    if df_candidates.shape[0] == 0:
        return None
    
    df_candidates = pd.DataFrame(df_candidates).reset_index().rename(columns={'index': 'pid', 'geometry':'rindex'})
    df_candidates = df_candidates.merge(traj['geometry'], left_on='pid', right_index=True)\
                                 .merge(edges['geometry'], left_on='rindex', right_index=True)\
                                 .rename(columns={'geometry_x': 'point_geom', 'geometry_y': 'edge_geom'})\
                                 .sort_index()
    # set_crs: out_sys=32649;  CRS(f"EPSG:{out_sys}")
    df_candidates.loc[:, 'dist_to_line'] = df_candidates.apply(lambda x: x.point_geom.distance(x.edge_geom) / DIS_FACTOR, axis=1)

    if top_k is not None:
        candidates_filtered = _filter_candidate(df_candidates, top_k)
        
    if plot:
        ax = edges.loc[df_candidates.rindex.values].plot()
        if top_k is not None:
            edges.loc[candidates_filtered.rindex.values].plot(ax=ax, color='red')
        traj.plot(ax=ax)

    return candidates_filtered if filter else df_candidates.sort_values(['pid', 'dist_to_line']).groupby('pid').head(5)


def cal_observ_prob(df, std_deviation=20):
    observ_prob_factor = 1 / (np.sqrt(2*np.pi) * std_deviation)
    
    def helper(x):
        return observ_prob_factor * np.exp(-np.power(x, 2)/(2*np.power(std_deviation, 2)))
    
    df.loc[:, 'observ_prob'] = df.dist_to_line.apply( helper)
    df.loc[:, 'observ_prob'] = df.observ_prob / df.observ_prob.max()
    
    observ_prob_dict = df.set_index(['pid', 'rindex'])['observ_prob'].to_dict()

    return observ_prob_dict


def cal_relative_offset(node:Point, polyline:LineString, verbose=False):
    """Calculate the relative offset between the node's foot and the polyline.

    Args:
        node (Point): [description]
        polyline (LineString): [description]
        verbose (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
        
    Example: 
        ```
        from shapely import wkt
        node = wkt.loads('POINT (114.051228 22.539597)')
        linestring = wkt.loads('LINESTRING (114.0516508 22.539516, 114.0515715 22.5395317, 114.0515222 22.5395533, 114.0514758 22.5395805, 
        114.0514441 22.5396039, 114.0514168 22.5396293, 114.0513907 22.5396616, 114.0513659 22.5396906, 114.0513446 22.5397236, 114.0512978 22.5398214)')
        cal_relative_offset(node, linestring)
        ```
    """
    # The case continuous repeating point
    lines = [LineString((polyline.coords[i], polyline.coords[i+1])) for i in range(len(polyline.coords)-1) if polyline.coords[i] != polyline.coords[i+1] ]
    lines = gpd.GeoDataFrame( {'geometry': lines} )

    # 避免使用保留字段
    # lines.loc[:, 'length'] = lines.geometry.apply(lambda x: coords_pair_dist(x.coords[0], x.coords[-1], xy=True))
    lines.loc[:, '_len'] = lines.geometry.apply(lambda x: coords_pair_dist(x.coords[0], x.coords[-1], xy=True))
    lines.loc[:, '_dist'] = lines.geometry.apply(lambda x: x.distance(node)/DIS_FACTOR)

    nearest_line_id = lines['_dist'].idxmin()

    dist_prev_lines = np.sum(lines.loc[:nearest_line_id]._len.values[:-1])

    line = lines.loc[nearest_line_id].geometry
    foot_dict = cal_foot_point_on_polyline(node, line)
    _dist = coords_pair_dist(foot_dict['foot'], line.coords[0], xy=True)

    return dist_prev_lines + _dist if foot_dict['flag'] > 0 else dist_prev_lines - _dist


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
    
    lst = gpd.GeoDataFrame([ {'s': path[i], 'e': path[i+1]} for i in range(len(path)-1) ])
    lines = lst.merge(net.df_edges, on=['s', 'e']).geometry.values
    points = [ l.coords[:] for l in lines ]
    
    res = []
    for lst in points:
        res += lst
    
    return LineString(res)


def cal_trans_prob(df_candidates, traj, net):
    df_candidates.loc[:, 'offset'] = df_candidates.apply(lambda x: 
        cal_relative_offset(traj.loc[x.pid].geometry, net.df_edges.loc[x.rindex].geometry), axis=1 )

    tList, graph_t = [], []
    for _, sub in df_candidates.groupby('pid'):
        tList.append(sub)

    for i in range(len(tList)-1):
        base_atts = ['pid', 'rindex','s', 'e', 'offset']
        a, b = tList[i][base_atts], tList[i+1][base_atts+['observ_prob']]
        # Cartesian product
        a.loc[:, 'tmp'], b.loc[:, 'tmp'] = 1, 1
        graph_t.append(a.merge(b, on='tmp', suffixes=["_0", '_1']).drop(columns='tmp') )

    def _od_in_same_link(gt):
        # 匹配在同一个线段上面，寻找最短路的 src ＆ dst 的设置
        # The case: o and d all on the same link
        con = gt.rindex_0 == gt.rindex_1
        
        # case 1: 轨迹前行方向 和 道路前进方向 一致
        case1 = con & (gt.offset_0 < gt.offset_1)
        if case1.sum()>0:
            # gt.loc[case1, 'e_0'] = gt.loc[case1, 's_0']
            # gt.loc[case1, 's_1'] = gt.loc[case1, 'e_1']
            gt.loc[:, 'same_link_forward'] = False
            gt.loc[case1, 'same_link_forward'] = True
            if logger is not None :
                logger.warning(f"Adjust the (o, d) for the case the trajectory moving direction is the same with the road direction:\n\t {gt.loc[case1].to_dict(orient='record')}")
        
        # case 2: 轨迹前行方向 和 道路前进方向 相反，就是默认的逻辑
        # case2 = con & (gt.offset_0 > gt.offset_1)
        
        # TODO
        # case 3：相同的情况, 无需处理
        case3 = con & (gt.offset_0 == gt.offset_1)
        if case3.sum()>0:
            gt.loc[:, 'same_link_same_point'] = False
            gt.loc[case1, 'same_link_same_point'] = True
            if logger is not None:
                logger.warning(f"The overlap of trajectory points:\n\t {gt.loc[case3].to_dict(orient='record')}")
        
        gt.e_0 = gt.e_0.astype(np.int)
        gt.s_1 = gt.s_1.astype(np.int)
        
        return gt    

    ordered_cols = ['pid_0', 'pid_1', 'rindex_0', 'rindex_1', 's_0', 'e_0', 's_1', 'e_1', 'offset_0', 'offset_1', 'observ_prob']
    graph_t = pd.concat(graph_t)[ordered_cols].reset_index(drop=True)
    graph_t = _od_in_same_link(graph_t)

    graph_t.loc[:, 'shortest_path'] = graph_t.apply(lambda x: net.a_star(x.e_0, x.s_1, plot=False), axis=1)
    graph_t.loc[:, 'd_sht']   = graph_t.shortest_path.apply(lambda x: x['cost'] if x is not None else np.inf )
    graph_t.loc[:, 'd_euc']   = graph_t.apply(lambda x: coords_pair_dist(traj.loc[x.pid_0].geometry, traj.loc[x.pid_1].geometry), axis=1)
    graph_t.loc[:, 'd_step0'] = graph_t.apply(lambda x: net.edge[x.s_0, x.e_0], axis=1)

    if 'same_link_forward' in graph_t.columns:
        graph_t.loc[graph_t.same_link_forward, 'd_sht'] = -graph_t.loc[graph_t.same_link_forward, 'd_step0']
        graph_t.loc[graph_t.same_link_forward, 'shortest_path' ] = None

    graph_t.loc[:, 'w']       = graph_t.d_sht + (graph_t.d_step0 - graph_t.offset_0) + graph_t.offset_1 
    graph_t.loc[:, 'v']       = graph_t.apply(lambda x:  x.d_euc/x.w if x.d_euc/x.w <= 1 else 1, axis=1 )  
    
    # The case: o and d all on the same link
    if 'same_link_same_point' in graph_t.columns:
        graph_t.loc[graph_t.same_link_same_point, ['v', 'shortest_path']] = [1, None]

    graph_t.loc[:, 'f'] = graph_t.v * graph_t.observ_prob
    
    gt = graph_t.drop_duplicates(['e_0','s_1']).set_index(['e_0', 's_1'])

    return tList, gt, graph_t


def find_matched_sequence(graph_t, df_candidates, tList, drop_dulplicates=True, logger=logger):
    prev_dict, f_score = {}, {}

    for i, item in tList[0].iterrows():
        f_score[i] = item.observ_prob

    for i in range(1, len(tList)):
        for j, nxt in tList[i].iterrows():
            _max = -np.inf
            for k, cur in tList[i-1].iterrows():
                alt = f_score[k] + graph_t.loc[cur.e].loc[nxt.s].f
                if alt > _max:
                    _max = alt
                    prev_dict[j] = k
                f_score[j] = _max
    
    rList = []
    c = max(f_score, key=f_score.get)
    
    for i in range(len(tList)-1, 0, -1):
        rList.append(c)
        c = prev_dict[c]
    rList.append(c)

    logger.debug(f'max score: {c}, f_score: {f_score}')
    logger.debug(f'{rList}')
    
    rList = df_candidates.loc[rList[::-1]][['s', 'e']]

    if drop_dulplicates:
        rList = rList[(rList.s != rList.s.shift(1)) | (rList.e != rList.e.shift(1))]
    
    return rList


def get_path(rList, graph_t, net):
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
        res = graph_t.loc[x.e].loc[x.nxt_s].shortest_path
        
        if res is None:
            return None
        
        return res['path']
    
    rList.loc[:, 'nxt_s'] = rList.s.shift(-1).fillna(0).astype(np.int)
    steps = rList[:-1].apply(lambda x: _helper(x), axis=1)

    if len(steps) == 1:
        return None

    coords = []
    for step in steps.values:
        if step is None:
            continue
        coords += step
    print(coords)
    
    path = net.node_sequence_to_edge(coords)
    
    return path


def drop_consecutive_identical_points(traj):
    con = traj.geometry == traj.geometry.shift(1)
    
    return traj[~con]

def st_matching(traj, net, std_deviation=20, georadius=50, plot=True, save_fn=None, debug_in_levels=False, plot_candidate=False, satellite=False):
    if traj.shape[0] == 0:
        return None

    traj = drop_consecutive_identical_points(traj)

    # step 1: candidate prepararation
    df_candidates = get_candidates(traj, net.df_edges, georadius=georadius)
    if df_candidates is None:
        return None
    
    if df_candidates.pid.nunique() <= 1:
        #TODO The cloesest edge.
        print('Only one level has candidates.')
        return
    
    # step 2.1: Spatial analysis, obervation prob
    cal_observ_prob(df_candidates, std_deviation)
    
    # step 2.2: Spatial analysis, transmission prob
    tList, graph_t, gt_with_dulplicates = cal_trans_prob(df_candidates, traj, net)

    # TODO step 3: temporal analysis
    # step 4: find matched sequence
    rList = find_matched_sequence(graph_t, df_candidates, tList)
    path = get_path(rList, graph_t, net)

    if debug_in_levels:
        matching_debug(traj, tList, gt_with_dulplicates, net, debug_in_levels)

    if not plot:
        return path

    def _matching_plot():
        # plot， trajectory point
        if satellite:
            try:
                _, ax = map_visualize(traj, alpha=.5, scale=.2, color='blue')
            except:
                ax = traj.plot(alpha=.5, color='blue')
                ax.axis('off')       
        else:
            ax = traj.plot(alpha=.5, color='blue')
            ax.axis('off')
            
        traj.head(1).plot(ax=ax, marker = '*', color='red', zorder=9, label= 'Start point')
        # network
        edge_lst = net.df_edges.sindex.query(box(*traj.total_bounds), predicate='intersects')
        net.df_edges.loc[edge_lst].plot(ax=ax, color='black', linewidth=.8, alpha=.3, label='Network' )
        # candidate
        net.df_edges.loc[df_candidates.rindex.values].plot(
            ax=ax, label='Candidates', color='blue', linestyle='--', linewidth=.8,alpha=.8)
        # path
        if path is not None:
            path.plot(ax=ax, label='Path', color='red', alpha=.5)
        plt.legend()
        
        if save_fn is not None:
            plt.savefig(os.path.join(DEBUG_FOLDER, f'{save_fn}.jpg'), dpi=300)
            plt.close()

    _matching_plot()

    return path



#%%
# 42, 64 卫星瓦片请求失败
# 15, 可能一条不存在的道路
# 38 放大查看
# 52 匝道处理
# 143， 148


# %%

def check(fn):
    traj = load_trajectory(fn).reset_index()
    
    traj = drop_consecutive_identical_points(traj)

    # step 1: candidate prepararation
    df_candidates = get_candidates(traj, net.df_edges, georadius=50, plot=True)
    # step 2.1: Spatial analysis, obervation prob
    cal_observ_prob(df_candidates, std_deviation=20)

    # step 2.2: Spatial analysis, transmission prob
    tList, graph_t, gt_with_dulplicates = cal_trans_prob(df_candidates, traj, net)

    # step 4: find matched sequence
    rList = find_matched_sequence(graph_t, df_candidates, tList)
    path = get_path(rList, graph_t, net)

    matching_debug(traj, tList, gt_with_dulplicates, net, debug=True)

fn ="../input/traj_debug.geojson"
# # fn ="../input/traj_debug_141.geojson"
traj = load_trajectory(fn).reset_index()
check(fn)


#%%

node, polyline = traj.iloc[1].geometry, net.df_edges.loc[115094].geometry

cal_relative_offset(traj.iloc[0].geometry, net.df_edges.loc[115094].geometry)

cal_relative_offset(traj.iloc[1].geometry, net.df_edges.loc[115094].geometry)

#%%
#%%
if __name__ == '__main__':
    """ Load network object """
    net = load_net_helper(bbox=SZ_BBOX, combine_link=True, convert_to_geojson=True)

    """ a_star 最短路算法测试 """
    # net.a_star(1491845161, 1933843924, plot=True)

    """ test for cal_relative_offset """
    # pid, ridx = df_candidates.iloc[0]['pid'], df_candidates.iloc[0]['rindex']
    # node = traj.loc[pid].geometry
    # line = net.df_edges.loc[ridx].geometry
    # for i in [925, 926, 927, 928]:
    #     cal_relative_offset(node, net.df_edges.loc[i].geometry)

    # for i in [1103, 979, 980, 981]:
    #     cal_relative_offset(node, net.df_edges.loc[i].geometry)

    """" matching plot debug helper """
    # matching_debug(tList, graph_t)
    # matching_debug_level(tList, graph_t, 3)
    # matching_debug_level(tList, graph_t, 2)

    """ matching test"""
    traj = load_trajectory("../input/traj_0.geojson")
    path = st_matching(traj, net, plot=True)

    # st_matching(traj, net, False)
    # error_lst = [ 37, 38, 40]
    # planning_error = [43, 50, 56] 
    # key_error = [64]
    # uncontinous = [ 31, 40, 42,69, 76, 119] # 修改filter后解决问题
    # code_testing(65, 150)


    """ matching test"""
    # traj = load_trajectory("../input/traj_debug.geojson")
    # path = st_matching(traj, net, plot=True, debug_in_levels=False, satellite=True)

