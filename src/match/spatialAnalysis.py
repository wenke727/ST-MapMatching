#%%
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame

import sys
sys.path.append('..')

from graph.geograph import GeoDigraph
from geo.geo_helper import point_to_polyline_process, coords_pair_dist
from geo.azimuth_helper import azimuth_cos_similarity_for_linestring, azimuthAngle
from match.geometricAnalysis import project_point_to_line_segment, cal_observ_prob

pd.set_option('display.max_columns', 500)

""" candidatesGraph """
def _construct_graph(traj, cands):
    """
    Construct the candiadte graph (level, src, dst) for spatial and temporal analysis.
    """
    graph = []
    tList = [layer for _, layer in cands.groupby('pid')]

    base_atts = ['pid', 'eid','src', 'dst', 'len_0', 'len_1', 'seg_0', 'seg_1']
    cols_filter = [
        'pid_0',
        'pid_1',
        'eid_0',
        'eid_1',
        'dst_0',
        'src_1',
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


def get_att_helper(item, ids):
    if isinstance(ids, (list, tuple)):
        if len(ids) == 1:
            return item[ids[0]]
        return tuple(item[k] for k in ids)
    
    if isinstance(ids, str):
        return item[ids]
    
    return NotImplementedError


def construct_graph(df, level, keys):
    """
    Construct the candiadte graph (level, src, dst) for spatial and temporal analysis.
    """
    layers = {}
    graph = {}
    tList = [layer for _, layer in df.groupby(level)]

    for l in range(0, len(tList) - 1):
        cur_level, nxt_level = tList[l], tList[l + 1]
        layers[l] = {}
        
        for _, i in cur_level.iterrows():
            src = get_att_helper(i, keys)
            layers[l][src] = set()
            for __, j in nxt_level.iterrows():
                dst = get_att_helper(j, keys)
                link = (src, dst)
                layers[l][src].add(dst)
                graph[link] = {'src': i.dst, 'dst': j.src}

    return layers, graph


""" Geometric info"""

def _move_dir_similarity(traj:GeoDigraph, graph:GeoDataFrame, geograph):
    if 'geometry' not in list(graph):
        graph.loc[:, 'geometry'] = graph.path.apply(geograph.transform_node_seq_to_polyline)
                
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


""" topological info """
def _trans_prob(graph:GeoDataFrame, net:GeoDigraph, max_steps:int, max_dist:int):
    same_link_mask = graph.eid_0 == graph.eid_1
    ods = graph[~same_link_mask][['dst_0', 'src_1']].drop_duplicates().values
    if len(ods) > 0:
        df_planning = pd.DataFrame([
            {'dst_0':o, 
             'src_1':d, 
             **net.search(o, d, max_steps=max_steps, max_dist=max_dist) 
             } for o, d in ods
        ])

        graph = graph.merge(df_planning, on=['dst_0', 'src_1'], how='left')
        # `w` is the shortest path from `ci-1` to `ci`
        graph.loc[:, 'w'] = graph.cost + graph.offset_0 + graph.offset_1 
        # transmission probability
        graph.loc[:, 'v'] = graph.apply(lambda x: x.d_euc / x.w if x.d_euc < x.w else x.w / x.d_euc * 1.00, axis=1 )

    graph.loc[same_link_mask, 'path'] = graph.loc[same_link_mask, 'eid_0'].apply(
        lambda x: net.get_edge(x, att=['src', 'dst']).values.tolist())
    graph.loc[same_link_mask, 'v'] = 1
    
    return graph


""" spatial_analysis """
def analyse_spatial_info(geograph, points, cands, dir_trans=False):
    """Geometric and topological info, the product of `observation prob` and the `transmission prob`
    
    Special Case:
        a. same_link_same_point
    """
    cands = project_point_to_line_segment(geograph, points, cands)
    cands.loc[:, 'observ_prob'] = cal_observ_prob(cands.dist_p2c)
    gt = _construct_graph(points, cands)
    gt = _trans_prob(gt, geograph, 20000, 2000)
    if dir_trans:
        gt = _move_dir_similarity(points, gt, geograph)
        
    # spatial analysis: observ_prob * trans_prob
    gt.loc[:, 'f'] = gt.observ_prob * gt.v * (gt.f_dir if dir_trans else 1)

    atts = ['pid_0', 'eid_0', 'eid_1']
    gt = gt.drop_duplicates(atts).set_index(atts).sort_index()

    return cands, gt


if __name__ == "__main__":
    #! graph 
    from shapely import wkt
    import geopandas as gpd
    from tqdm import tqdm
    from osmnet.build_graph import build_geograph
    import heapq

    # 读取轨迹点
    points = gpd.read_file("../../input/traj_1.geojson")
    
    # 读取数据
    fn = '../../tmp/candidates.geojson'
    cands = gpd.read_file(fn, crs='epsg:4326')
    cands.loc[:, 'point_geom'] = cands.point_geom.apply(wkt.loads)
    
    # 读取基础路网
    fn = '../../cache/Shenzhen_graph.ckpt'
    geograph = build_geograph(ckpt=fn)

    cands, gt = analyse_spatial_info(geograph, points, cands)
    
#%%

    # 构建图片
    layers, gt = construct_graph(cands, 'pid', ['eid'])
    
    # tmp = project_point_to_line_segment(geograph, points, candidates)
    
    #%%
    # 构建优先队列, 最小堆
    queue = []
    factor = 5
    counter = 0
    
    for level, links in layers.items():
        for o, nxts in links.items():
            remained_nodes = set()
            for d in nxts:
                counter += 1
                info = geograph.search(**gt[(o, d)])
                if 'cost' not in info:
                    continue
                if queue and info.get('cost', np.inf) > queue[0][0] * factor:
                    continue
                
                remained_nodes.add(d)
                heapq.heappush(queue, (info['cost'], (o, d)))
                
        break
    
    print(len(remained_nodes), len(queue) / counter)
    
    #%%
    for keys, od in gt.items():
        info = geograph.search(**od)
        # od.update(info)

# %%
