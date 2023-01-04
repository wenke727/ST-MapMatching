import numpy as np
import pandas as pd

from ..utils.interval_helper import merge_intervals
from ..utils.parallel_helper import parallel_process


def calculate_degree(df_edges):
    indegree = df_edges.groupby('dst').agg({'order': 'count'}).rename(columns={'order': 'indegree'})
    outdegree = df_edges.groupby('src').agg({'order': 'count'}).rename(columns={'order': 'outdegree'})

    return pd.concat([indegree, outdegree], axis=1).fillna(0).astype(int)

def get_aux_points(df_edges, exclude_list=None):
    degree = calculate_degree(df_edges)
    
    aux_node_lst = degree.query( "indegree == 1 and outdegree == 1" ).index.unique()
    if exclude_list is not None:
        aux_node_lst = [id for id in aux_node_lst if id not in exclude_list]

    return aux_node_lst

def combine_links(edges, combine_intervals):
    """Combine OSM links with `rid`.

    Args:
        rid (int): The id of link in OSM.
        nodes (gdf.GeoDataFrame): The all nodes related to `rid` road.
        links (gdf.GeoDataFrame): The all links related to `rid` road. 
        omit_points (list): The points don't meet: 1) only has 1 indegree and 1 outdegree; 2) not the traffic_signals point.

    Returns:
        gpd.GeoDataFrame: The links after combination.
    """
    if len(combine_intervals) == 0:
        return edges
    
    if 'order' in edges.columns:
        edges.set_index('order', inplace=True)

    drop_index = []
    for start, end, _ in combine_intervals:
        segs = edges.query(f"{start} <= order <= {end}")
        _dst = segs.iloc[-1]['dst']
        nids = np.append(segs.src.values, _dst)

        edges.loc[start, 'dst'] = _dst
        edges.loc[start, 'dist'] = segs.dist.sum()
        edges.loc[start, "waypoints"] = ",".join((str(i) for i in nids))

        drop_index += [i for i in range(start+1, end+1)]

    edges.drop(index=drop_index, inplace=True)
    edges.reset_index(inplace=True)
    
    return edges

def pipeline_combine_links(df_edges:pd.DataFrame, exclude_list, n_jobs=8):
    # BUG multi_edges
    aux_nids = get_aux_points(df_edges, exclude_list=exclude_list)

    cands_edges = df_edges.query("src in @aux_nids").sort_values(by=['way_id', 'order'])
    cands_way_ids = cands_edges.way_id.unique().tolist()
    aux_edge_intervals = cands_edges.groupby('way_id')\
                                    .order.apply(list)\
                                    .apply(lambda lst: merge_intervals([[i-1, i] for i in lst if i > 0]))

    # parallel process, 不能使用 cands_edges，因为涉及到上下游的合并
    _df_edges = df_edges.query(f"way_id in @cands_way_ids")
    params = ((df, aux_edge_intervals[i]) 
                for i, df in _df_edges.groupby('way_id'))
    combined_edges = parallel_process(combine_links, params, pbar_switch=True, 
                                      n_jobs=n_jobs, total=len(cands_way_ids), desc='Combine edges')

    # keep edges
    keep_edges = df_edges.query(f"way_id not in @cands_way_ids")

    # combine
    df_edges_ = pd.concat(combined_edges + [keep_edges]).sort_values(['way_id', 'order']).reset_index(drop=True)
    
    return df_edges_
