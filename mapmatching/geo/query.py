import shapely
import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
import warnings

from .ops import project_points_2_linestring
from .haversineDistance import haversine_geoseries


def get_K_neigh_geoms(query: GeoDataFrame, gdf: GeoDataFrame, query_id='qid', radius: float = 5e-4, top_k=5,
                    predicate: str = 'intersects', check_diff=True, ll=True, project=True, keep_geom=False, normalized=True):
    """Get k nearest geometry of query, within a searching radius. 
    This step can be efficiently perfermed with the build-in grid-based spatial index.

    Args:
        query (GeoDataFrame, GeoSeries, geometry): Query object
        gdf (GeoDataFrame): Base geometry
        query_id (str, optional): The index of query object. Defaults to 'qid'.
        radius (float, optional): The searching radius. Defaults to 5e-4 (about 50m is wgs system).
        top_k (int, optional): The topk elements. Defaults to 5.
        predicate (str, optional): predicate ops in geopandas. Defaults to 'intersects'.
        check_diff (bool, optional): Check if there is no mathcing query. Defaults to True.
        ll (bool, optional): lat/lon system. Defaults to True.
        project (bool, optional): Project the query object to gdf. Only support Point. Defaults to True.

    Returns:
        GeoDataFrame: Query result
    """

    """
    Example:
    ```
        # case 1
        import geopandas as gpd
        from stmm.geo.query import get_K_neighbors

        traj = matcher.load_points("./data/trajs/traj_4.geojson").head(4)
        query = traj[['PID','geometry']].head(1).copy()
        gdf = net.df_edges[['eid', 'geometry']].copy()

        df_cands, no_cands_query = get_K_neighbors(query, gdf, top_k=8)
        plot_candidates(query, gdf, df_cands)

        # case 2
        import geopandas as gpd
        from shapely import LineString, Point
        from stmm.geo.query import plot_candidates, get_K_neighbors
        # edges
        lines = [LineString([[0, i], [10, i]]) for i in range(0, 10)]
        lines += [LineString(([5.2,5.2], [5.8, 5.8]))]
        edges = gpd.GeoDataFrame({'geometry': lines, 
                                    'way_id':[i for i in range(10)] + [5]})
        # points
        a, b = Point(1, 1.1), Point(5, 5.1) 
        points = gpd.GeoDataFrame({'geometry': [a, b]}, index=[1, 3])
        points.loc[:, 'PID'] = points.index
        # candidates

        res, _ = get_K_neighbors(points, edges, buffer=2, top_k=2, ll=False)
        ax = plot_candidates(points, edges, res)
    ```
    """
    # TODO gdf 的 index 怎么设置比较合适
    # check sindex
    if not gdf.has_sindex:
        try:
            gdf.sindex
        except:
            raise ValueError()

    # prepare query
    if isinstance(query, shapely.geometry.base.BaseGeometry):
        query = gpd.GeoSeries([query])
    if isinstance(query, GeoDataFrame):
        if query_id in list(query):
            _query = query.set_index(query_id)['geometry']
        else:
            _query = query['geometry'].copy()
            _query.index.set_names(query_id, inplace=True)
    elif isinstance(query, gpd.GeoSeries):
        _query = query.copy()
        _query.index.set_names(query_id, inplace=True)
    else:
        raise TypeError(query)
    
    # spatial query
    query_geoms = _query.apply(lambda i: shapely.box(i.x - radius, i.y - radius, i.x + radius, i.y + radius))
    
    # cands = gdf.sindex.query_bulk(
        # _query.buffer(radius) if radius else _query, predicate)
    cands = gdf.sindex.query_bulk(query_geoms, predicate)
    if len(cands[0]) == 0:
        return None

    _points = _query.iloc[cands[0]]
    df_cands = gdf.iloc[cands[1]].rename(columns={'geometry': 'edge_geom'}).reset_index(drop=True)
    edge_atts = list(df_cands)
    df_cands.loc[:, query_id] = _points.index
    df_cands.loc[:, "query_geom"] = _points.values
    df_cands = df_cands[[query_id] + edge_atts + ['query_geom']]

    # dist_p2c
    if project:
        assert np.all(query.geom_type == 'Point'), "Project only support `Point`"
        get_cut_point = lambda x: project_points_2_linestring(x['query_geom'], x['edge_geom'], normalized)
        df_cands.loc[:, ['proj_point', 'offset']] = df_cands.apply(get_cut_point, axis=1, result_type='expand').values

        if not ll:
            cal_proj_dist = lambda x: x['query_geom'].distance(x['proj_point'])
            df_cands.loc[:, 'dist_p2c'] = df_cands.apply(cal_proj_dist, axis=1)
        else:
            df_cands.loc[:, 'dist_p2c'] = haversine_geoseries(
                df_cands['query_geom'], df_cands['proj_point'])
    else:
        cal_proj_dist = lambda x: x['query_geom'].distance(x['edge_geom'])
        df_cands.loc[:, 'dist_p2c'] = df_cands.apply(cal_proj_dist, axis=1)
    df_cands = _filter_candidate(df_cands, query_id, top_k)
    
    if not keep_geom:
        df_cands.drop(columns=["query_geom", "edge_geom"], inplace=True)

    # check diff
    no_cands_query = None
    if check_diff:
        cands_pid = set(cands[0])
        all_pid = set(_query.index.unique())
        no_cands_query = all_pid.difference(cands_pid)
        warnings.warn(f"{no_cands_query} has not neighbors within the {radius} searching zone.")

    return df_cands, no_cands_query

def plot_candidates(points, edges, match_res):
    ax = points.plot()
    # edges.plot(ax=ax)

    res = gpd.GeoDataFrame(match_res)
    res.set_geometry('edge_geom', inplace=True)

    for _, group in res.groupby('PID'):
        group.plot(ax=ax, column='dist_p2c', alpha=.5, linestyle='--', legend=True, cmap='RdBu_r')
        
    return ax

def _filter_candidate(df: gpd.GeoDataFrame,
                      pid: str = 'pid',
                      top_k: int = 5,
                      ):
    """Filter candidates, which belongs to the same way, and pickup the nearest one.

    Args:
        df (gpd.GeoDataFrame): df candidates.
        top_k (int, optional): _description_. Defaults to 5.
        pid (str, optional): _description_. Defaults to 'pid'.

    Returns:
        gpd.GeoDataFrame: The filtered candidates.
    """
    # origin_size = df.shape[0]
    df = df.sort_values([pid, 'dist_p2c'])\
           .groupby(pid)\
           .head(top_k)\
           .reset_index(drop=True)

    return df
