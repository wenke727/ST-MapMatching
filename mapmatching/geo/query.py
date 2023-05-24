import shapely
import warnings
import numpy as np
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely import geometry as shapely_geom

from .ops.linear_referencing import linear_referencing_geom
from ..utils import timeit

@timeit
def get_k_neigh_geoms(query: GeoDataFrame, gdf: GeoDataFrame, query_id='qid', 
                      radius: float = 50, top_k=None, predicate: str = 'intersects', 
                      check_diff=True, project=True, keep_geom=True):
    """
    Get the k nearest geometries of the query within a search radius using a built-in grid-based spatial index.

    Args:
        query (GeoDataFrame, GeoSeries, geometry): The query object.
        gdf (GeoDataFrame): The base geometry.
        query_id (str, optional): The index of the query object. Defaults to 'qid'.
        radius (float, optional): The search radius. Defaults to 50 (in meters for the WGS system).
        top_k (int, optional): The number of top-k elements to retrieve. Defaults to None (retrieve all).
        predicate (str, optional): The predicate operation in geopandas. Defaults to 'intersects'.
        check_diff (bool, optional): Check if there are no matching queries. Defaults to True.
        project (bool, optional): Project the query object to gdf. Only supports Point geometries. Defaults to True.
        keep_geom (bool, optional): Whether to keep the geometry columns in the result. Defaults to True.
        normalized (bool, optional): Normalize the distances. Defaults to False.

    Returns:
        GeoDataFrame: The query result.

    Example:
        # Example usage 1
        import geopandas as gpd
        from stmm.geo.query import get_K_neighbors

        traj = matcher.load_points("./data/trajs/traj_4.geojson").head(4)
        query = traj[['PID','geometry']].head(1).copy()
        gdf = net.df_edges[['eid', 'geometry']].copy()

        df_cands, no_cands_query = get_K_neighbors(query, gdf, top_k=8)
        plot_candidates(query, gdf, df_cands)

        # Example usage 2
        import geopandas as gpd
        from shapely import LineString, Point
        from stmm.geo.query import plot_candidates, get_K_neighbors

        lines = [LineString([[0, i], [10, i]]) for i in range(0, 10)]
        lines += [LineString(([5.2,5.2], [5.8, 5.8]))]
        edges = gpd.GeoDataFrame({'geometry': lines, 
                                    'way_id':[i for i in range(10)] + [5]})

        a, b = Point(1, 1.1), Point(5, 5.1) 
        points = gpd.GeoDataFrame({'geometry': [a, b]}, index=[1, 3])
        points.loc[:, 'PID'] = points.index

        res, _ = get_K_neighbors(points, edges, buffer=2, top_k=2, ll=False)
        ax = plot_candidates(points, edges, res)
    """

    # TODO: Determine appropriate index for gdf

    # Check spatial index
    if not gdf.has_sindex:
        try:
            print("rebuild sindex: ")
            gdf.sindex
        except:
            raise ValueError()

    # Prepare query
    if isinstance(query, shapely_geom.base.BaseGeometry):
        _query = gpd.GeoSeries([query])
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

    if _query.crs != gdf.crs:
        _query = _query.to_crs(gdf.crs)
    _query.index.set_names(query_id, inplace=True)

    # Query bulk
    get_box = lambda i: shapely_geom.box(i.x - radius, i.y - radius, i.x + radius, i.y + radius)
    query_geoms = _query.apply(get_box)
    cands = gdf.sindex.query_bulk(query_geoms, predicate)
    if len(cands[0]) == 0:
        return None, None

    df_cands = _get_cands(_query, gdf, cands, query_id)
    _project(df_cands, project)

    if radius:
        df_cands.query(f"dist_p2c <= {radius}", inplace=True)
    if top_k:
        df_cands = _filter_candidate(df_cands, query_id, top_k)

    if not keep_geom:
        df_cands.drop(columns=["query_geom", "edge_geom"], inplace=True)

    # Check difference
    no_cands_query = None
    if check_diff:
        cands_pid = set(cands[0])
        all_pid = set(_query.index.unique())
        no_cands_query = all_pid.difference(cands_pid)
        warnings.warn(f"{no_cands_query} has no neighbors within the {radius} search zone.")

    return df_cands.set_geometry('edge_geom').set_crs(gdf.crs), no_cands_query


@timeit
def _get_cands(_query, gdf, cands, query_id):
    _points = _query.iloc[cands[0]]
    
    df_cands = gdf.iloc[cands[1]]
    df_cands.rename(columns={'geometry': 'edge_geom'}, inplace=True)
    df_cands.loc[:, query_id] = _points.index
    df_cands.loc[:, "query_geom"] = _points.values

    return df_cands

@timeit
def _project(df_cands, project=True):
    # dist_p2c
    if not project:
        cal_proj_dist = lambda x: x['query_geom'].distance(x['edge_geom'])
        df_cands.loc[:, 'dist_p2c'] = df_cands.apply(cal_proj_dist, axis=1)

        return df_cands

    df_projs = linear_referencing_geom(df_cands['query_geom'], df_cands['edge_geom'])
    df_cands.loc[:, df_projs.keys()] = df_projs.values()
    # df_cands = gpd.GeoDataFrame(df_cands, crs=gdf.crs, geometry='proj_point')

    return df_cands
    

def plot_candidates(cands):
    # TODO draw buffer
    from ..geo.vis import plot_geodata
    _, ax = plot_geodata(cands, color='r', tile_alpha=.6, alpha=0)

    cands.set_geometry('edge_geom').plot(ax=ax, column='dist_p2c', cmap='Reds_r', legend='candidates')
    if 'proj_point' in list(cands):
        cands.loc[:, 'proj_point'] = cands['proj_point'].apply(shapely.Point)
        cands.set_geometry('proj_point').plot(ax=ax, cmap='Reds_r')
    cands.set_geometry('query_geom').plot(ax=ax, marker='*', label='Point', zorder=9)

    return ax

@timeit
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
