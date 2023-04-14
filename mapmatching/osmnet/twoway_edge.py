import shapely
import numpy as np
import pandas as pd
from loguru import logger
from shapely.geometry import LineString


def swap_od(df_edge_rev:pd.DataFrame, od_attrs=['src', 'dst']):
    if df_edge_rev.empty:
        return df_edge_rev
    
    df_edge_rev.loc[:, 'dir']       = -1
    df_edge_rev.loc[:, 'order']     = -df_edge_rev.order - 1
    df_edge_rev.loc[:, 'waypoints'] = df_edge_rev.waypoints.apply(lambda x: x[::-1])
    df_edge_rev.rename(columns={od_attrs[0]: od_attrs[1], od_attrs[1]: od_attrs[0]}, inplace=True)
    if 'geometry' in list(df_edge_rev):
        df_edge_rev.loc[:, 'geometry']  = df_edge_rev.geometry.apply(lambda x: LineString(x.coords[::-1]) )

    return df_edge_rev


def add_reverse_edge(df_edges, df_ways, od_attrs=['src', 'dst'], offset=True):
    """Add reverse edge.

    Args:
        df_edges (gpd.GeoDataFrame): The edge file parsed from OSM.
    Check:
        rid = 34900355
        net.df_edges.query( f"rid == {rid} or rid == -{rid}" ).sort_values(['order','rid'])
    """
    assert 'oneway' in df_ways.columns, "Check `oneway` tag"
    df_edges.loc[:, 'dir'] = 1

    idxs = df_ways.query('oneway == False').index
    df_edge_rev = df_edges.query("way_id in @idxs")

    has_geom = 'geometry' in list(df_edges)
    if has_geom:
        ring_mask = df_edge_rev.geometry.apply(lambda x: x.is_ring)
        df_edge_rev = df_edge_rev[~ring_mask]

    df_edge_rev = swap_od(df_edge_rev, od_attrs)

    df_edges = pd.concat([df_edges, df_edge_rev]).reset_index(drop=True)

    if offset:
        df_edges = edge_offset(df_edges)
    
    return df_edges


def edge_offset(df_edges):
    way_ids = df_edges.query("dir == -1").way_id.unique()
    _df_edges = df_edges.query("way_id in @way_ids")
    
    _df_edges.loc[:, 'geom_origin'] = _df_edges.geometry.copy()
    # df_edge.loc[:, 'geom_origin'] = df_edge.geometry.apply(lambda x: x.to_wkt())
    geom_offset = _df_edges.apply( lambda x: parallel_offset_edge(x), axis=1 )
    _df_edges.loc[geom_offset.index, 'geometry'] = geom_offset
    
    df_edges = pd.concat([df_edges.query("way_id not in @way_ids"), _df_edges])
    

    return df_edges


def parallel_offset_edge(record:pd.Series, distance=1.25/110/1000, process_two_point=True, keep_endpoint_pos=True, logger=None):
    """Returns a LineString or MultiLineString geometry at a distance from the object on its right or its left side

    Args:
        record (LineString): The record object should have the `geometry` attitude.
        distance (float, optional): [description]. Defaults to 2/110/1000.
        keep_endpoint_pos (bool, optional): keep hte endpoints position or not. Defaults to False.

    Returns:
        [LineString]: The offset LineString.
    """
    if 'geometry' not in record:
        return None
    geom = record.geometry
    
    if len(geom.coords) <= 1:
        if logger is not None:
            logger.warning(f"{geom}: the length of it is less than 1.")
        return geom
    
    if geom.is_ring:
        return geom

    def _cal_dxdy(p0, p1, scale = 15):
        return ((p1[0]-p0[0])/scale, (p1[1]-p0[1])/scale)

    def _point_offset(p, dxdy, add=True):
        if add:
            return (p[0]+dxdy[0], p[1]+dxdy[1])

        return (p[0]-dxdy[0], p[1]-dxdy[1])
    
    try:
        # shapely 2.0 以上，`[::-1]` 需删除
        offset_coords = geom.parallel_offset(distance, side='right').coords
        if int(shapely.__version__.split('.')[0]) < 2:
            offset_coords = offset_coords[::-1]

        ori_s, ori_e = geom.coords[0], geom.coords[-1]
        dxdy_s = _cal_dxdy(*geom.coords[:2])
        dxdy_e = _cal_dxdy(*geom.coords[-2:])
        turing_s =  _point_offset(offset_coords[0], dxdy_s, add=True )
        turing_e =  _point_offset(offset_coords[-1], dxdy_e, add=False )
        
        coords = [ori_s] + [turing_s] + offset_coords[1:-1] + [turing_e] + [ori_e]
        coords = np.round(coords, 7)
        geom_new = LineString(coords)
        
        if logger is not None:
            logger.info(f"{len(geom.coords)},{len(geom_new.coords)}\n{geom}\n{geom_new}")
        
        return geom_new
    except:
        if logger is not None:
            logger.error(f"{record.name}, geom: {geom}, offset error")    

    return geom
