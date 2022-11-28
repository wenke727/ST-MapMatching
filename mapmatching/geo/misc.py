import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString
from haversine import haversine, haversine_vector, Unit


def geom_buffer(df:gpd.GeoDataFrame, by, buffer_dis=100, att='buffer_', crs_wgs=4326, crs_prj=900913):
    df.loc[:, att] = df.to_crs(epsg=crs_prj).buffer(buffer_dis).to_crs(epsg=crs_wgs)
    df.set_geometry(att, inplace=True)
    
    whole_geom = df.dissolve(by=by).iloc[0][att]
    
    return df, whole_geom


""" convert helper """
def geom_lst_to_gdf(lst):
    """Convert geometry or geometry list to gdf.

    Args:
        lst (geometry|list(geometry)): The geometry or geometries.

    Returns:
        gpd.GeoDataFrame: The geodataframe.
    """
    
    if not isinstance(lst, list):
        lst = [lst]
        
    return gpd.GeoDataFrame( {'geometry':lst} )


""" Linstring helper """
def linestring_length(df:gpd.GeoDataFrame, add_to_att=False, key='length'):
    """caculate the length of LineString
    @return: pd:Series, length
    """
    # """" caculate the length of road segment  """
    # DB_roads.loc[:, 'length'] = DB_roads.to_crs('epsg:3395').length
    if df.crs is None:
        df.set_crs(epsg=4326, inplace=True)
    dis =  df.to_crs('epsg:3395').length
    
    if add_to_att:
        df.loc[:, key] = dis
        return
    
    return dis


""" Distance helper """
def geom_series_distance(col1, col2, in_crs=4326, out_crs=900913):
    if isinstance(col1, pd.Series):
        a = gpd.GeoSeries(col1).set_crs(in_crs, allow_override=True).to_crs(out_crs)
    if isinstance(col2, pd.Series):
        b = gpd.GeoSeries(col2).set_crs(in_crs, allow_override=True).to_crs(out_crs)
    
    assert isinstance(a, gpd.GeoSeries) and isinstance(b, gpd.GeoSeries)
    return a.distance(b)

