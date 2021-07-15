import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from haversine import haversine, Unit


def gdf_to_geojson(gdf, fn):
    if 'geojson' not in fn:
        fn = f'{fn}.geojson'
    
    gdf.to_file(fn, driver="GeoJSON")

    return 


def df_query(df, key, value):
    return df.query( f"{key} == @value" )


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


def coords_pair_dist(o:Point, d:Point):
    return haversine((o.y, o.x), (d.y, d.x), unit=Unit.METERS)


