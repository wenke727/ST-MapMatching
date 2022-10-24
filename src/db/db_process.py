#%%
import os, sys
import geopandas as gpd
import pandas as pd
from sqlalchemy import create_engine
from shapely.geometry import Point, box

sys.path.append("..") 
# from setting import PANO_FOLFER, postgre_config

# ENGINE   = create_engine(postgre_config['connect'])
IP = "192.168.135.16"
connect= f"postgresql://postgres:pcl_A5A@{IP}:5432/gis"
ENGINE   = create_engine(connect)

#%%

"""" io """
def gdf_to_postgis(gdf, name, engine=ENGINE, if_exists='replace', *args, **kwargs):
    """Save the GeoDataFrame to the db

    Args:
        gdf ([type]): [description]
        name ([type]): [description]
        engine ([type], optional): [description]. Defaults to ENGINE.
        if_exists (str, optional): [description]. Defaults to 'replace'. if_exists{‘fail’, ‘replace’, ‘append’}

    Returns:
        [type]: [description]
    """
    
    # try:
    #     gdf.to_postgis( name=name, con=engine, if_exists=if_exists )
    #     return True
    # except:
    #     print('gdf_to_postgis error!')
    
    # return False

    gdf.to_postgis( name=name, con=engine, if_exists=if_exists )


def gdf_to_geojson(gdf, fn):
    if not isinstance(gdf, gpd.GeoDataFrame):
        print('Check the format of the gdf.')
        return False

    if 'geojson' not in fn:
        fn = f'{fn}.geojson'
    
    gdf.to_file(fn, driver="GeoJSON")

    return 

