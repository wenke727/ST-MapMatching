import os
import sys
import shapely
import warnings
import pandas as pd
from shapely import wkt
import geopandas as gpd
from loguru import logger

try:
    import sqlalchemy
    from ..setting import postgre_url
    ENGINE = sqlalchemy.create_engine(postgre_url)
except:
    ENGINE=None


def has_table(name, con=None, engine=None):
    flag = False
    if con is None:
        con = engine.connect()
        flag = True
    
    status = con.dialect.has_table(con, name)
    if flag: 
        con.close()
    
    return status
 
def read_postgis(name, atts="*", condition=None, engine=ENGINE, bbox=None, mask=None, geom_col='geometry', *args, **kwargs):
    """
    Refs: https://geopandas.org/en/stable/docs/reference/api/geopandas.read_postgis.html#geopandas.read_postgis
    """
    with engine.connect() as conn:
        if not has_table(name, con=conn):
            warnings.warn(f"Not exist {name}")
            return None
        
        if bbox is not None:
            wkt = shapely.box(*bbox).to_wkt()
        elif shapely.is_geometry(mask):
            wkt = mask.wkt
        else:
            wkt = None
        
        if mask is None:
            sql = f"SELECT {atts} FROM {name}"
        else:
            sql = f"""SELECT {atts} FROM {name} WHERE ST_Intersects( geometry, ST_GeomFromText('{wkt}', 4326) )"""
            
        if condition:
            sql += f" WHERE {condition}" if wkt is None else f" {condition}"
        
        gdf = gpd.read_postgis(sqlalchemy.text(sql), con=conn, geom_col=geom_col, *args, **kwargs)
    
    return gdf

def to_postgis(gdf:gpd.GeoDataFrame, name, duplicates_idx=None, engine=ENGINE, if_exists='fail', *args, **kwargs):
    """
    Upload GeoDataFrame into PostGIS database.

    This method requires SQLAlchemy and GeoAlchemy2, and a PostgreSQL
    Python driver (e.g. psycopg2) to be installed.

    Parameters
    ----------
    name : str
        Name of the target table.
    con : sqlalchemy.engine.Connection or sqlalchemy.engine.Engine
        Active connection to the PostGIS database.
    if_exists : {'fail', 'replace', 'append'}, default 'fail'
        How to behave if the table already exists:

        - fail: Raise a ValueError.
        - replace: Drop the table before inserting new values.
        - append: Insert new values to the existing table.
    schema : string, optional
        Specify the schema. If None, use default schema: 'public'.
    index : bool, default False
        Write DataFrame index as a column.
        Uses *index_label* as the column name in the table.
    index_label : string or sequence, default None
        Column label for index column(s).
        If None is given (default) and index is True,
        then the index names are used.
    chunksize : int, optional
        Rows will be written in batches of this size at a time.
        By default, all rows will be written at once.
    dtype : dict of column name to SQL type, default None
        Specifying the datatype for columns.
        The keys should be the column names and the values
        should be the SQLAlchemy types.
    """

    ori_gdf = None
    flag = False
    if if_exists=='append' and duplicates_idx is not None:
        if has_table(name, engine=engine):
            ori_gdf = read_postgis(name, engine=engine)
            # FIXME 目前因为版本的原因出问题
            if_exists = 'replace'
            flag = True
    
    with engine.connect() as conn:
        if flag:
            tmp = ori_gdf.append(ori_gdf).append(gdf).drop_duplicates(duplicates_idx)
            if tmp.shape[0] == 0:
                print(f"There is no new record in {name}")
                return True

            # Check newly added att, if exist then delete it 
            drop_cols = []
            remain_cols = []
            for i in tmp.columns:
                if i not in ori_gdf.columns:
                    drop_cols.append(i)
                    continue
                remain_cols.append(i)

            if drop_cols:
                logger.warning(f"Drop column `{drop_cols}`, for not exit in the db")
        
            gdf = tmp[remain_cols]
        
        status = gdf.to_postgis(name=name, con=conn, if_exists=if_exists, *args, **kwargs)
    
    return status 

def to_geojson(gdf, fn):
    if not isinstance(gdf, gpd.GeoDataFrame):
        print('Check the format of the gdf.')
        return False

    if 'geojson' not in str(fn):
        fn = f'{fn}.geojson'
    
    gdf.to_file(fn, driver="GeoJSON")

    return 

def read_csv_to_geodataframe(file_path, crs="EPSG:4326"):
    df = pd.read_csv(file_path)
    
    df['geometry'] = df['geometry'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=crs)
    
    return gdf


def set_engine(url):
    global ENGINE
    ENGINE = sqlalchemy.create_engine(url)
    
    return ENGINE
