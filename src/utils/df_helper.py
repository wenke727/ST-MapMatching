import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from haversine import haversine, Unit



def df_query(df, key, value):
    return df.query( f"{key} == @value" )



