import numpy as np
from numba import jit
import geopandas as gpd

def points_geoseries_2_ndarray(geoms:gpd.GeoSeries):
    return np.concatenate([np.array(i.coords) for i in geoms])

def geoseries_to_coords(geoms):
    return [np.array(i.coords) for i in geoms]
