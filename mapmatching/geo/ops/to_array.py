import numba
import numpy as np
import geopandas as gpd

@numba.jit
def points_geoseries_2_ndarray(geoms:gpd.GeoSeries):
    return np.concatenate([np.array(i.coords) for i in geoms])

@numba.jit
def geoseries_to_coords(geoms):
    return [np.array(i.coords) for i in geoms]
