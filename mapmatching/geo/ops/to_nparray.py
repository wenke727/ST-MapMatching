import numpy as np
import geopandas as gpd

def points_geoseries_2_ndarray(geoms:gpd.GeoSeries):
    return np.concatenate(geoms.apply(lambda x: x.coords).values)


