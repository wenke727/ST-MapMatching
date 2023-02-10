#%%
import numpy as np
import geopandas as gpd
from shapely import LineString
from tilemap import plot_geodata

from stmm.geo.haversineDistance import points_geoseries_2_ndarray
from stmm.geo.vis.linestring import plot_linestring_with_arrows


gdf = gpd.read_file('./data/case_5.geojson')
gdf_line = gpd.GeoDataFrame(
    geometry=[LineString(points_geoseries_2_ndarray(gdf.geometry))]
)

fig, ax = plot_geodata(gdf, reset_extent=False)
plot_linestring_with_arrows(gdf_line, ax, 'red')

# %%
