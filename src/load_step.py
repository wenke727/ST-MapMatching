# %%
import sys
import geopandas as gpd
from shapely.geometry import Point

sys.path.append('/home/pcl/traffic/GBA_tranportation/src/utils')
from classes import PathSet


def split_line_to_points(linestring):
    return gpd.GeoDataFrame({ 'geometry': [Point(*i) for i in linestring.coords[:]]})

path_set = PathSet(cache_folder='/home/pcl/traffic/GBA_tranportation/cache', file_name='wkt_step')
df_path  = path_set.convert_to_gdf()
shenzhen_boundary = gpd.read_file('../input/ShenzhenBoundary_wgs_citylevel.geojson')

df_path = gpd.sjoin(df_path, shenzhen_boundary, op='within')


if __name__ == '__main__':
    traj = split_line_to_points(df_path.iloc[16].geometry)
    traj = traj[traj.index%12==0].head(3).reset_index(drop=True)

