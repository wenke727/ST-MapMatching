# %%
import sys
import math
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry.linestring import LineString
from douglasPeucker import dp_compress

sys.path.append('/home/pcl/traffic/GBA_tranportation/src/utils')
from classes import PathSet


def trajectory_segmentation(points, degree_threshold=5, verbose = False):
    """线段降低维度

    Args:
        points (pd.DataFrame): Points in sequence.
        degree_threshold (int, optional): [description]. Defaults to 5.
        verbose (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    points.reset_index(drop=True)
    traj_lst = np.array(points.geometry.apply(lambda x: x.coords[0]).tolist())

    hold_index_lst = []
    previous_azimuth = 1000

    for pid, point in enumerate(traj_lst[:-1]):
        next_point = traj_lst[pid + 1]
        diff_vector = next_point - point
        azimuth = (math.degrees(math.atan2(*diff_vector)) + 360) % 360

        if abs(azimuth - previous_azimuth) > degree_threshold:
            hold_index_lst.append(pid)
            previous_azimuth = azimuth
            
    # Last point of trajectory is always added
    hold_index_lst.append(traj_lst.shape[0] - 1)

    if verbose: 
        print(hold_index_lst)
    
    return points.loc[hold_index_lst]


def split_line_to_points(linestring: LineString, compress=True, config={'dist_max': 8, 'verbose': True}):
    coords = linestring.coords[:]
    if not compress:
        return gpd.GeoDataFrame({ 'geometry': [Point(*i) for i in coords]})

    coords = [ (val[0], val[1], i) for i, val in enumerate(coords)]

    coords = dp_compress(coords, 15, True)
    coords = gpd.GeoDataFrame([{'index': i[2], 'geometry':Point(i[0], i[1])} for i in coords]).set_index('index')

    return coords


path_set = PathSet(cache_folder='/home/pcl/traffic/GBA_tranportation/cache', file_name='wkt_step')
df_path  = path_set.convert_to_gdf()
shenzhen_boundary = gpd.read_file('../input/ShenzhenBoundary_wgs_citylevel.geojson')

df_path = gpd.sjoin(df_path, shenzhen_boundary, op='within')


if __name__ == '__main__':
    path = split_line_to_points(df_path.iloc[16].geometry, compress=True)
    # path.plot()
    # traj = traj[traj.index%12==0].head(3).reset_index(drop=True)

    # trajectory_segmentation(path).shape[0]

    path

# %%
# coords = path.apply(lambda x: x.geometry.coords[0], axis=1).to_dict()
# coords = [ (val[0], val[1], key) for key, val in  coords.items()]

# res = dp_compress(coords, 15, True)

# # %%
# line = LineString( [ x[:2] for x in res] )
# ax = gpd.GeoDataFrame( {'geometry': [line]} ).plot(color='red')
# path.plot(ax=ax)

# res

