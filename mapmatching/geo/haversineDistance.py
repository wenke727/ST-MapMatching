import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from haversine import haversine, haversine_vector, Unit


def haversine_matrix(array1, array2, xy=True, unit=Unit.METERS):
    '''
    The exact same function as "haversine", except that this
    version replaces math functions with numpy functions.
    This may make it slightly slower for computing the haversine
    distance between two points, but is much faster for computing
    the distance matrix between two vectors of points due to vectorization.
    '''
    if xy:
        array1 = array1[:, ::-1]
        array2 = array2[:, ::-1]
    
    dist = haversine_vector(np.repeat(array1, len(array2), axis=0), 
                            np.concatenate([array2] * len(array1)),
                            unit=unit)

    matrix = dist.reshape((len(array1), len(array2)))

    return matrix

def haversine_vector_xy(array1, array2, unit=Unit.METERS, comb=False, normalize=False):
    # ensure arrays are numpy ndarrays
    if not isinstance(array1, np.ndarray):
        array1 = np.array(array1)
    if not isinstance(array2, np.ndarray):
        array2 = np.array(array2)

    array1 = array1[:, ::-1]
    array2 = array2[:, ::-1]
    ans = haversine_vector(array1, array2, unit, comb, normalize)
    
    return ans

def coords_pair_dist(o, d, xy=True):
    if isinstance(o, Point) and isinstance(d, Point):
        return haversine((o.y, o.x), (d.y, d.x), unit=Unit.METERS)
    
    if (isinstance(o, tuple) and isinstance(d, tuple)) or \
       (isinstance(o, list) and isinstance(d, list)):
        if xy:
            return haversine(o[:2][::-1], d[:2][::-1], unit=Unit.METERS)
        else:
            return haversine(o[:2], d[:2], unit=Unit.METERS)
    
    return np.inf

def points_geoseries_2_ndarray(geoms:gpd.GeoSeries):
    return np.concatenate(geoms.apply(lambda x: x.coords[:]).values).reshape(-1, 2)

def cal_coords_seq_distance(points:np.ndarray, xy=True):
    if xy:
        points = points.copy()
        points = points[:, ::-1]
    
    dist_np = haversine_vector(points[:-1], points[1:], unit=Unit.METERS)
    
    return dist_np, dist_np.sum()

def cal_points_geom_seq_distacne(geoms:gpd.GeoSeries):
    coords = points_geoseries_2_ndarray(geoms)
    dist, total = cal_coords_seq_distance(coords, xy=True)

    return dist, coords

def haversine_geoseries(points1, points2, unit=Unit.METERS, comb=False, normalize=False):
    coords_0 = points_geoseries_2_ndarray(points1)
    coords_1 = points_geoseries_2_ndarray(points2)
    dist = haversine_vector_xy(coords_0, coords_1, unit, comb, normalize)

    return dist


if __name__ == "__main__":
    matrix = haversine_matrix(traj_points, points_, xy=True)
