import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from haversine import haversine, haversine_vector, Unit

from .to_array import points_geoseries_2_ndarray


def get_length(geoms):
    crs = geoms.estimate_utm_crs()
    return geoms.to_crs(crs).length

def geoseries_distance(arr1, arr2, align=True):
    """calculate two geoseries distance

    Args:
        arr1 (gpd.GeoSeries): Geom array 1.
        arr2 (gpd.GeoSeries): Geom array 2.
        align (bool, optional): Align the two  Geom arrays. Defaults to True.

    Returns:
        pd.Series: Distance array
    """
    if isinstance(arr1, pd.Series):
        arr1 = gpd.GeoSeries(arr1)
    if isinstance(arr2, pd.Series):
        arr2 = gpd.GeoSeries(arr2)

    crs_1 = arr1.crs
    crs_2 = arr2.crs
    assert crs_1 is None or crs_2 is None, "arr1 and arr2 must have one has crs"
    
    if align:
        if crs_1 is None:
            arr1.set_crs(crs_2, inplace=True)
        if crs_2 is None:
            arr2.set_crs(crs_1, inplace=True)
    else:
        assert crs_1 is not None and crs_2 is not None, "Turn `align` on to align geom1 and geom2"

    if arr1.crs.to_epsg() == 4326:
        crs = arr1.estimate_utm_crs()
        dist = arr1.to_crs(crs).distance(arr2.to_crs(crs))
    else:
        dist = arr1.distance(arr2)

    return dist

def coords_seq_distance(coords):
    # for matrix
    dist_np = np.linalg.norm(coords[:-1] - coords[1:], axis=1)
    
    return dist_np, np.sum(dist_np)

def get_vertical_dist(pointX, pointA, pointB, ll=False):
    if ll:
        a, b, c = haversine_vector(
            np.array([pointA, pointA, pointB])[:, ::-1],
            np.array([pointB, pointX, pointX])[:, ::-1],
            unit=Unit.METERS
        )
    else:
        a, b, c = np.linalg.norm(
            np.array([pointA, pointA, pointB]) - np.array([pointB, pointX, pointX]), axis = 1)

    #当弦两端重合时,点到弦的距离变为点间距离
    if a==0:
        return b

    p = (a + b + c) / 2
    S = np.sqrt(np.abs(p*(p-a)*(p-b)*(p-c)))
    
    vertical_dist = S * 2 / a

    return vertical_dist

""" haversine """
def geom_series_distance(col1, col2, in_crs=4326, out_crs=900913):
    assert isinstance(col1, gpd.GeoSeries) and isinstance(col2, gpd.GeoSeries)

    if in_crs == out_crs:
        return col1.distance(col2)

    if isinstance(col1, pd.Series):
        a = gpd.GeoSeries(col1).set_crs(in_crs, allow_override=True).to_crs(out_crs)
    if isinstance(col2, pd.Series):
        b = gpd.GeoSeries(col2).set_crs(in_crs, allow_override=True).to_crs(out_crs)
    
    return a.distance(b)

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

def cal_coords_seq_distance(points:np.ndarray, xy=True):
    if xy:
        points = points.copy()
        points = points[:, ::-1]
    
    # FIXME
    try:
        dist_np = haversine_vector(points[:-1], points[1:], unit=Unit.METERS)
    except:
        dist_np = np.linalg.norm(points[:-1] - points[1:], axis=1)
    
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
