import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from haversine import haversine, haversine_vector, Unit


def cal_haversine_matrix(array1, array2, xy=True, unit=Unit.METERS):
    '''
    The exact same function as "haversine", except that this
    version replaces math functions with numpy functions.
    This may make it slightly slower for computing the haversine
    distance between two points, but is much faster for computing
    the distance matrix between two vectors of points due to vectorization.
    '''
    if xy:
        array1 = array1[:, ::-1]
        array2 = array2[:,::-1]
    
    dist = haversine_vector(np.repeat(array1, len(array2), axis=0), 
                            np.concatenate([array2] * len(array1)),
                            unit=unit)

    matrix = dist.reshape((len(array1), len(array2)))

    return matrix

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

def haversine_vector_xy(array1, array2, unit=Unit.KILOMETERS, comb=False, normalize=False):
    '''
    The exact same function as "haversine", except that this
    version replaces math functions with numpy functions.
    This may make it slightly slower for computing the haversine
    distance between two points, but is much faster for computing
    the distance between two vectors of points due to vectorization.
    '''
    try:
        import numpy
    except ModuleNotFoundError:
        return 'Error, unable to import Numpy,\
        consider using haversine instead of haversine_vector.'

    # ensure arrays are numpy ndarrays
    if not isinstance(array1, numpy.ndarray):
        array1 = numpy.array(array1)
    if not isinstance(array2, numpy.ndarray):
        array2 = numpy.array(array2)

    # ensure will be able to iterate over rows by adding dimension if needed
    if array1.ndim == 1:
        array1 = numpy.expand_dims(array1, 0)
    if array2.ndim == 1:
        array2 = numpy.expand_dims(array2, 0)

    # Asserts that both arrays have same dimensions if not in combination mode
    if not comb:
        if array1.shape != array2.shape:
            raise IndexError(
                "When not in combination mode, arrays must be of same size. If mode is required, use comb=True as argument.")

    # normalize points or ensure they are proper lat/lon, i.e., in [-90, 90] and [-180, 180]
    if normalize:
        array1 = numpy.array([_normalize(p[0], p[1]) for p in array1])
        array2 = numpy.array([_normalize(p[0], p[1]) for p in array2])
    else:
        [_ensure_lat_lon(p[0], p[1]) for p in array1]
        [_ensure_lat_lon(p[0], p[1]) for p in array2]

    # unpack latitude/longitude
    lat1, lng1 = array1[:, 0], array1[:, 1]
    lat2, lng2 = array2[:, 0], array2[:, 1]

    # convert all latitudes/longitudes from decimal degrees to radians
    lat1 = numpy.radians(lat1)
    lng1 = numpy.radians(lng1)
    lat2 = numpy.radians(lat2)
    lng2 = numpy.radians(lng2)

    # If in combination mode, turn coordinates of array1 into column vectors for broadcasting
    if comb:
        lat1 = numpy.expand_dims(lat1, axis=0)
        lng1 = numpy.expand_dims(lng1, axis=0)
        lat2 = numpy.expand_dims(lat2, axis=1)
        lng2 = numpy.expand_dims(lng2, axis=1)

    # calculate haversine
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = (numpy.sin(lat * 0.5) ** 2
         + numpy.cos(lat1) * numpy.cos(lat2) * numpy.sin(lng * 0.5) ** 2)

    return 2 * get_avg_earth_radius(unit) * numpy.arcsin(numpy.sqrt(d))

def cal_points_seq_distance(points:np.ndarray, xy=True):
    if xy:
        points = points.copy()
        points = points[:, ::-1]
    
    dist_np = haversine_vector(points[:-1], points[1:], unit=Unit.METERS)
    
    return dist_np, dist_np.sum()

def cal_points_geom_seq_distacne(geoms:gpd.GeoSeries):
    coords = np.concatenate(geoms.apply(lambda x: x.coords[:])).reshape(-1, 2)
    dist, total = cal_points_seq_distance(coords, xy=True)

    return dist, total


if __name__ == "__main__":
    matrix = cal_haversine_matrix(traj_points, points_, xy=True)
