import numpy as np
from haversine import haversine_vector, Unit


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


if __name__ == "__main__":
    matrix = cal_haversine_matrix(traj_points, points_, xy=True)
