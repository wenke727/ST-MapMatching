import shapely
import numpy as np


def substrings(linestring:np.ndarray, start_dist:float, end_dist:float, normalized=False) -> np.ndarray:
    """Cut a linestring at two offset values

    Args:
        linestring (np.ndarray): input line
        start_dist (float): starting offset, distance to the start point of linestring
        end_dist (float): ending offset, distance to the start point of linestring
        normalized (bool, optional): If the normalized arg is True, the distance will be 
            interpreted as a fraction of the geometry’s length. Defaults to False.

    Returns:
        np.ndarray: a linestring containing only the part covering starting offset to ending offset
    
    Ref: 
        https://github.com/cyang-kth/fmm/blob/master/src/algorithm/geom_algorithm.cpp#L351-L417
        https://shapely.readthedocs.io/en/stable/manual.html?highlight=substring#shapely.ops.substring
    """

    return NotImplementedError

