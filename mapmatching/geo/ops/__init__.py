import numpy as np
import geopandas as gpd

def check_duplicate_points(points:gpd.GeoDataFrame):
    """Check for duplicate nodes in a sequence of coordinates

    Args:
        points (gpd.GeoDataFrame): _description_

    Returns:
        _type_: _description_
    """
    coords = np.concatenate(points.geometry.apply(lambda x: x.coords))
    mask = np.sum(coords[:-1] == coords[1:], axis=1) == 2
    mask = np.concatenate([mask, [False]])

    if mask.sum():
        idxs = np.where(mask == True)[0]
        print(f"Exist duplicate points, idx: {idxs}.")
    
        return points[~mask]

    return points

from .point2line import project_point_2_linestring, project_points_2_linestrings