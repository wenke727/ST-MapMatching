import math
import numpy as np
from shapely import wkt
from haversine import haversine, haversine_vector, Unit
from shapely.geometry import Point, LineString
from .haversineDistance import coords_pair_dist


def azimuth_diff(a, b, unit='radian'):
    """calcaluate the angle diff between two azimuth, the imput unit is `degree`.
    Args:
        a (float): Unit: degree
        b (float): Unit: degree
        unit(string): `radian` or `degree`
    Returns:
        [type]: [description]
    """
    assert unit in ['degree', 'radian']
    diff = np.abs(a-b)

    if isinstance(diff, np.ndarray):
        diff[diff > 180] = 360 - diff[diff > 180]
    else:
        if diff > 180:
            diff = 360 - diff

    return diff if unit =='degree' else diff * math.pi / 180


def azimuthAngle(x1, y1, x2, y2):
    """calculate the azimuth angle from (x1, y1) to (x2, y2)

    Args:
        x1 (float): [description]
        y1 (float): [description]
        x2 (float): [description]
        y2 (float): [description]

    Returns:
        float: The angle in degree.
    """
    angle = 0.0
    dx, dy = x2 - x1, y2 - y1

    if dx == 0:
        angle = math.pi * 0
        if y2 == y1 :
            angle = 0.0
        elif y2 < y1 :
            angle = math.pi
    elif dy == 0:
        angle = 0
        if dx > 0:
            angle = math.pi / 2.0
        else:
            angle = math.pi / 2.0 * 3.0
    elif x2 > x1 and y2 > y1:
        angle = math.atan(dx / dy)
    elif x2 > x1 and y2 < y1 :
        angle = math.pi / 2 + math.atan(-dy / dx)
    elif x2 < x1 and y2 < y1 :
        angle = math.pi + math.atan(dx / dy)
    elif x2 < x1 and y2 > y1 :
        angle = 3.0 * math.pi / 2.0 + math.atan(dy / -dx)

    return angle * 180 / math.pi


def azimuthAngle_vector(x1, y1, x2, y2):
    angle = 0
    dx = x2 - x1
    dy = y2 - y1
    
    ans = np.zeros_like(dx)
    
    x_euqal = dx == 0
    x_smaller = dx < 0
    x_bigger = dx > 0
    
    y_equal = dy == 0
    y_smaller = dy < 0
    y_bigger = dy > 0    
    
    ans[x_euqal] = 0.0
    # ans[dx == 0 and dy == 0] = 0.0
    ans[x_euqal & y_smaller ] = np.pi
    
    ans[y_equal & x_bigger] = np.pi / 2.0
    ans[y_equal & x_smaller] = np.pi / 2.0 * 3.0
    
    ans[x_bigger & y_bigger] = np.arctan(dx[x_bigger & y_bigger] / dy[x_bigger & y_bigger])
    ans[x_bigger & y_smaller] = np.pi / 2.0 \
        + np.arctan(-dy[x_bigger & y_smaller] / dx[x_bigger & y_smaller])

    ans[x_smaller & y_smaller] = np.pi \
        + np.arctan(dx[x_smaller & y_smaller] / dy[x_smaller & y_smaller])
    ans[x_smaller & y_bigger] = np.pi / 2.0 * 3.0 \
        + np.arctan(dy[x_smaller & y_bigger] / -dx[x_smaller & y_bigger])

    return ans * 180 / np.pi


def azimuth_cos_similarity(angel_0:float, angel_1:float, normal=False):
    """Calculate the `cosine similarity` bewteen `angel_0` and `angel_1`.

    Args:
        angel_0 (float): Angel 0, unit degree.
        angel_1 (float): Angel 1, unit degree.
        normal (bool): Normal the cosine similarity from [-1, 1] to [0, 1].

    Returns:
        cos similarity(float): [-1, 1]
    """

    res =  np.cos(azimuth_diff(angel_0, angel_1, unit='radian'))
    if normal:
        res = (res + 1) / 2
    
    return res
    

def azimuth_cos_distance(angel_0:float, angel_1:float):
    """Calculate the `cosine distance` bewteen `angel_0` and `angel_1`.

    Args:
        angel_0 (float): Angel 0, unit degree.
        angel_1 (float): Angel 1, unit degree.

    Returns:
        cos distance(float): [0, 2]
    """

    return 1 - azimuth_cos_similarity(angel_0, angel_1)


def cal_polyline_azimuth(geom):
    """caculate the azimuth of eahc line segment in a polyline.

    Args:
        geom (LineString): The polyline geometry.

    Returns:
        [list]: The list of azimuth(unit: degree).
    """
    if isinstance(geom, LineString):
        coords = np.array(geom.coords)
    if isinstance(geom, (list, np.ndarray)):
        coords = geom

    seg_angels = azimuthAngle_vector(coords[:-1, 0], coords[:-1, 1], 
                                     coords[1:, 0], coords[1:, 1])

    return seg_angels


def cal_points_azimuth(geoms:list):
    """caculate the azimuth of a trajectory.

    Args:
        geom (LineString): The polyline geometry.

    Returns:
        [list]: The list of azimuth (unit: degree).
    """
    if not geoms or not geoms[0]:
        return None
    if not isinstance( geoms[0], Point):
        return None
    
    coords = [ g.coords[0] for g in geoms ]
    seg_angels = [azimuthAngle( *coords[i], *coords[i+1] ) for i in range(len(coords)-1) ]
    
    return seg_angels


def cal_linestring_azimuth_cos_dist(geom, head_azimuth, weight=True, offset=1):
    if geom is None:
        return None
    
    if isinstance(geom, LineString):
        coords = np.array(geom.coords)
    elif isinstance(geom, list):
        coords = np.array(geom)
    else:    
        assert False, print(type(geom), geom)
    
    road_angels = cal_polyline_azimuth(coords)

    lst = azimuth_cos_similarity(road_angels, head_azimuth)
    if offset:
        lst = (lst + 1) / 2
        
    if not weight or np.sum(weight) == 0:
        val = np.mean(lst)
    else:
        # weights = np.array([coords_pair_dist(coords[i], coords[i+1], xy=True) for i in range(len(coords)-1)]) 
        coords = coords[:, ::-1]
        weights = haversine_vector(coords[:-1], coords[1:], unit=Unit.METERS)
        val = np.average(lst, weights=weights)
    
    return val

    
if __name__ == '__main__':
    p0 = wkt.loads('POINT (113.934151 22.577512)')
    p1 = wkt.loads('POINT (113.934144 22.577979)')
    # net.df_edges.loc[82190].geometry
    polyline = wkt.loads('LINESTRING (113.9340705 22.577737, 113.9340788 22.5777828, 113.934093 22.5778236, 113.9341161 22.5778661, 113.934144 22.5779051, 113.934186 22.57795, 113.9342268 22.5779823, 113.9342743 22.5780131, 113.9343212 22.5780352, 113.9343734 22.5780515, 113.9344212 22.5780605, 113.9344796 22.5780669)')

    import matplotlib.pyplot as plt 
    import geopandas as gpd
    gpd.GeoDataFrame({'geometry': [p0, p1, polyline]}).plot()
    plt.show()

    angels = azimuthAngle(*p0.coords[0], *p1.coords[0])

    road_angels  = cal_polyline_azimuth(polyline)
    head_azimuth = cal_polyline_azimuth(LineString([p0.coords[0], p1.coords[0]]))
    
    cal_linestring_azimuth_cos_dist(LineString([p0.coords[0], p1.coords[0]]), head_azimuth, True)
    # head_azimuth = cal_points_azimuth([p0, p1])
    # head_azimuth = cal_points_azimuth([p1, p0])

    # azimuth_cos_distance(road_angels, head_azimuth[0])
    
    cal_linestring_azimuth_cos_dist(polyline, head_azimuth[0], True)
    
    cal_linestring_azimuth_cos_dist(polyline, head_azimuth[0], False)
    