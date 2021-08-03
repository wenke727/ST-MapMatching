import math
import numpy as np
from shapely import wkt
from shapely.geometry import Point, LineString


def azimuth_diff(a, b):
    """calcaluate the angle diff between two azimuth

    Args:
        a ([type]): [description]
        b ([type]): [description]

    Returns:
        [type]: [description]
    """
    diff = abs(a-b)

    if diff > 180:
        diff = 360-diff

    return diff


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
    angle = 0.0;
    dx, dy = x2 - x1, y2 - y1

    if x2 == x1:
        angle = math.pi / 2.0
        if y2 == y1 :
            angle = 0.0
        elif y2 < y1 :
            angle = 3.0 * math.pi / 2.0
    elif x2 > x1 and y2 > y1:
        angle = math.atan(dx / dy)
    elif x2 > x1 and y2 < y1 :
        angle = math.pi / 2 + math.atan(-dy / dx)
    elif x2 < x1 and y2 < y1 :
        angle = math.pi + math.atan(dx / dy)
    elif x2 < x1 and y2 > y1 :
        angle = 3.0 * math.pi / 2.0 + math.atan(dy / -dx)

    return (angle * 180 / math.pi)


def cal_polyline_azimuth(geom):
    """caculate the azimuth of a polyline.

    Args:
        geom (LineString): The polyline geometry.

    Returns:
        [list]: The list of azimuth.
    """
    if isinstance(geom, LineString):
        coords = geom.coords[:]
    if isinstance(geom, list):
        coords = geom
    seg_angels = [azimuthAngle( *coords[i], *coords[i+1] ) for i in range(len(coords)-1) ]

    return seg_angels


def cal_points_azimuth(geoms:list):
    """caculate the azimuth of a trajectory.

    Args:
        geom (LineString): The polyline geometry.

    Returns:
        [list]: The list of azimuth.
    """
    if not geoms or not geoms[0]:
        return None
    if not isinstance( geoms[0], Point):
        return None
    
    coords = [ g.coords[0] for g in geoms ]
    seg_angels = [azimuthAngle( *coords[i], *coords[i+1] ) for i in range(len(coords)-1) ]
    
    return seg_angels


def azimuth_cos_similarity(road_angels, head_azimuth):
    # Ref: https://www.cnblogs.com/bymo/p/8489037.html
    val = np.mean(
            np.cos(
                [(azimuth_diff(i, head_azimuth) * math.pi/180) for i in road_angels ]
            )
        )
        
    return val


if __name__ == '__main__':
    p0 = wkt.loads('POINT (113.934151 22.577512)')
    p1 = wkt.loads('POINT (113.934144 22.577979)')
    # net.df_edges.loc[82190].geometry
    polyline = wkt.loads('LINESTRING (113.9340705 22.577737, 113.9340788 22.5777828, 113.934093 22.5778236, 113.9341161 22.5778661, 113.934144 22.5779051, 113.934186 22.57795, 113.9342268 22.5779823, 113.9342743 22.5780131, 113.9343212 22.5780352, 113.9343734 22.5780515, 113.9344212 22.5780605, 113.9344796 22.5780669)')

    angels =azimuthAngle(*p0.coords[0], *p1.coords[0])

    road_angels = cal_polyline_azimuth(polyline)
    head_azimuth = cal_points_azimuth([p0, p1])

    azimuth_cos_similarity(road_angels, head_azimuth[0])
    