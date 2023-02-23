import numpy as np
from shapely import LineString, Point

from .distance import coords_seq_distance

def _check(point, line):
    res = linear_referencing(point, polyline, cut=False)
    dist = res['offset']
    _dist = line.project(point)

    assert (dist - _dist) / (dist + 1e-8) < 1e-8, "check"

def plot(point, polyline, res):
    proj = res['proj_point']
    if 'seg_0' in res:
        seg_0 = LineString(res['seg_0'])
    if 'seg_1' in res:
        seg_1 = LineString(res['seg_1'])
    
    import geopandas as gpd
    ax = gpd.GeoDataFrame({
        'geometry':[point, polyline],
        'name': ['seg_0', 'seg_1']
    }).plot(color='red', linewidth=5, alpha=.5)
    
    gpd.GeoDataFrame({"geometry": [proj]}).plot(ax=ax, color='blue', label='Project')

    segs = gpd.GeoDataFrame({"name": ['seg_0', "seg_1"],
                             "geometry": [LineString(seg_0), LineString(seg_1)]})
    segs.plot(ax=ax, column='name', legend=True, linestyle="--")

    return ax

def closest_point_on_segments(point:np.array, lines:np.array, eps=1e-9):
    """_summary_

    Args:
        point (np.array): _description_
        lines (np.array): _description_
        eps (float, optional): _description_. Defaults to 1e-9.

    Returns:
        tuple: (proj, dist, ratio)
    """
    pq = lines[:, 1] - lines[:, 0]
    d = np.power(pq, 2).sum(axis=1)
    d[d == 0] = eps

    x, y = point
    dx = x - lines[:, 0, 0]
    dy = y - lines[:, 0, 1]
    t = pq[:, 0] * dx + pq[:, 1] * dy
    
    ratio = t / d
    ratio[ratio < 0] = 0
    ratio[ratio > 1] = 1

    offset = pq * ratio[:, np.newaxis]
    proj =  offset + lines[:, 0]
    dist = np.linalg.norm(point - proj, axis=1)

    return proj, dist, ratio

def linear_referencing(point:Point, polyline:LineString, cut=True, to_geom=False):
    # TODO vectorized
    # iterating through each segment in the polyline and returning the one with minimum distance
    NONE_COORD = None

    def cut_lines(idx, proj, ratio, coords):
        if ratio == 0:
            if idx == 0:
                seg_0, seg_1 = NONE_COORD, coords
            else:
                seg_0 = coords[:idx + 1] 
                seg_1 = coords[idx:]    
        elif ratio < 1:
            seg_0 = np.concatenate([coords[:idx+1], [proj]]) 
            seg_1 = np.concatenate([[proj], coords[idx+1:]])     
        else:
            if idx == len(dists) - 1:
                seg_0, seg_1 = coords, NONE_COORD
            else:
                seg_0 = coords[:idx+2]
                seg_1 = coords[idx+1:]
        
        return seg_0, seg_1
        
    p_coords = np.array(point.coords[0])
    coords = np.array(polyline.coords)
    l_coords = np.hstack([coords[:-1][:, np.newaxis],
                          coords[1:][:, np.newaxis]])

    projs, dists, ratios = closest_point_on_segments(p_coords, l_coords)
    idx = np.argmin(dists)
    proj = projs[idx]
    ratio = ratios[idx]
    len_np, total_len = coords_seq_distance(coords)
    offset = len_np[:idx].sum() + len_np[idx] * ratio

    res = {}
    res['proj_point'] = Point(proj) if to_geom else proj
    res['dist_p2c'] = dists[idx]
    if not cut:
        res['offset'] = offset
    else:
        seg_0, seg_1 = cut_lines(idx, proj, ratio, coords)
        res['seg_0'] = LineString(seg_0) if to_geom else seg_0
        res['seg_1'] = LineString(seg_1) if to_geom else seg_1
        res['len_0'] = offset
        res['len_1'] = total_len - offset

    return res


if __name__ == "__main__":
    polyline = LineString([[-1,0], [0, 0], [1,1], [1, 1], [2,3]])
    point = Point([-0.5, 1])
    res = linear_referencing(point, polyline)

    # case 0
    point = Point([-0.5, 1])
    _check(point, polyline)
    res = linear_referencing(point, polyline)
    plot(point, polyline, res)

    # case 1
    point = Point([-1.5, .5])
    _check(point, polyline)
    res = linear_referencing(point, polyline)
    plot(point, polyline, res)

    # case 2
    point = Point([2.2, 3.5])
    _check(point, polyline)
    res = linear_referencing(point, polyline)
    plot(point, polyline, res)

    # case 3
    point = Point([0.5, 1])
    # _check(point, polyline)
    res = linear_referencing(point, polyline)
    plot(point, polyline, res);

    # case 4
    point = Point([-.1, 1.2])
    polyline = LineString([[0, 0], [0, 1], [1,1]])
    res = linear_referencing(point, polyline)
    plot(point, polyline, res)


    from shapely import wkt
    # case 0
    point = wkt.loads('POINT (113.934194 22.577979)')
    # case 1
    point = wkt.loads('POINT (113.934144 22.577979)')

    # case 0, 创科路/打石二路路口
    polyline = wkt.loads("LINESTRING (113.934186 22.57795, 113.934227 22.577982, 113.934274 22.578013, 113.934321 22.578035, 113.934373 22.578052, 113.934421 22.57806, 113.93448 22.578067)")
    
    res = linear_referencing(point, polyline)
    plot(point, polyline, res)
    