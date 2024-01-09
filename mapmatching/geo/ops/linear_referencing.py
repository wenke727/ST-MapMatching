import numba
import numpy as np
from geopandas import GeoSeries
from shapely import LineString, Point
from .distance import coords_seq_distance
from .to_array import geoseries_to_coords, points_geoseries_2_ndarray

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

def closest_point_on_segments(point:np.ndarray, lines:np.ndarray, eps=1e-9):
    """Calculate the closest point p' and its params on each segments on a polyline.

    Args:
        point (np.array): Point (shape: [2,]). 
        lines (np.array): Polyline in the form of coords sequence(shape: [n, 2]).
        eps (float, optional): Defaults to 1e-9.

    Returns:
        (array, array, array): proj, dist, ratio
    """
    segs = np.hstack([lines[:-1][:, np.newaxis],
                      lines[1:][:, np.newaxis]])
    pq = segs[:, 1] - segs[:, 0]
    d = np.power(pq, 2).sum(axis=1)
    d[d == 0] = eps

    x, y = point
    dx = x - segs[:, 0, 0]
    dy = y - segs[:, 0, 1]
    t = pq[:, 0] * dx + pq[:, 1] * dy
    
    ratio = t / d
    ratio[ratio < 0] = 0
    ratio[ratio > 1] = 1

    offset = pq * ratio[:, np.newaxis]
    proj =  offset + segs[:, 0]
    dist = np.linalg.norm(point - proj, axis=1)

    return proj, dist, ratio

def linear_referencing(point:Point, polyline:LineString, cut=True, to_geom=False):
    # iterating through each segment in the polyline and returning the one with minimum distance

    p_coords = np.array(point.coords[0])
    l_coords = np.array(polyline.coords)

    projs, dists, ratios = closest_point_on_segments(p_coords, l_coords)
    idx = np.argmin(dists)
    proj = projs[idx]
    ratio = ratios[idx]
    len_np, total_len = coords_seq_distance(l_coords)
    offset = len_np[:idx].sum() + len_np[idx] * ratio

    res = {}
    res['proj_point'] = Point(proj) if to_geom else proj
    res['dist_p2c'] = dists[idx]
    if not cut:
        res['offset'] = offset
    else:
        seg_0, seg_1 = cut_linestring_coords(idx, proj, ratio, l_coords)
        if to_geom:
            seg_0 = LineString(seg_0)
            seg_1 = LineString(seg_1)
        res['seg_0'] = seg_0
        res['seg_1'] = seg_1
        res['len_0'] = offset
        res['len_1'] = total_len - offset

    return res

# @numba.jit
def lines_to_matrix(lines, n_rows, n_cols):
    _lines = np.zeros((n_rows, n_cols, 2))
    mask = np.ones((n_rows, n_cols), dtype=np.bool_)

    for i, line in enumerate(lines):
        n = len(line)
        _lines[i, :n] = line
        _lines[i, n:] = line[-1]
        mask[i, n:] = 0
    
    return _lines, mask

# @numba.jit
def cut_linestring_coords(idx, proj, ratio, coords):
    NONE_COORD = None
    if idx == 0 and ratio == 0:
        return NONE_COORD, coords
    if idx == coords.shape[0] - 2 and ratio == 1:
        return coords, NONE_COORD

    if ratio == 0:
        seg_0 = coords[:idx + 1] 
        seg_1 = coords[idx:]    
    elif ratio < 1:
        seg_0 = np.concatenate([coords[:idx+1], [proj]]) 
        seg_1 = np.concatenate([[proj], coords[idx+1:]])     
    else:
        seg_0 = coords[:idx+2]
        seg_1 = coords[idx+1:]
    
    return seg_0, seg_1

# @numba.jit
def numba_cut_lines(col_idxs, closest, ratio, lines):
    res = [cut_linestring_coords(i, c, r, s) 
            for i, c, r, s in zip(col_idxs, closest, ratio, lines)]
    
    return res

def compute_point_to_line_proximity(point_geoms: GeoSeries, line_geoms: GeoSeries, cut: bool = True, eps: float = 1e-9):
    """
    Calculate the proximity of each point in a GeoSeries to the nearest line in another GeoSeries. 
    This function computes the closest point on each linestring to each point, the distance to this closest point, 
    and optionally cuts the line at the closest point.

    Parameters:
    - point_geoms (GeoSeries): A GeoSeries containing the point geometries.
    - line_geoms (GeoSeries): A GeoSeries containing the line geometries.
    - cut (bool, optional): If True, the function will cut each line at the closest point to each point. Defaults to True.
    - eps (float, optional): A small epsilon value to prevent division by zero in distance calculations. Defaults to 1e-9.

    Returns:
    dict: A dictionary containing the following keys:
        'proj_point' (list): The closest point on each line to each point.
        'dist_p2c' (ndarray): The distance from each point to the closest point on the line.
        If 'cut' is False:
            'offset' (ndarray): The offset distance along the line to the closest point.
        If 'cut' is True:
            'seg_0' (list): The segment of each line before the closest point.
            'seg_1' (list): The segment of each line after the closest point.
            'len_0' (ndarray): The length of the line segment before the closest point.
            'len_1' (ndarray): The length of the line segment after the closest point.

    Raises:
    ValueError: If `point_geoms` or `line_geoms` is empty.

    Note:
    The function assumes that the input geometries are valid and properly formatted as required by the GeoSeries.
    """
    if point_geoms.empty or line_geoms.empty:
        return {}
    
    points = points_geoseries_2_ndarray(point_geoms)
    lines = geoseries_to_coords(line_geoms)

    n_len = [len(i) for i in lines]
    n_cols = max(n_len)
    n_rows = len(lines)
    _lines, mask = lines_to_matrix(lines, n_rows, n_cols)

    segs = np.dstack([_lines[:, :-1][:,:,np.newaxis],
                      _lines[:, 1:][:,:,np.newaxis]])
    pq = segs[:, :, 1] - segs[:, :, 0]
    # d = np.power(pq, 2).sum(axis=-1)
    d = np.einsum('ijk,ijk->ij', pq, pq)
    len_np = np.sqrt(d)
    d[d == 0] = eps

    x, y = points[:, 0], points[:, 1]
    dx = x[:, np.newaxis] - segs[:, :, 0, 0]
    dy = y[:, np.newaxis] - segs[:, :, 0, 1]
    t = pq[:, :, 0] * dx + pq[:, :, 1] * dy
    
    ratios = t / d
    ratios = np.clip(ratios, 0, 1)

    offset = pq * ratios[:, :, np.newaxis] # (n, l, 2)
    closests = offset + segs[:, :, 0]
    dists = np.linalg.norm(points[:, np.newaxis] - closests, axis=-1)

    # dist_p2c
    row_idxs = np.arange(n_rows)
    col_idxs = np.argmin(dists, axis=1)
    r = ratios[row_idxs, col_idxs]
    dist_p2c = dists[row_idxs, col_idxs]
    proj_points = closests[row_idxs, col_idxs]

    # offset
    sum_mask = np.zeros((n_rows, n_cols-1), dtype=np.bool_)
    for i, col in enumerate(col_idxs):
        sum_mask[i, :col] = True
    offsets = np.sum(len_np, axis=1, where=sum_mask) + len_np[row_idxs, col_idxs] * r
    # offset = np.add.reduceat(len_np, row_idxs, where=(row_idxs[:, None] <= col_idxs), axis=1) \
    #          + len_np[row_idxs, col_idxs] * r

    # res
    res = {
        'proj_point': list(proj_points), 
        'dist_p2c': dist_p2c
    }
    if not cut:
        res['offset'] = offsets
    else:
        # TODO normalized = True
        tmp = numba_cut_lines(col_idxs, proj_points, r, lines)
        seg_0, seg_1 = list(zip(*tmp))
        res.update({
            'seg_0': seg_0,
            'seg_1': seg_1,
            'len_0': offsets,
            'len_1': len_np.sum(axis=1) - offsets
        })
    
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
    