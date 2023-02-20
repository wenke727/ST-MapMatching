import numpy as np
from shapely import LineString, Point


def _check(point, line):
    proj, dist, seg_0, seg_1 = linear_referencing(point, polyline)
    _dist = line.project(point)

    assert (dist - _dist) / (dist + 1e-8) < 1e-8, "check"


def plot(point, polyline, proj, seg_0=None, seg_1=None):
    import geopandas as gpd
    ax = gpd.GeoDataFrame({
        'geometry':[point, polyline],
        'name': ['seg_0', 'seg_1']
    }).plot(color='red', linewidth=5, alpha=.5)
    
    gpd.GeoDataFrame({"geometry": [proj]}).plot(ax=ax, color='blue', label='Project')

    segs = gpd.GeoDataFrame({"name": ['seg_0', "seg_1"],
                             "geometry": [LineString(seg_0), LineString(seg_1)]})
    segs.plot(ax=ax, column='name', legend=True, linestyle="--")
    print(seg_0.tolist())
    print(seg_1.tolist())
    # ax.legend()

    return ax

def coords_seq_distance(coords):
    dist_np = np.linalg.norm(coords[:-1] - coords[1:], axis=1)
    
    return dist_np, np.sum(dist_np)

def closest_point_on_segment(point:np.array, lines:np.array, eps=1e-8)-> np.array:
    pq = lines[:, 1] - lines[:, 0]
    d = np.power(pq, 2).sum(axis=1)

    x, y = point
    dx = x - lines[:, 0, 0]
    dy = y - lines[:, 0, 1]
    t = pq[:, 0] * dx + pq[:, 1] * dy
    
    ratio = t / (d + eps)
    ratio[ratio < 0] = 0
    ratio[ratio > 1] = 1

    offset = pq * ratio[:, np.newaxis]
    proj =  offset + lines[:, 0]
    dist = np.linalg.norm(point - proj, axis=1)

    return proj, dist, ratio

def linear_referencing(point:Point, polyline:LineString, prject_point=True, ll=False):
    # iterating through each segment in the polyline and returning the one with minimum distance

    p_coords = np.array(point.coords[0])
    coords = np.array(polyline.coords)
    l_coords = np.hstack([coords[:-1][:, np.newaxis],
                          coords[1:][:, np.newaxis]])

    proj_arr, dist_arr, ratio_arr = closest_point_on_segment(p_coords, l_coords)
    idx = np.argmin(dist_arr)
    proj = proj_arr[idx]
    len_np, total_len = coords_seq_distance(coords)
    offset = len_np[:idx].sum() + len_np[idx] * ratio_arr[idx]

    NONE_COORDS = np.array([])
    if ratio_arr[idx] == 0:
        if idx == 0:
            coords_0, coords_1 = NONE_COORDS, coords
        else:
            coords_0 = coords[:idx + 1] 
            coords_1 = coords[idx:]    
    elif ratio_arr[idx] < 1:
        coords_0 = np.concatenate([coords[:idx+1], [proj]]) 
        coords_1 = np.concatenate([[proj], coords[idx+1:]])     
    else:
        if idx == len(dist_arr) - 1:
            coords_0, coords_1 = coords, NONE_COORDS
        else:
            coords_0 = coords[:idx+2]
            coords_1 = coords[idx+1:]
    
    len_1, len_2 = offset, total_len - offset

    return proj, offset, coords_0, coords_1


if __name__ == "__main__":
    polyline = LineString([[-1,0], [0, 0], [1,1], [2,3]])

    # case 0
    point = Point([-0.5, 1])
    _check(point, polyline)
    proj, dist, seg_0, seg_1 = linear_referencing(point, polyline)
    plot(point, polyline, Point(proj), seg_0, seg_1)

    # case 1
    point = Point([-1.5, .5])
    _check(point, polyline)
    proj, dist, seg_0, seg_1 = linear_referencing(point, polyline)
    plot(point, polyline, Point(proj), seg_0, seg_1)

    # case 2
    point = Point([2.2, 3.5])
    _check(point, polyline)
    proj, dist, seg_0, seg_1 = linear_referencing(point, polyline)
    plot(point, polyline, Point(proj), seg_0, seg_1)

    # case 3
    point = Point([0.5, 1])
    _check(point, polyline)
    proj, dist, seg_0, seg_1 = linear_referencing(point, polyline)
    plot(point, polyline, Point(proj), seg_0, seg_1)


    # case 4
    point = Point([-.1, 1.2])
    polyline = LineString([[0, 0], [0, 1], [1,1]])
    proj, dist, seg_0, seg_1 = linear_referencing(point, polyline)
    plot(point, polyline, Point(proj), seg_0, seg_1)
