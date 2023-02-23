import numba
import numpy as np
import shapely
from shapely import Point, LineString
import geopandas as gpd
from geopandas import GeoDataFrame

from .distance import cal_coords_seq_distance, geoseries_distance


@numba.jit
def get_first_index(arr, val):
    """有效地返回数组中第一个值满足条件的索引
    Refs: https://blog.csdn.net/weixin_39707612/article/details/111457329;
    耗时： 0.279 us; np.argmax(arr> vak)[0] 1.91 us

    Args:
        A (np.array): Numpy arr
        k (float): value

    Returns:
        int: The first index that large that `val`
    """
    for i in range(len(arr)):
        if arr[i] >= val:
            return i + 1
        val -= arr[i]

    return -1

def project_point_2_linestring(point:Point, line:LineString, normalized:bool=True):
    dist = line.project(point, normalized)
    proj_point = line.interpolate(dist, normalized)

    return proj_point, dist

def cut_linestring(line:LineString, offset:float, point:Point=None, normalized=False):
    _len = 1 if normalized else line.length
    coords = np.array(line.coords)

    if offset <= 0:
        res = {"seg_0": None, "seg_1": coords}
    elif offset >= _len:
        res = {"seg_0": coords, "seg_1": None}
    else:
        # points = np.array([Point(*i) for i in coords])
        # dist_intervals = line.project(points, normalized)
        dist_arr, _ = cal_coords_seq_distance(coords)

        idx = get_first_index(dist_arr, offset)
        pd = np.sum(dist_arr[:idx])
        if pd == offset:
            coords_0 = coords[:idx+1]
            coords_1 = coords[idx:]
        else:
            if point is None:
                point = line.interpolate(offset, normalized)
            cp = np.array(point.coords)
            coords_0 = np.concatenate([coords[:idx], cp]) 
            coords_1 = np.concatenate([cp, coords[idx:]]) 
        
        res = {'seg_0': coords_0, 'seg_1': coords_1}

    res['seg_0'] = LineString(res['seg_0'])
    res['seg_1'] = LineString(res['seg_1'])

    return res

def test_cut_linestring(line, point):
    # test: project_point_2_linestring
    cp, dist = project_point_2_linestring(point, line)
    data = {'name': ['point', 'line', 'cp'],
            'geometry': [point, line, cp]
            }
    ax = gpd.GeoDataFrame(data).plot(column='name', alpha=.5)

    # test: cut_linestring
    seg_0, seg_1 = cut_linestring(line, dist)
    data = {'name': ['ori', 'seg_0', 'seg_1'],
            'geometry': [line, seg_0, seg_1]
            }
    gpd.GeoDataFrame(data).plot(column='name', legend=True, linestyle="--", ax=ax)

def project_points_2_linestrings(points:GeoDataFrame, lines:GeoDataFrame, 
                                 normalized:bool=True, drop_ori_geom=True, 
                                 keep_attrs:list=['eid', 'geometry'], precision=1e-7, 
                                 ll=True, cal_dist=True):
    """projects points to the nearest linestring

    Args:
        panos (GeoDataFrame | GeoSeries): Points
        paths (GeoDataFrame | GeoSeries): Edges
        keep_attrs (list, optional): _description_. Defaults to ['eid', 'geometry'].
        drop_ori_geom (bool, optional): Drop the origin point and line geometry. Defaults to True.

    Returns:
        GeoDataFrame: The GeoDataFrame of projected points with `proj_point`, `offset`
        
    Example:
        ```
        import geopandas as gpd
        from shapely import Point, LineString

        points = gpd.GeoDataFrame(
            geometry=[
                Point(113.93195659801206, 22.575930582940785),
                Point(113.93251505775076, 22.57563203614608),
                Point(113.93292030671412, 22.575490522559665),
                Point(113.93378178962489, 22.57534631453745)
            ]
        )

        lines = gpd.GeoDataFrame({
            "eid": [63048, 63935],
            "geometry": [
                LineString([(113.9319709, 22.5759509), (113.9320297, 22.5759095), (113.9321652, 22.5758192), (113.9323286, 22.575721), (113.9324839, 22.5756433), (113.9326791, 22.5755563), (113.9328524, 22.5754945), (113.9330122, 22.5754474), (113.933172, 22.5754073), (113.9333692, 22.5753782), (113.9334468, 22.5753503), (113.9335752, 22.5753413), (113.9336504, 22.5753383)]),
                LineString([(113.9336504, 22.5753383), (113.9336933, 22.5753314), (113.9337329, 22.5753215), (113.9337624, 22.5753098), (113.933763, 22.5753095)])]
        })

        prod_ps = project_points_2_linestrings(points.geometry, lines)
        _, ax = plot_geodata(prod_ps, color='red', label='proj', marker='*')
        lines.plot(ax=ax, label='lines')
        points.plot(ax=ax, label='points', alpha=.5)
        ax.legend()
        ```
    """
    proj_df = points.geometry.apply(lambda x: lines.loc[lines.distance(x).idxmin(), keep_attrs])\
                            .rename(columns={"geometry": 'edge_geom'})

    att_lst = ['proj_point', 'offset']
    proj_df.loc[:, 'point_geom'] = points.geometry
    proj_df.loc[:, att_lst] = proj_df.apply(
        lambda x: project_point_2_linestring(
            x.point_geom, x.edge_geom, normalized, precision), 
        axis=1, result_type='expand'
    ).values

    proj_df.loc[:, 'dist_p2c'] = geoseries_distance(proj_df['query_geom'], proj_df['proj_point'])

    if drop_ori_geom:
        proj_df.drop(columns=['point_geom', 'edge_geom'], inplace=True)

    return gpd.GeoDataFrame(proj_df).set_geometry('proj_point')


""" decrapted """
def get_foot_point(point, line_p1, line_p2):
    """
    @point, line_p1, line_p2 : [x, y, z]
    """
    x0 = point[0]
    y0 = point[1]
    # z0 = point[2]

    x1 = line_p1[0]
    y1 = line_p1[1]
    # z1 = line_p1[2]

    x2 = line_p2[0]
    y2 = line_p2[1]
    # z2 = line_p2[2]
    assert not (x1 == x2 and y1 == y2), f"check line {line_p1}, {line_p2}"
    # k = -((x1 - x0) * (x2 - x1) + (y1 - y0) * (y2 - y1) + (z1 - z0) * (z2 - z1)) / \
    #     ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)*1.0
    k = -((x1 - x0) * (x2 - x1) + (y1 - y0) * (y2 - y1)) / ((x2 - x1) ** 2 + (y2 - y1) ** 2 )*1.0
    xn = k * (x2 - x1) + x1
    yn = k * (y2 - y1) + y1
    # zn = k * (z2 - z1) + z1

    return (round(xn, 6), round(yn, 6))

def relation_bet_point_and_line( point, line ):
    """Judge the realtion between point and the line, there are three situation:
    1) the foot point is on the line, the value is in [0,1]; 
    2) the foot point is on the extension line of segment AB, near the starting point, the value < 0; 
    3) the foot point is on the extension line of segment AB, near the ending point, the value >1; 

    Args:
        point ([double, double]): point corrdination
        line ([x0, y0, x1, y1]): line coordiantions

    Returns:
        [float]: the realtion between point and the line (起点 < 0 <= 线段中 <= 1 < 终点)
    """
    pqx = line[2] - line[0]
    pqy = line[3] - line[1]
    dx  = point[0]- line[0]
    dy  = point[1]- line[1]
    
    d = pow(pqx, 2) + pow(pqy, 2) 
    t = pqx * dx + pqy * dy

    flag = 1
    if(d > 0): 
        t = t / d
        flag = t

    return flag

def cal_foot_point_on_polyline( point: Point, line: LineString, foot=True, ratio_thres=.0):
    """caculate the foot point is on the line or not

    Args:
        point (list): coordination (x, y)
        line (pd.Series): [description]
        ratio_thres (float, optional): [ratio threshold]. Defaults to 0.005.

    Returns:
        [bool]: locate on the lane or not
    """
    line_ = line.coords[0] + line.coords[-1]
    factor = relation_bet_point_and_line((point.x, point.y), line_)
    flag = 0 - ratio_thres <= factor <= 1 + ratio_thres

    if foot:
        _foot = get_foot_point((point.x, point.y), line.coords[0], line.coords[-1])
        return {'flag': factor, 'foot':_foot}
    
    return flag


if __name__ == "__main__":
    line = LineString([(0, 0), (0, 1), (1, 1)])

    test_cut_linestring(line, Point((0.5, 0)))
    test_cut_linestring(line, Point((0, 1)))
    test_cut_linestring(line, Point((1.1, 1.5)))


