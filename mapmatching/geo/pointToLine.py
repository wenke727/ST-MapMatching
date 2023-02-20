import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString
from haversine import haversine, haversine_vector, Unit
from .ops.distance import cal_coords_seq_distance


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


def project_point_to_polyline(point:Point, polyline:LineString, plot=False, ll=False):
    """Find the shortest distance and foot point from a point to a line segment, and split the segment into two part and update some attributes.
    
    Judge the realtion between point and the line, there are three situation:
    1) the foot point is on the line, the value is in [0,1]; 
    2) the foot point is on the extension line of segment AB, near the starting point, the value < 0; 
    3) the foot point is on the extension line of segment AB, near the ending point, the value >1; 

    Args:
        point (Point): Point
        polyline (LineString): Polyline

    Returns:
        dict: the realted attribuites between point and the line (src < 0 <= imme <= 1 < dst)

    Example: 
        # case 0
        point = Point([-0.5, 1])
        # case 1
        point = Point([-1.5, .5])
        # case 2
        point = Point([2.2, 3.5])
        polyline = LineString([[-1,0], [0, 0], [1,1], [1,1], [2,3]])

        info = project_point_to_polyline(point, polyline, True)
    """
    
    def _cal_dist(point, lines, foots, factors):
        dxdy = [point.x, point.y] - foots

        con1  = factors < 0 
        dxdy1 = [point.x, point.y] - lines[:, 0, :]
        dxdy[con1] = dxdy1[con1]

        con2  = factors > 1 
        dxdy2 = [point.x, point.y] - lines[:, 1, :]
        dxdy[con2] = dxdy2[con2]

        dists = np.linalg.norm(dxdy, axis=1)

        return dists
    
    coords = np.array(polyline.coords[:])
    lines  = np.hstack((coords[:-1][:, np.newaxis], coords[1:][:, np.newaxis]))

    dx, dy = point.x - lines[:, 0, 0], point.y - lines[:, 0, 1]
    pqx, pqy = lines[:, 1, 0] - lines[:, 0, 0], lines[:, 1, 1] - lines[:, 0, 1]
    d = np.power(pqx, 2) + np.power(pqy, 2)
    t = pqx * dx + pqy * dy
    factors = t / d
    factors[d==0] = 0 # fixme the situtaion of `d` is 0 
    
    foots = np.vstack([factors * pqx + lines[:, 0, 0], 
                       factors * pqy + lines[:, 0, 1]]).T.round(7)
    if ll:
        line_length = haversine_vector(coords[:-1, ::-1], coords[1:, ::-1], unit=Unit.METERS)  
    else:
        line_length = np.sqrt(d)
    
    dist_lst = _cal_dist(point, lines, foots, factors)
    split_idx = np.argmin(dist_lst)

    if split_idx == 0 and factors[0] <= 0:
        seg_0, seg_1 = None, coords
        len_0, len_1 = 0, sum(line_length)
    elif split_idx == len(lines) - 1 and factors[-1] >= 1:
        seg_0, seg_1 = coords, None
        len_0, len_1 = sum(line_length), 0
    else:
        seg_0 = np.vstack([coords[:split_idx+1, :], foots[split_idx]])
        seg_1 = np.vstack([foots[split_idx], coords[split_idx+1:, :]])
        if ll:
            len_0 = sum(line_length[:split_idx]) + haversine(seg_0[-2][::-1], seg_0[-1][::-1], unit=Unit.METERS)
            len_1 = sum(line_length[split_idx+1:]) + haversine(seg_1[0][::-1], seg_1[1][::-1], unit=Unit.METERS)
        else:
            len_0 = sum(line_length[:split_idx]) + np.linalg.norm(seg_0[-2] - seg_0[-1])
            len_1 = sum(line_length[split_idx+1:]) + np.linalg.norm(seg_1[0] - seg_1[1])
            
    if plot:
        ax = gpd.GeoDataFrame({
            'geometry':[point, polyline],
            'name': ['seg_0', 'seg_1']
        }).plot(color='red')
        
        gpd.GeoDataFrame({
            'geometry':[LineString(seg_0) if seg_0 is not None else None, 
                        LineString(seg_1) if seg_1 is not None else None],
            'name': ['seg_0', 'seg_1']
        }).plot(legend=True, column='name', ax=ax)

    if factors[split_idx] <= 0:
        projection = coords[0]
    elif factors[split_idx] < 1:
        projection = foots[split_idx]
    else:
        projection = coords[-1]

    res = {
        'factors': factors,
        'foots': foots,
        'dist_lst': dist_lst, 
        'line_length': line_length,
        'split_idx': split_idx,
        'projection': foots[split_idx], # projection point
        'distance': dist_lst[split_idx], # dist bet `point` to `projection point`
        'seg_0': seg_0,
        'seg_1': seg_1,
        'len_0': len_0,
        'len_1': len_1
    }

    return res

def point_to_polyline_process_wgs(point:Point, polyline:LineString, in_crs:int=4326, out_crs:int=900913, plot:bool=False):
    """Find the shortest distance and foot point from a point to a line segment, and split the segment into two part and update some attributes.

    Args:
        point (Point): Point.
        polyline (LineString): Polyline
        in_crs (int, optional): Input coordinate system. Defaults to 4326.
        out_crs (int, optional): Output coordinate system. Defaults to 900913.
        plot (Boolean, optional): Verbose. Defaults to False.

    Returns:
        dict: lines and attibutes.
    
    Example: 
        ```
        from shapely import wkt
        # case 0
        node = wkt.loads('POINT (113.934194 22.577979)')
        # case 1
        node = wkt.loads('POINT (113.934144 22.577979)')

        # case 0, 创科路/打石二路路口
        polyline = wkt.loads("LINESTRING (113.934186 22.57795, 113.934227 22.577982, 113.934274 22.578013, 113.934321 22.578035, 113.934373 22.578052, 113.934421 22.57806, 113.93448 22.578067)")
        point_to_polyline_process_wgs(node, polyline, plot=True)
        ```
    """
    tmp = gpd.GeoSeries([point, polyline]).set_crs(in_crs, allow_override=True).to_crs(out_crs)
    p_, l_ = tmp.loc[0], tmp.loc[1]
    
    res = project_point_to_polyline(p_, l_, ll=True, plot=plot)
    
    helper = lambda x: LineString(x) if x is not None else LineString([])
    tmp = gpd.GeoSeries([helper(res['seg_0']), helper(res['seg_1'])]).set_crs(out_crs, allow_override=True).to_crs(in_crs)
    res['seg_0'] = tmp.loc[0].coords[:]
    res['seg_1'] = tmp.loc[1].coords[:]
    
    return res

def project_point_2_linestring(node:Point, polyline:LineString, plot=False):
    """Linear referencing

    Args:
        node (Point): _description_
        polyline (LineString): _description_
        plot (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    
    Refs:
        https://shapely.readthedocs.io/en/stable/manual.html#linear-referencing-methods
    """
    dist = polyline.project(node)
    cut_point = polyline.interpolate(dist)
    seg_0, seg_1 = cut(polyline, dist)

    # FIXME the distance along this geometric object to a point nearest the other object.
    coords = np.array(polyline.coords)
    len, total_len = cal_coords_seq_distance(coords)
    normalized_dist = dist / polyline.length

    len_0 = total_len * normalized_dist
    len_1 = total_len - len_0

    if plot:
        ax = gpd.GeoDataFrame({
            'geometry':[node, polyline],
            'name': ['seg_0', 'seg_1']
        }).plot(color='red', alpha=.2)

        gpd.GeoDataFrame({
            'geometry':[cut_point],
        }).plot(ax=ax, color='b')

        gpd.GeoDataFrame({
            'geometry':[LineString(seg_0) if seg_0 is not None else None, 
                        LineString(seg_1) if seg_1 is not None else None],
            'name': ['seg_0', 'seg_1']
        }).plot(legend=True, column='name', ax=ax, zorder=9, linestyle='--')
    
    return seg_0, seg_1, len_0, len_1

def cut(line, distance):
    # Cuts a line in two at a distance from its starting point
    if distance <= 0.0:
        return [None, LineString(line)]
    if distance >= line.length:
        return [LineString(line), None]
    
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.project(Point(p))
        if pd == distance:
            return [
                LineString(coords[:i+1]),
                LineString(coords[i:])]
        if pd > distance:
            cp = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:])]

if __name__ == "__main__":
    
    from shapely import wkt
    # case 0
    node = wkt.loads('POINT (113.934194 22.577979)')
    # case 1
    # node = wkt.loads('POINT (113.934144 22.577979)')

    # case 0, 创科路/打石二路路口
    polyline = wkt.loads("LINESTRING (113.934186 22.57795, 113.934227 22.577982, 113.934274 22.578013, 113.934321 22.578035, 113.934373 22.578052, 113.934421 22.57806, 113.93448 22.578067)")
    # point_to_polyline_process_wgs(node, polyline, plot=True)

    # 63 us
    seg_0, seg_1, len_0, len_1 = project_point_2_linestring(node, polyline, True)

    # 180 us
    project_point_to_polyline(node, polyline, plot=False)
