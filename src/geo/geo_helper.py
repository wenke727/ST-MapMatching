import pandas as pd
import numpy as np
import geopandas as gpd
from haversine import haversine, Unit
from shapely.geometry import Point, LineString, box
from .haversine import haversine, haversine_np, Unit


def geom_buffer(df:gpd.GeoDataFrame, by, buffer_dis=100, att='buffer_', crs_wgs=4326, crs_prj=900913):
    df.loc[:, att] = df.to_crs(epsg=crs_prj).buffer(buffer_dis).to_crs(epsg=crs_wgs)
    df.set_geometry(att, inplace=True)
    
    whole_geom = df.dissolve(by=by).iloc[0][att]
    
    return df, whole_geom


"""" Point helper """
def coords_pair_dist(o, d, xy=True):
    if isinstance(o, Point) and isinstance(d, Point):
        return haversine((o.y, o.x), (d.y, d.x), unit=Unit.METERS)
    
    if (isinstance(o, tuple) and isinstance(d, tuple)) or \
       (isinstance(o, list) and isinstance(d, list)):
        if xy:
            return haversine(o[:2][::-1], d[:2][::-1], unit=Unit.METERS)
        else:
            return haversine(o[:2], d[:2], unit=Unit.METERS)
    
    return np.inf


""" convert helper """
def geom_lst_to_gdf(lst):
    """Convert geometry or geometry list to gdf.

    Args:
        lst (geometry|list(geometry)): The geometry or geometries.

    Returns:
        gpd.GeoDataFrame: The geodataframe.
    """
    
    if not isinstance(lst, list):
        lst = [lst]
        
    return gpd.GeoDataFrame( {'geometry':lst} )


""" Point & LineString helper """
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
    
    # 线段长度的平方
    d = pow(pqx,2) + pow(pqy,2) 
    # 向量 点积 pq 向量（p相当于A点，q相当于B点，pt相当于P点）
    t = pqx*dx + pqy*dy

    flag = 1
    if(d>0): 
        t = t/d
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


def get_vertical_dist(pointX, pointA, pointB):
    a = coords_pair_dist(pointA, pointB, xy=True)

    #当弦两端重合时,点到弦的距离变为点间距离
    if a==0:
        return coords_pair_dist(pointA,pointX)

    b = coords_pair_dist(pointA, pointX, xy=True)
    c = coords_pair_dist(pointB, pointX, xy=True)
    p = (a+b+c)/2
    S = np.sqrt(np.abs(p*(p-a)*(p-b)*(p-c)))
    
    vertical_dist = S*2/a

    return vertical_dist


def point_to_polyline_process(point:Point, polyline:LineString, plot=False, coord_sys=False):
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

        info = point_to_polyline_process(point, polyline, True)
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
                       factors * pqy + lines[:, 0, 1]]).T
    line_length = haversine_np(coords[:-1], coords[1:], xy=True, unit=Unit.METERS) if coord_sys else np.sqrt(d)
    
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
        if coord_sys:
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

    res = {
        'factors': factors,
        'foots': foots,
        'dist_lst': dist_lst, 
        'line_length': line_length,
        'split_idx': split_idx,
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
    
    res = point_to_polyline_process(p_, l_, plot=plot)
    
    helper = lambda x: LineString(x) if x is not None else LineString([])
    tmp = gpd.GeoSeries([helper(res['seg_0']), helper(res['seg_1'])]).set_crs(out_crs, allow_override=True).to_crs(in_crs)
    res['seg_0'] = tmp.loc[0].coords[:]
    res['seg_1'] = tmp.loc[1].coords[:]
    
    return res


""" Linstring helper """
def edge_parallel_offset(record:pd.Series, distance=1.25/110/1000, process_two_point=True, keep_endpoint_pos=True, logger=None):
    """Returns a LineString or MultiLineString geometry at a distance from the object on its right or its left side

    Args:
        record (LineString): The record object should have the `geometry` attitude.
        distance (float, optional): [description]. Defaults to 2/110/1000.
        keep_endpoint_pos (bool, optional): keep hte endpoints position or not. Defaults to False.
        logger(logbook.Logger): Logger.

    Returns:
        [LineString]: The offset LineString.
    """
    if 'geometry' not in record:
        return None
    geom = record.geometry
    
    if len(geom.coords) <= 1:
        if logger is not None:
            logger.warning(f"{geom}: the length of it is less than 1.")
        return geom
    
    if geom.is_ring:
        return geom

    def _cal_dxdy(p0, p1, scale = 15):
        return ((p1[0]-p0[0])/scale, (p1[1]-p0[1])/scale)

    def _point_offset(p, dxdy, add=True):
        if add:
            return (p[0]+dxdy[0], p[1]+dxdy[1])

        return (p[0]-dxdy[0], p[1]-dxdy[1])
    
    try:
        offset_coords = geom.parallel_offset(distance, side='right').coords[::-1]

        ori_s, ori_e = geom.coords[0], geom.coords[-1]
        dxdy_s = _cal_dxdy(*geom.coords[:2])
        dxdy_e = _cal_dxdy(*geom.coords[-2:])
        turing_s =  _point_offset(offset_coords[0], dxdy_s, add=True )
        turing_e =  _point_offset(offset_coords[-1], dxdy_e, add=False )
        
        coords = [ori_s] + [turing_s] + offset_coords[1:-1] + [turing_e] + [ori_e]
        coords = np.round(coords, 7)
        geom_new = LineString(coords)
        
        if logger is not None:
            logger.info(f"{len(geom.coords)},{len(geom_new.coords)}\n{geom}\n{geom_new}")
        
        return geom_new
    except:
        if logger is not None:
            logger.error(f"{record.name}, geom: {geom}, offset error")    

    return geom
        

def linestring_length(df:gpd.GeoDataFrame, add_to_att=False, key='length'):
    """caculate the length of LineString
    @return: pd:Series, length
    """
    # """" caculate the length of road segment  """
    # DB_roads.loc[:, 'length'] = DB_roads.to_crs('epsg:3395').length
    if df.crs is None:
        df.set_crs(epsg=4326, inplace=True)
    dis =  df.to_crs('epsg:3395').length
    
    if add_to_att:
        df.loc[:, key] = dis
        return
    
    return dis


""" Distance helper """
def geom_series_distance(col1, col2, in_crs=4326, out_crs=900913):
    if isinstance(col1, pd.Series):
        a = gpd.GeoSeries(col1).set_crs(in_crs, allow_override=True).to_crs(out_crs)
    if isinstance(col2, pd.Series):
        b = gpd.GeoSeries(col2).set_crs(in_crs, allow_override=True).to_crs(out_crs)
    
    assert isinstance(a, gpd.GeoSeries) and isinstance(b, gpd.GeoSeries)
    return a.distance(b)

