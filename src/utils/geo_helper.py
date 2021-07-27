import pandas as pd
import numpy as np
import geopandas as gpd
from haversine import haversine, Unit
from shapely.geometry import Point, LineString, box


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


def get_vertical_dist(pointX, pointA,pointB):
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


""" io helper """
import geopandas as gpd
from sqlalchemy import create_engine
ENGINE = create_engine("postgresql://postgres:pcl@A5A@192.168.135.15:5432/road_network")


def load_postgis(table, bbox=None, geom_wkt=None, engine=ENGINE):
    if bbox is not None:
       geom_wkt = box(*bbox).to_wkt()
     
    if geom_wkt is None:
        sql = 'select * from {table}'
    else:        
        sql = f"""select * from {table} where ST_Within( geometry, ST_GeomFromText('{geom_wkt}', 4326) )"""
        
    df = gpd.read_postgis( sql, geom_col='geometry', con=engine )
    # shenzhen_boundary = gpd.read_file('../input/ShenzhenBoundary_wgs_citylevel.geojson')
    # return gpd.sjoin(df, shenzhen_boundary, op='within')

    return df


def gdf_to_geojson(gdf, fn):
    if 'geojson' not in fn:
        fn = f'{fn}.geojson'
    
    gdf.to_file(fn, driver="GeoJSON")

    return 


def gdf_to_postgis(gdf, name, engine=ENGINE, if_exists='replace', *args, **kwargs):
    try:
        gdf.to_postgis( name=name, con=engine, if_exists=if_exists )
        return True
    except:
        print('gdf_to_postgis error!')
    
    return False

