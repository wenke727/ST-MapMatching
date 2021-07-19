
from shapely.geometry import Point, LineString

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

