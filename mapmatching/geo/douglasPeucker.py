import numpy as np
import geopandas as gpd
from .pointToLine import get_vertical_dist


def dp_compress(point_list, dist_thres=8, verbose=False):
    """Douglas-Peucker compress alg Douglas-Peucker.

    Args:
        point_list (lst): The ordered coordinations [(x1, y1, id1), (x2, y2, id2), ... , (xn, yn, idn)]
        dist_max (int, optional): The max distance (Unit: meters). Defaults to 8.
        verbose (bool, optional): [description]. Defaults to False.
    """
    def _dfs(point_list, start, end, res, dist_max):
        # start, end = 0, len(point_list)-1
        if start >= end:
            return
        
        res.append(point_list[start])
        res.append(point_list[end])

        if start < end:
            index = start + 1
            max_vertical_dist = 0
            key_point_index = 0

            while(index < end):
                cur_vertical_dist = get_vertical_dist(
                    point_list[index][:2], point_list[start][:2], point_list[end][:2])
                if cur_vertical_dist > max_vertical_dist:
                    max_vertical_dist = cur_vertical_dist
                    key_point_index = index
                index += 1

            if max_vertical_dist >= dist_max:
                _dfs(point_list, start, key_point_index, res, dist_max)
                _dfs(point_list, key_point_index, end, res, dist_max)

    res = []
    _dfs(point_list, 0, len(point_list)-1, res, dist_thres)

    res = list(set(res))
    res = sorted(res, key=lambda x:x[2])
    
    if verbose:
        print(f"Compression rate {len(res)/len(point_list)*100:.2f}% (={len(point_list)}/{len(res)}), "\
              f"mean error: {get_MeanErr(point_list,res):.2f}")
    
    return res


def get_MeanErr(point_list, output_point_list):
    Err=0
    start, end = 0, len(output_point_list)-1

    while(start < end):
        pointA_id = int(output_point_list[start][2])
        pointB_id = int(output_point_list[start+1][2])

        id = pointA_id + 1
        while(id < pointB_id):
            Err += get_vertical_dist(output_point_list[start][:2], output_point_list[start+1][:2], point_list[id][:2])
            id += 1
        start += 1

    return Err/len(point_list)


def dp_compress_for_points(df, dist_thres=10, verbose=False, reset_index=True):
    traj = df.copy()
    traj.loc[:, 'pid_order'] = traj.index
    point_lst = traj.apply(lambda x: (x.geometry.x, x.geometry.y, x.pid_order), axis=1).values.tolist()
    point_lst = dp_compress(point_lst, dist_thres, verbose)

    if reset_index:
        return traj.loc[[ i[2] for i in point_lst]].reset_index()
    
    return traj.loc[[ i[2] for i in point_lst]]


def simplify_trajetory_points(points: gpd.GeoDataFrame, tolerance: int = None, inplace=False, logger=None):
    """The algorithm (Douglas-Peucker) recursively splits the original line into smaller parts 
    and connects these parts’ endpoints by a straight line. Then, it removes all points whose 
    distance to the straight line is smaller than tolerance. It does not move any points and 
    it always preserves endpoints of the original line or polygon.

    Args:
        points (gpd.GeoDataFrame): _description_
        traj_thres (int, optional): The compression threshold(Unit: meter). Defaults to None.
        inplace (bool, optional): _description_. Defaults to False.

    Returns:
        gpd.GeoDataFrame: _description_
    """
    ori_size = points.shape[0]
    if ori_size == 1:
        return points

    points = points if inplace else points.copy()
    points = dp_compress_for_points(points, dist_thres=tolerance)

    if ori_size == 2:
        if points.iloc[0].geometry.distance(points.iloc[1].geometry) < 1e-6:
            points = points.head(1)
            if logger:
                logger.info(
                    f"Trajectory only has one point or all the same points.")
            return points

    if logger:
        logger.debug(
            f"Trajectory compression rate: {points.shape[0]/ori_size*100:.1f}% ({ori_size} -> {points.shape[0]})")

    return points


if __name__ == '__main__':
    point_list = []
    output_point_list = []

    fd=open(r"./Dguiji.txt",'r')
    for line in fd:
        line=line.strip()
        id=int(line.split(",")[0])
        longitude=float(line.split(",")[1])
        latitude=float(line.split(",")[2])
        point_list.append((longitude,latitude,id))
    fd.close()

    output_point_list = dp_compress(point_list, dist_thres=8, verbose=True)

    import geopandas as gpd
    traj = gpd.read_file("../traj_for_compress.geojson")
    dp_compress_for_points(traj, 8, True)
