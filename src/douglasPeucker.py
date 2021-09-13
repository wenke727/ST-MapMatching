import numpy as np
from utils.geo_helper import get_vertical_dist


def dp_compress(point_list, dist_max=8, verbose=False):
    """Douglas-Peucker compress alg Douglas-Peucker.

    Args:
        point_list (lst): The ordered coordinations [(x1, y1, id1), (x2, y2, id2), ... , (xn, yn, idn)]
        dist_max (int, optional): The max . Defaults to 8.
        verbose (bool, optional): [description]. Defaults to False.
    """
    def _dfs(point_list, start, end, res, dist_max=2):
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
                cur_vertical_dist = get_vertical_dist(point_list[index], point_list[start], point_list[end])
                if cur_vertical_dist > max_vertical_dist:
                    max_vertical_dist = cur_vertical_dist
                    key_point_index = index
                index += 1

            if max_vertical_dist >= dist_max:
                _dfs(point_list, start, key_point_index, res, dist_max)
                _dfs(point_list, key_point_index, end, res, dist_max)

    res = []
    _dfs(point_list, 0, len(point_list)-1, res, dist_max)

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
            Err += get_vertical_dist(output_point_list[start],output_point_list[start+1],point_list[id])
            id += 1
        start += 1

    return Err/len(point_list)


def dp_compress_for_points(df, dis_thred=10, verbose=False, reset_index=True):
    traj = df.copy()
    traj.loc[:, 'pid_order'] = traj.index
    point_lst = traj.apply(lambda x: (x.geometry.x, x.geometry.y, x.pid_order), axis=1).values.tolist()
    point_lst = dp_compress(point_lst, dis_thred, verbose)

    if reset_index:
        return traj.loc[[ i[2] for i in point_lst]].reset_index()
    
    return traj.loc[[ i[2] for i in point_lst]]


#%%

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

    output_point_list = dp_compress(point_list, dist_max=8, verbose=True)

    import geopandas as gpd
    traj = gpd.read_file("../traj_for_compress.geojson")
    dp_compress_for_points(traj, 8, True)


    # #将压缩数据写入输出文件
    # fd=open(r".\output.txt",'w')
    # for point in output_point_list:
    #     fd.write("{},{},{}\n".format(point[2],point[0],point[1]))
    # fd.close()
