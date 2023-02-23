import numba
import shapely
import warnings
import numpy as np
from shapely import LineString
from shapely.ops import linemerge

def merge_step_arrs(x, check=True):
    lst = [i for i in [x.step_0, x.step_1, x.step_n] if i is not None]
    if len(lst) == 0:
        warnings.warn("All geoms are None")
        return None
    
    if len(lst) == 1:
        return lst[0]

    # TODO 连接处会重复节点
    coords = np.concatenate(lst)
    
    return coords

def merge_step_geoms(x, check=True):
    lst = [i for i in [x.step_0, x.geometry, x.step_n] if not i.is_empty]
    if len(lst) == 0:
        warnings.warn("All geoms are None")
        # 存在 三者均为 empty 的情况，如 两点之间没有联通
        return LineString([])
    
    # TODO 连接处会重复节点
    coords = np.concatenate([i.coords for i in lst])
    
    return LineString(coords)

@numba.jit
def get_shared_arr(arr1:np.ndarray, arr2:np.ndarray):
    lst = [arr1[0]]
    right = 0
    left = 1    
    n, m = len(arr1), len(arr2)
    
    while left < n:
        while right < m and np.all(arr1[left] != arr2[right]):
            right += 1
        if right >= m:
            break
        lst.append(arr1[left])
        left += 1

    if np.all(arr2[-1] != lst[-1]):
        lst.append(arr2[-1])
        
    return lst

def get_shared_line(line_1:np.ndarray, line_2:np.ndarray):
    if line_1 is not None:
        # 这种情况不应发生, 因为起点的相对位置比终点的相对位置更后
        warnings.warn('line_1 is empty')
        coords = line_2
    elif line_2 is not None:
        warnings.warn('line_2 is empty')
        coords = line_1
    else:
        coords = get_shared_arr(line_1, line_2)
    
    return shapely.LineString(coords)

def get_shared_linestring(line_1:shapely.LineString, line_2:shapely.LineString):
    if line_1.is_empty:
        # 这种情况不应发生, 因为起点的相对位置比终点的相对位置更后
        warnings.warn('line_1 is empty')
        coords = line_2
    elif line_2.is_empty:
        warnings.warn('line_2 is empty')
        coords = line_1
    else:
        coords_1 = np.array(line_1.coords)
        coords_2 = np.array(line_2.coords)
        coords = get_shared_arr(coords_1, coords_2)
    
    return shapely.LineString(coords)

# deprecated
def _merge_steps(gt):
    # step_0 + x.geometry + step_n
    def get_coords(item):
        if item is None or item.is_empty == 0:
            return None
        
        return item
    
    def helper(x):
        first = get_coords(x.step_0)
        last = get_coords(x.step_n)
        if x.flag == 1:
            coords = get_shared_line(first, last)
            return LineString(coords)

        lst = []
        if first is not None:
            lst.append(first)
        if x.geometry:
            lst.append(x.geometry.coords[:])
        if last is not None:
            lst.append(last)

        if len(lst) == 0:
            return None
        
        polyline = LineString(np.concatenate(lst))
        return polyline
    
    # return gt.apply(helper, axis=1)
    
    def merger(x):
        if x.geometry.is_empty:
            res = get_shared_line(x.step_0, x.step_n)
        else:
            res = linemerge(
                filter(lambda i: not i.is_empty, 
                    [x.step_0, x.geometry, x.step_n])
            )
        return res
    
    return gt.apply(merger, axis=1)

# deprecated
def merge_coords_intervals_on_same_edge(step_0:np.ndarray, step_n:np.ndarray):
    if step_0 is None:
        # 这种情况不应发生, 因为起点的相对位置比终点的相对位置更后
        coords = step_n
    elif step_n is None:
        coords = step_0
    else:
        # 也会存在反方向的情况，但在这里先忽略不计，认为是在同一个线段上
        coords = np.concatenate((step_0[0][np.newaxis, :], 
                                step_0[[p in step_n for p in step_0]], 
                                step_n[-1][np.newaxis, :])
        )
    
    return coords

if __name__ == "__main__":
    # get_shared_line
    line_1 = LineString([[.9, .9], [1, 1], [2,2]])
    line_2 = LineString([[0,0], [1,1], [1.5, 1.5]])
    print(get_shared_line(line_1, line_2))
    
