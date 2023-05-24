import numba
import shapely
import warnings
import numpy as np
from shapely import LineString
from shapely.ops import linemerge

def merge_step_arrs(x, check=True):
    lst = [i for i in [x.step_0, x.step_1, x.step_n] if isinstance(i, (np.ndarray, list))]
    if len(lst) == 0:
        warnings.warn("All geoms are None")
        return None
    
    if len(lst) == 1:
        return lst[0]

    # TODO Nodes may be duplicated at the join
    coords = np.concatenate(lst)
    
    return coords

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
        warnings.warn('line_1 is empty')
        coords = line_2
    elif line_2 is not None:
        warnings.warn('line_2 is empty')
        coords = line_1
    else:
        coords = get_shared_arr(line_1, line_2)
    
    return coords


if __name__ == "__main__":
    # get_shared_line
    line_1 = LineString([[.9, .9], [1, 1], [2,2]])
    line_2 = LineString([[0,0], [1,1], [1.5, 1.5]])
    print(get_shared_line(line_1, line_2))
    
