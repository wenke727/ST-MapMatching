import numpy as np
import pandas as pd
from shapely import LineString
from geopandas import GeoDataFrame

from ..utils import timeit
from .status import CANDS_EDGE_TYPE
from ..geo.azimuth import cal_coords_seq_azimuth
from ..geo.ops.distance import coords_seq_distance
from ..geo.ops.to_array import points_geoseries_2_ndarray
from .misc import get_shared_line, merge_step_arrs


def cal_traj_params(points, move_dir=True, check=False):
    """
    Calculate trajectory parameters (e.g., euc dist, move dir) based on a series of points.

    Args:
        points (GeoSeries): A GeoSeries containing the trajectory points.
        move_dir (bool, optional): Whether to calculate the movement direction. Defaults to True.
        check (bool, optional): Whether to check for duplicate points. Defaults to False.

    Returns:
        DataFrame: A DataFrame containing the calculated trajectory parameters.

    Example:
        >>> points = gpd.GeoSeries([...])  # GeoSeries containing trajectory points
        >>> traj_params = cal_traj_params(points, move_dir=True, check=False)
        >>> print(traj_params)

    Notes:
        - The input points should be in a GeoSeries with a valid geometry column.
        - The DataFrame returned will contain columns such as 'pid_0', 'pid_1', 'euc_dist' (Euclidean distance),
          and 'move_dir' (movement direction) if move_dir=True.

    """
    coords = points_geoseries_2_ndarray(points.geometry)
    dist_arr, _ = coords_seq_distance(coords)
    idxs = points.index
    
    if check:
        zero_idxs = np.where(dist_arr==0)[0]
        if len(zero_idxs):
            print(f"Exists dumplicates points: {[(i, i+1) for i in zero_idxs]}")
        
    _dict = {'pid_0': idxs[:-1],
             'pid_1': idxs[1:],
             'euc_dist': dist_arr}

    if move_dir:
        dirs = cal_coords_seq_azimuth(coords)
        _dict['move_dir'] = dirs
    
    res = pd.DataFrame(_dict)

    return res

def identify_edge_flag(gt: pd.DataFrame, cands: GeoDataFrame, ratio_eps: float = 0.05, dist_eps: float = 5):
    """
    Identify the type of querying the shortest path from the candidate `src` to `dst` on the graph.

    Args:
        gt (pd.DataFrame): The graph DataFrame.
        cands (GeoDataFrame): The DataFrame containing candidate edges.
        ratio_eps (float, optional): The ratio epsilon parameter. Defaults to 0.05.
        dist_eps (float, optional): The distance epsilon parameter. Defaults to 5.

    Returns:
        pd.DataFrame: The graph DataFrame with the 'flag' column appended.

    Example:
        >>> graph = pd.DataFrame([...])  # Graph DataFrame
        >>> candidates = gpd.GeoDataFrame([...])  # Candidate edges DataFrame
        >>> flagged_graph = identify_edge_flag(graph, candidates, ratio_eps=0.05, dist_eps=5)
        >>> print(flagged_graph)

    Notes:
        - The 'gt' DataFrame represents the graph and should contain necessary columns such as 'eid_0', 'eid_1',
          'dist_0', 'step_0_len', 'step_n_len', etc.
        - The 'cands' DataFrame should contain candidate edges information, including columns such as 'pid', 'eid',
          'seg_0', 'len_0', etc.
        - The 'ratio_eps' and 'dist_eps' parameters control the thresholds for identifying different edge types.
        - The resulting graph DataFrame will have an additional 'flag' column indicating the edge type.

    Refs:
        - Fast map matching, an algorithm integrating hidden Markov model with precomputation, Fig 4.
    """
    # (src, dst) on the same edge
    gt.loc[:, 'flag'] = CANDS_EDGE_TYPE.NORMAL

    same_edge = gt.eid_0 == gt.eid_1
    tmp = gt['dist_0'] - gt['step_0_len']
    cond_1 = tmp <= gt['step_n_len']

    tmp = tmp.apply(lambda x: min(max(0, x - dist_eps), x * (1 - ratio_eps)))
    cond = tmp <= gt['step_n_len']

    # Perform merging of adjacent nodes within a certain range (5 meter)
    cond_approx_points = cond & (~cond_1)
    _cands = cands[['pid', 'eid', 'seg_0', 'len_0']]\
                  .set_index(['pid', 'eid']).to_dict('index')
    # reset related params
    gt.loc[cond_approx_points, ['step_n', 'step_n_len']] = gt.loc[cond_approx_points].apply(
        lambda x: _cands[(x.pid_0, x.eid_0)].values(), axis=1, result_type='expand'
    ).rename(columns={0: 'step_n', 1: 'step_n_len'})

    same_edge_normal = same_edge & cond
    gt.loc[same_edge_normal, 'flag'] = CANDS_EDGE_TYPE.SAME_SRC_FIRST
    gt.loc[same_edge_normal, ['src', 'dst']] = gt.loc[same_edge_normal, ['dst', 'src']].values

    same_edge_revert = same_edge & (~cond)
    gt.loc[same_edge_revert, 'flag'] = CANDS_EDGE_TYPE.SAME_SRC_LAST

    return gt

@timeit
def construct_graph( points,
                     cands,
                     common_attrs = ['pid', 'eid', 'dist', 'speed'], 
                     left_attrs = ['dst', 'len_1', 'seg_1'], 
                     right_attrs = ['src', 'len_0', 'seg_0', 'observ_prob'],
                     rename_dict = {
                            'seg_0': 'step_n',
                            'len_0': 'step_n_len',
                            'seg_1': 'step_0',
                            'len_1': 'step_0_len',
                            # 'cost': 'sp_dist'
                            },
                     dir_trans = True,
                     gt_keys = ['pid_0', 'eid_0', 'eid_1']
    ):
    """
    Construct the candiadte graph (level, src, dst) for spatial and temporal analysis.

    Parameters:
        path = step_0 + step_1 + step_n
    """
    layer_ids = np.sort(cands.pid.unique())
    prev_layer_dict = {cur: layer_ids[i]
                          for i, cur in enumerate(layer_ids[1:])}
    prev_layer_dict[layer_ids[0]] = -1

    # left
    left = cands[common_attrs + left_attrs]
    left.loc[:, 'mgd'] = left.pid

    # right
    right = cands[common_attrs + right_attrs]
    right.loc[:, 'mgd'] = right.pid.apply(lambda x: prev_layer_dict[x])
    right.query("mgd >= 0", inplace=True)

    # Cartesian product
    gt = left.merge(right, on='mgd', suffixes=["_0", '_1'])\
             .drop(columns='mgd')\
             .reset_index(drop=True)\
             .rename(columns=rename_dict)

    identify_edge_flag(gt, cands)
    traj_info = cal_traj_params(points.loc[cands.pid.unique()], move_dir=dir_trans)
    
    gt = gt.merge(traj_info, on=['pid_0', 'pid_1'])
    gt.loc[:, ['src', 'dst']] = gt.loc[:, ['src', 'dst']].astype(np.int64)

    if gt_keys:
        gt.set_index(gt_keys, inplace=True)
    
    return gt

def get_shortest_geometry(gt:GeoDataFrame, geom='geometry', format='LineString'):
    """
    Generate the shortest path geometry based on the given conditions.

    Parameters:
        gt (GeoDataFrame): A geospatial dataframe containing geometry objects and other attributes.
        geom (str, optional): The column name for the geometry objects. Default is 'geometry'.
        format (str, optional): The format of the returned geometry objects. Available options are 'LineString' or 'array'.
                                Default is 'LineString'.

    Returns:
        GeoDataFrame: An updated geospatial dataframe with the shortest path geometry objects.

    Notes:
        - Only 'LineString' and 'array' formats are supported.
        - The input GeoDataFrame must have a 'flag' column indicating whether it represents the shortest path.

    Example:
    >>> shortest_geo = get_shortest_geometry(geo_data, format='array')
    >>> print(shortest_geo)

    Raises:
    - AssertionError: If the provided format is not supported.
    """
    assert format in ['LineString', 'array']

    # FIXME: 1) step_1 is None; 2) same edge：w2h, level=27, 555->555
    mask = gt.flag == 1
    gt.loc[mask, geom] = gt.loc[mask].apply(lambda x: 
        get_shared_line(x.step_0, x.step_n), axis=1)
    gt.loc[~mask, geom] = gt.loc[~mask].apply(
        merge_step_arrs, axis=1)
    
    if format == 'LineString':
        gt.loc[:, geom] = gt[geom].apply(LineString)

    return gt
