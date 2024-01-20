import warnings
import numpy as np
from geopandas import GeoDataFrame

from ..graph import GeoDigraph
from .status import CANDS_EDGE_TYPE
from .dir_similarity import cal_dir_prob
from .candidatesGraph import construct_graph


def cal_dist_prob(gt: GeoDataFrame, net: GeoDigraph, max_steps: int = 2000, max_dist: int = 10000, eps: float = 1e-6):
    """
    Calculate the distance probability for each edge in the graph.

    Args:
        gt (GeoDataFrame): The graph GeoDataFrame.
        net (GeoDigraph): The network GeoDigraph.
        max_steps (int, optional): The maximum number of steps for route planning. Defaults to 2000.
        max_dist (int, optional): The maximum distance for route planning. Defaults to 10000.
        eps (float, optional): The epsilon value for comparing distances. Defaults to 1e-6.

    Returns:
        GeoDataFrame: The graph GeoDataFrame with additional columns 'sp_dist' and 'dist_prob'.

    Example:
        >>> graph = GeoDataFrame([...])  # Graph GeoDataFrame
        >>> network = GeoDigraph([...])  # Network GeoDigraph
        >>> graph = cal_dist_prob(graph, network, max_steps=3000, max_dist=15000, eps=1e-5)
        >>> print(graph)

    Notes:
        - The 'gt' GeoDataFrame should contain the graph data with required columns including 'flag', 'cost', 'avg_speed', 'epath', 'coords', 'step_0_len', 'step_n_len', 'dist_0', 'euc_dist'.
        - The 'net' GeoDigraph should be a network representation used for route planning.
        - The 'max_steps' parameter specifies the maximum number of steps for route planning.
        - The 'max_dist' parameter specifies the maximum distance for route planning.
        - The 'eps' parameter is used for comparing distances and should be a small positive value.
        - The function calculates the shortest paths and temporal probabilities for each edge in the graph.
        - It adds the following columns to the 'gt' GeoDataFrame:
            - 'cost': The cost of the shortest path.
            - 'avg_speed': The average speed on the shortest path.
            - 'epath': The edge path of the shortest path.
            - 'step_1': The first step of the shortest path.
            - 'sp_dist': The total distance of the shortest path.
            - 'dist_prob': The distance probability for the edge.
        - The function modifies the 'gt' GeoDataFrame in place and returns the modified GeoDataFrame.
    """

    assert 'flag' in gt, "Check the attribute `flag` in gt or not"
    if gt.empty:
        warnings.warn("Empty graph layer")
        return gt

    sp_attrs = ['dist', "avg_speed", 'epath', 'coords', 'weight']
    gt_sp_attrs = ['dist', "avg_speed", 'epath', 'step_1', 'weight']
    rout_planning = lambda x: net.search(x.dst, x.src, max_steps, max_dist)
    paths = gt.apply(rout_planning, axis=1, result_type='expand')[sp_attrs]
    gt.loc[:, gt_sp_attrs] = paths.values

    cal_temporal_prob(gt)
    
    assert not gt.dist.hasnans, "check distance"
    gt.loc[:, 'sp_dist'] = gt.dist.fillna(0) + gt.step_0_len + gt.step_n_len 
    inf_mask = gt['weight'] != gt['weight']
    gt[inf_mask, 'sp_dist'] = np.inf

    # OD is on the same edge, but the starting point is relatively ahead of the endpoint
    flag_1_idxs = gt.query(f"flag == {CANDS_EDGE_TYPE.SAME_SRC_FIRST}").index
    if len(flag_1_idxs):
        gt.loc[flag_1_idxs, ['epath', 'step_1']] = None, None
        gt.loc[flag_1_idxs, 'sp_dist'] = gt.step_0_len + gt.step_n_len - gt.dist_0

        idx = gt.query(f"flag == {CANDS_EDGE_TYPE.SAME_SRC_FIRST} and sp_dist < {eps}").index
        gt.loc[idx, 'sp_dist'] = gt.euc_dist

    # distance trans prob
    dist_prpb = gt.euc_dist / gt.sp_dist
    mask = dist_prpb > 1 
    dist_prpb[mask] = 1 / dist_prpb[mask]
    gt.loc[:, 'dist_prob'] = dist_prpb

    return gt

def cal_temporal_prob(gt: GeoDataFrame, eps=1e-6):
    """
    Calculate the temporal probability for each edge in the graph.

    Args:
        gt (GeoDataFrame): The graph GeoDataFrame.
        eps (float, optional): The epsilon value for handling infinite or zero weights. Defaults to 1e-6.

    Returns:
        GeoDataFrame: The graph GeoDataFrame with additional column 'avg_speed'.

    Example:
        >>> graph = GeoDataFrame([...])  # Graph GeoDataFrame
        >>> graph = cal_temporal_prob(graph, eps=1e-5)
        >>> print(graph)

    Notes:
        - The 'gt' GeoDataFrame should contain the graph data with required columns including 'speed_0', 'speed_1', 'avg_speed', 'step_0_len', 'step_n_len', 'cost'.
        - The 'eps' parameter is used for handling infinite or zero weights and should be a small positive value.
        - The function calculates the average speed for each edge based on the given weights.
        - It adds the 'avg_speed' column to the 'gt' GeoDataFrame.
        - The function modifies the 'gt' GeoDataFrame in place and returns the modified GeoDataFrame.
    """
    speeds = gt[['speed_0', 'speed_1', 'avg_speed']].values
    weights = gt[['step_0_len', 'step_n_len', 'dist']].values
    weights[weights == np.inf] = eps
    weights[weights == 0] = eps
    avg_speeds = np.average(speeds, weights=weights, axis=1)

    gt.loc[:, 'avg_speed'] = avg_speeds
    # gt.loc[:, 'eta'] = gt.sp_dist.values / avg_speeds

    return gt

def cal_trans_prob(gt, geometry, dir_trans):
    if not dir_trans:
        gt.loc[:, 'trans_prob'] = gt.dist_prob
        return gt

    cal_dir_prob(gt, geometry)
    gt.loc[:, 'trans_prob'] = gt.dist_prob * gt.dir_prob
    
    return gt

def analyse_spatial_info(net: GeoDigraph,
                         points: GeoDataFrame,
                         cands: GeoDataFrame,
                         dir_trans=False,
                         max_steps: int = 2e3,
                         max_dist: int = 1e5,
                         gt_keys: list = ['pid_0', 'eid_0', 'eid_1'],
                         geometry='whole_path'):
    """
    Geometric and topological info, the product of `observation prob` and the `transmission prob`
    """
    gt = construct_graph(points, cands, dir_trans=dir_trans, gt_keys=gt_keys)
    
    gt = cal_dist_prob(gt, net, max_steps, max_dist)
    cal_trans_prob(gt, geometry, dir_trans)

    return gt

def get_trans_prob_bet_layers(gt, net, dir_trans=True, geometry='path'):
    """
    For beam-search
    """
    if gt.empty:
        return gt

    gt = cal_dist_prob(gt, net)
    cal_trans_prob(gt, geometry, dir_trans)
    
    return gt
