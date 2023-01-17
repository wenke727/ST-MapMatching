import numpy as np
import geopandas as gpd
from shapely.geometry import LineString

from .status import STATUS
from ..graph import GeoDigraph
from ..geo.misc import merge_coords_intervals_on_same_edge


def get_path(net:GeoDigraph, 
             traj:gpd.GeoDataFrame, 
             rList:gpd.GeoDataFrame, 
             graph:gpd.GeoDataFrame, 
             cands:gpd.GeoDataFrame,
             metric = {},
             prob_thres = .8
             ):
    """Get path by matched sequence node.

    Args:
        rList ([type]): [description]
        graph_t ([type]): [description]
        net ([type]): [description]

    Returns:
        [list]: [path, connectors, steps]
    
    Example:
        rList
       |    |   pid |   eid |         src |         dst |\n
       |---:|------:|------:|------------:|------------:|\n
       |  0 |     0 | 17916 |  8169270272 |  2376751183 |\n
       |  1 |     1 | 17916 |  8169270272 |  2376751183 |
    """ 
    steps = rList.copy()
    steps.loc[:, 'eid_1'] = steps.eid.shift(-1).fillna(0).astype(int)
    idxs = steps[['pid', 'eid', 'eid_1']].values[:-1].tolist()
    steps = graph.loc[idxs, ['epath', 'dist_prob', 'trans_prob']].reset_index()

    extract_eids = lambda x: np.concatenate([[x.eid_0], x.epath]) if x.epath else [x.eid_0]
    eids = np.concatenate(steps.apply(extract_eids, axis=1))
    eids = np.append(eids, [steps.iloc[-1].eid_1])
    keep_cond = np.append([True], eids[:-1] != eids[1:])
    eids_lst = eids[keep_cond].tolist()

    res = {'epath': eids_lst}
    step_0, step_n = _get_first_and_last_step(cands, rList)

    # Case: one step
    if len(eids_lst) == 1:
        tmp = merge_coords_intervals_on_same_edge(step_0, step_n)
        res['step_0'] = tmp.tolist() if tmp is not None else []
        if metric.get('prob', 1) < prob_thres:
            metric['status'] = STATUS.FAILED
        else:
            metric['status'] = STATUS.SAME_LINK 
            
        return res, None

    # update first/last step 
    n = len(eids_lst) - 1
    assert n > 0, "Check od list"
    res['step_0'] = step_0.tolist() if step_0 is not None else []
    res['step_n'] = step_n.tolist() if step_n is not None else []

    # update metric
    coef = 1 / len(steps.dist_prob)
    dist_prob = np.prod(steps.dist_prob)
    trans_prob = np.prod(steps.trans_prob)
    metric["norm_prob"], metric["dist_prob"], metric["trans_prob"] = \
        np.power([metric['prob'], dist_prob, trans_prob], coef)
    if "dir_prob" in list(graph):
        metric["dir_prob"] = metric["trans_prob"] / metric["dist_prob"]

    # status
    if metric["trans_prob"] < prob_thres:
        metric['status'] = STATUS.FAILED
    else:
        metric['status'] = STATUS.SUCCESS
            
    return res, steps


def _get_first_and_last_step(cands, rList):
    step_0 = cands.query(
        f'pid == {rList.iloc[0].pid} and eid == {rList.iloc[0].eid}').seg_1.values[0]
    step_n = cands.query(
        f'pid == {rList.iloc[-1].pid} and eid == {rList.iloc[-1].eid}').seg_0.values[0]

    return step_0, step_n


def get_connectors(traj, path):
    p_0, p_n = traj.iloc[0].geometry, traj.iloc[-1].geometry
    # BUG geometry 为空
    try:
        connector_0 = LineString([(p_0.x, p_0.y), path.loc[0, 'geometry'].coords[0]])
    except:
        connector_0 = LineString([(p_0.x, p_0.y), (p_0.x, p_0.y)])
    try:
        connector_1 = LineString([path.loc[path.shape[0] - 1, 'geometry'].coords[-1], (p_n.x, p_n.y)])
    except:
        connector_1 = LineString([(p_n.x, p_n.y), (p_n.x, p_n.y)])
        
    connectors = gpd.GeoDataFrame({
        'geometry': [
            connector_0, 
            connector_1], 
        'name':['connector_0', 'connector_1']})

    return connectors

        
