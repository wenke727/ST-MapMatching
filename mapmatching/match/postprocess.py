import numpy as np
import geopandas as gpd
from shapely.geometry import LineString

from .status import STATUS
from .misc import get_shared_line
from ..utils.timer import timeit


@timeit
def get_path(rList:gpd.GeoDataFrame, 
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
    step_0, step_n = _get_first_and_step_n(cands, rList)

    # Case: one step
    if len(eids_lst) == 1:
        tmp = get_shared_line(step_0, step_n)
        res['step_0'] = tmp
        if metric.get('prob', 1) < prob_thres:
            metric['status'] = STATUS.FAILED
        else:
            metric['status'] = STATUS.SAME_LINK 
            
        return res, None

    # update first/last step 
    n = len(eids_lst) - 1
    assert n > 0, "Check od list"
    res['step_0'] = step_0
    res['step_n'] = step_n

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


def _get_first_and_step_n(cands, rList):
    step_0 = cands.query(
        f'pid == {rList.iloc[0].pid} and eid == {rList.iloc[0].eid}').seg_1.values[0]
    step_n = cands.query(
        f'pid == {rList.iloc[-1].pid} and eid == {rList.iloc[-1].eid}').seg_0.values[0]

    return step_0, step_n

