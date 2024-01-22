import shapely
import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame

from .status import STATUS
from ..utils.timer import timeit
from ..graph import GeoDigraph
from ..geo.ops.point2line import project_points_2_linestrings


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
    steps = graph.loc[idxs, ['epath', 'sp_dist', 'avg_speed', 'dist_prob', 'trans_prob']].reset_index()

    # FIXME 使用 numba 加速 loop 测试
    extract_eids = lambda x: np.concatenate([[x.eid_0], x.epath]) if x.epath else [x.eid_0]
    eids = np.concatenate(steps.apply(extract_eids, axis=1))
    eids = np.append(eids, [steps.iloc[-1].eid_1])
    keep_cond = np.append([True], eids[:-1] != eids[1:])
    eids_lst = eids[keep_cond].tolist()

    res = {'epath': eids_lst}
    step_0, step_n = _get_first_and_step_n(cands, rList)

    # Case: one step
    if len(eids_lst) == 1:
        # tmp = get_shared_arr(step_0, step_n)
        res['step_0'] = step_0
        res['step_n'] = step_n
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
    res['dist'] = steps.sp_dist.sum()
    res['avg_speed'] = np.average(steps['avg_speed'].values, weights = steps['sp_dist'].values)

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
        f'pid == {rList.iloc[0].pid} and eid == {rList.iloc[0].eid}').iloc[0]
    step_n = cands.query(
        f'pid == {rList.iloc[-1].pid} and eid == {rList.iloc[-1].eid}').iloc[0]

    cal_offset = lambda x: x['len_0'] / (x['len_0'] + x['len_1'])

    return cal_offset(step_0), cal_offset(step_n)

def transform_mathching_res_2_path(res: dict, net: GeoDigraph, ori_crs: bool=True, attrs: list=None):
    if attrs is None:
        attrs = ['eid', 'way_id', 'src', 'dst', 'name', 'road_type', 'link', 'speed', 'dist', 'geometry']
        attrs = [i for i in attrs if i in net.df_edges.columns]
    
    path = net.get_edge(res['epath'], attrs, reset_index=True)

    _len = len(res['epath']) 
    if _len == 1:
        try:
            path.loc[0, 'dist'] *= res['step_n'] - res['step_0']
            path.loc[0, 'geometry'] = shapely.ops.substring(
                path.iloc[0].geometry, res['step_0'], res['step_n'], normalized=True)
        except:
            path.loc[0, 'dist'] = 0

    else:
        path.loc[0, 'dist'] *= 1 - res['step_0']
        path.loc[0, 'geometry'] = shapely.ops.substring(
            path.iloc[0].geometry, res['step_0'], 1, normalized=True)

        path.loc[_len - 1, 'dist'] *= res['step_n']
        path.loc[_len - 1, 'geometry'] = shapely.ops.substring(
            path.iloc[-1].geometry, 0, res['step_n'], normalized=True)
    
    path = path[~path.geometry.is_empty]
    if ori_crs:
        path = path.to_crs(res['ori_crs'])

    return path

def project(points: GeoDataFrame, path: GeoDataFrame, keep_attrs=['eid', 'proj_point'], normalized=True, reset_geom=True):
    """
    Project points onto a path represented by a GeoDataFrame.
    
    Args:
        points (GeoDataFrame): Points to be projected.
        path (GeoDataFrame): Path to project the points onto.
        keep_attributes (list, optional): Attributes to keep in the projected points. Defaults to ['eid', 'proj_point'].
        normalize (bool, optional): Whether to normalize the projection. Defaults to True.
        reset_geometry (bool, optional): Whether to reset the geometry column in the projected points. Defaults to True.

    Returns:
        GeoDataFrame: Projected points.

    Example:
        projected_points = project_points(points, path)
    """
    _points = points[[points.geometry.name]]
    ps = project_points_2_linestrings(_points, path.to_crs(points.crs), normalized=normalized)
    
    if keep_attrs:
        ps = ps[keep_attrs]

    ps = gpd.GeoDataFrame(pd.concat([_points, ps], axis=1), crs=points.crs)
    if reset_geom:
        ps.loc[:, 'ori_geom'] = points.geometry.apply(lambda x: x.wkt)
        ps.set_geometry('proj_point', inplace=True)
        ps.drop(columns=['geometry'], inplace=True)
    
    return ps
