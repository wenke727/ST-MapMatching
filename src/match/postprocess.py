from xmlrpc.client import Boolean
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString
from graph import GeoDigraph


def get_path(net:GeoDigraph, 
             traj:gpd.GeoDataFrame, 
             rList:gpd.GeoDataFrame, 
             graph:gpd.GeoDataFrame, 
             cands:gpd.GeoDataFrame,
             connector:Boolean = False
             ):
    """Get path by matched sequence node.

    Args:
        rList ([type]): [description]
        graph_t ([type]): [description]
        net ([type]): [description]

    Returns:
        [type]: [description]
    """
    if rList.shape[0] == 1:
        return net.merge_edge(rList, on=['src', 'dst']), None
    
    def _helper(x):
        res = graph.loc[x.pid].loc[x.eid].loc[x.nxt_eid].path
        return res if res is None else res
    
    rList.loc[:, 'nxt_eid'] = rList.eid.shift(-1).fillna(0).astype(np.int)
    steps = rList[:-1].apply(lambda x: _helper(x), axis=1)
    # Drop `Consecutive identical line segments`, and keep the first one record
    steps = steps[steps != steps.shift(1)]

    od_lst = [rList.iloc[0]['src']]
    for step in steps.values:
        if step is None or isinstance(step, np.float):
            continue
        
        if step[0] == od_lst[-1]:
            od_lst += step[1:]
        else:    
            od_lst += step
    od_lst += [rList.iloc[-1]['dst']]
    path = net.transform_node_seq_to_df_edge(od_lst)
    
    # update geometry of the first/last step 
    step_0 = cands.query(f'pid == {rList.iloc[0].pid} and eid == {rList.iloc[0].eid}').seg_1.values[0]
    step_n = cands.query(f'pid == {rList.iloc[-1].pid} and eid == {rList.iloc[-1].eid}').seg_0.values[0]
    n = path.shape[0] - 1
    if n == 0:
        coords = np.concatenate((step_0[0][np.newaxis, :], 
                                 step_0[[p in step_n for p in step_0]], 
                                 step_n[-1][np.newaxis, :]))
        path.loc[0, 'geometry'] = LineString(coords)
    else:
        path.loc[0, 'geometry'], path.loc[n, 'geometry'] = LineString(step_0), LineString(step_n)
        # filter empty geometry
        path = path[~path.geometry.is_empty]
        path.loc[0, 'memo'], path.loc[n, 'memo'] = 'first step', 'last step'

    # connector
    if not connector:
        return path, None
    
    p_0, p_n = traj.iloc[0].geometry, traj.iloc[-1].geometry
    # BUG path 的 geometry 为空
    try:
        connector_0 = LineString([(p_0.x, p_0.y), path.loc[0, 'geometry'].coords[0]])
    except:
        connector_0 = LineString([(p_0.x, p_0.y), (p_0.x, p_0.y)])
    try:
        connector_1 = LineString([path.loc[n, 'geometry'].coords[-1], (p_n.x, p_n.y)])
    except:
        connector_1 = LineString([(p_n.x, p_n.y), (p_n.x, p_n.y)])
        
    connectors = gpd.GeoDataFrame({
        'geometry': [
            connector_0, 
            connector_1], 
        'name':['connector_0', 'connector_1']})
    
    return path, connectors
