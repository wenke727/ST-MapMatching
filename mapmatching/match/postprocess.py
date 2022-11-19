import numpy as np
import geopandas as gpd
from shapely.geometry import LineString
from traitlets import Bool

from ..graph import GeoDigraph


def get_path(net:GeoDigraph, 
             traj:gpd.GeoDataFrame, 
             rList:gpd.GeoDataFrame, 
             graph:gpd.GeoDataFrame, 
             cands:gpd.GeoDataFrame,
             connector:Bool = False
             ):
    """Get path by matched sequence node.

    Args:
        rList ([type]): [description]
        graph_t ([type]): [description]
        net ([type]): [description]

    Returns:
        [type]: [description]
    
    Example:
        rList
       |    |   pid |   eid |         src |         dst |\n
       |---:|------:|------:|------------:|------------:|\n
       |  0 |     0 | 17916 |  8169270272 |  2376751183 |\n
       |  1 |     1 | 17916 |  8169270272 |  2376751183 |
    """
    # Case: one step
    if rList.eid.nunique() == 1:
        path = get_one_step(net, rList, cands)
        return path, get_connectors(traj, path) if connector else None
    
    # Case: normal
    steps = rList.copy()
    steps.loc[:, 'eid_1'] = steps.eid.shift(-1).fillna(0).astype(int)
    steps = steps.rename(columns={'pid':'pid_0', 'eid':'eid_0'})\
                 .query('eid_0 != eid_1')\
                 .set_index(['pid_0', 'eid_0', 'eid_1'])\
                 .merge(graph[['path']], left_index=True, right_index=True)\
                 .reset_index()

    extract_eids = lambda x: np.concatenate([[x.eid_0], x.path]) if x.path else [x.eid_0]
    eids = list(np.concatenate(steps.apply(extract_eids, axis = 1)))
    eids.append(steps.iloc[-1].eid_1)
    path = net.get_edge(eids, reset_index=True)

    # update first/last step 
    step_0 = cands.query(f'pid == {rList.iloc[0].pid} and eid == {rList.iloc[0].eid}').seg_1.values[0]
    step_n = cands.query(f'pid == {rList.iloc[-1].pid} and eid == {rList.iloc[-1].eid}').seg_0.values[0]
    n = path.shape[0] - 1
    assert n > 0, "Check od list"
    path.loc[0, 'geometry'] = LineString(step_0)
    path.loc[n, 'geometry'] = LineString(step_n)
    path.loc[0, 'memo'] = 'first step'
    path.loc[n, 'memo'] = 'last step'
    
    # filter empty geometry
    path = path[~path.geometry.is_empty]
    
    return path, get_connectors(traj, path) if connector else None


def get_one_step(net, rList, cands):
    r = rList.iloc[0]
    step_0 = cands.query(f'pid == {rList.iloc[0].pid} and eid == {rList.iloc[0].eid}').seg_1.values[0]
    step_n = cands.query(f'pid == {rList.iloc[-1].pid} and eid == {rList.iloc[-1].eid}').seg_0.values[0]

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

    path = net.get_edge([r.eid], reset_index=True)
    path.loc[0, 'geometry'] = LineString(coords)    

    return path


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

        
