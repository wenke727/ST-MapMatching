from xmlrpc.client import Boolean
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString
from utils.timer import Timer
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
    '''|    |   pid |   eid |         src |         dst |\n
       |---:|------:|------:|------------:|------------:|\n
       |  0 |     0 | 17916 |  8169270272 |  2376751183 |\n
       |  1 |     1 | 17916 |  8169270272 |  2376751183 |'''
    # case 1
    if rList.eid.nunique() == 1:
        # 仅有一条记录的情况, 且 `eid_1` 为 0
        path = get_one_step(net, rList, cands)
        
        return path, get_connectors(traj, path) if connector else None
        
    # new 
    rList.loc[:, 'eid_1'] = rList.eid.shift(-1).fillna(0).astype(np.int)
    steps = rList.rename(columns={'pid':'pid_0', 'eid':'eid_0'})\
                 .query('eid_0 != eid_1')\
                 .set_index(['pid_0', 'eid_0', 'eid_1'])\
                 .merge(graph, left_index=True, right_index=True)['path']

    od_lst = np.concatenate(steps.values) if len(steps.values) > 1 else steps.values.tolist()[0]
    od_lst = np.concatenate([
        [rList.iloc[0]['src']], od_lst, [rList.iloc[-1]['dst']]])
    
    timer = Timer()
    timer.start()
    # TODO 通过 eids_list 快速定位
    path = net.transform_node_seq_to_df_edge(od_lst)
    print(f"transform_node_seq_to_df_edge: {timer.stop():.4f} s")

    # update geometry of the first/last step 
    step_0 = cands.query(f'pid == {rList.iloc[0].pid} and eid == {rList.iloc[0].eid}').seg_1.values[0]
    step_n = cands.query(f'pid == {rList.iloc[-1].pid} and eid == {rList.iloc[-1].eid}').seg_0.values[0]
    n = path.shape[0] - 1
    assert n > 0, "Check od list"

    path.loc[0, 'geometry'], path.loc[n, 'geometry'] = LineString(step_0), LineString(step_n)
    path.loc[0, 'memo'], path.loc[n, 'memo'] = 'first step', 'last step'
    # filter empty geometry
    path = path[~path.geometry.is_empty]
    
    return path, get_connectors(traj, path) if connector else None


def get_one_step(net, rList, cands):
    r = rList.iloc[0]
    step_0 = cands.query(f'pid == {rList.iloc[0].pid} and eid == {rList.iloc[0].eid}').seg_1.values[0]
    step_n = cands.query(f'pid == {rList.iloc[-1].pid} and eid == {rList.iloc[-1].eid}').seg_0.values[0]
    coords = np.concatenate((step_0[0][np.newaxis, :], 
                                step_0[[p in step_n for p in step_0]], 
                                step_n[-1][np.newaxis, :]))

    od_lst = [r.src, r.dst]
    # TODO 通过 eids_list 快速定位
    path = net.transform_node_seq_to_df_edge(od_lst)
    path.loc[0, 'geometry'] = LineString(coords)    

    return path


def get_connectors(traj, path):
    p_0, p_n = traj.iloc[0].geometry, traj.iloc[-1].geometry
    # BUG path 的 geometry 为空
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
