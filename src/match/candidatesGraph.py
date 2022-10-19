
def _construct_graph(self, traj, cands):
    """Construct the candiadte graph (level, src, dst) for spatial and temporal analysis.
    """
    graph = []
    tList = [layer for _, layer in cands.groupby('pid')]

    base_atts = ['pid', 'eid','src', 'dst', 'len_0', 'len_1', 'seg_0', 'seg_1']
    cols_filter = [
        'pid_0',
        'pid_1',
        'eid_0',
        'eid_1',
        'dst_0',
        'src_1',
        'seg_0_1',
        'seg_1_0',
        'observ_prob',
        'len_0_1',
        'len_1_0',
    ]
    rename_dict = {
        'seg_0_1': 'step_first',
        'seg_1_0': 'step_last',
        'len_0_1': 'offset_0',
        'len_1_0': 'offset_1',
        'cost': 'd_sht',
    }
    
    # Cartesian product
    for i in range(len(tList)-1):
        a, b = tList[i][base_atts], tList[i+1][base_atts+['observ_prob']]
        a.loc[:, 'tmp'], b.loc[:, 'tmp'] = 1, 1 
        graph.append(a.merge(b, on='tmp', suffixes=["_0", '_1']).drop(columns='tmp') )
    graph = pd.concat(graph).reset_index(drop=True)
    
    graph = graph[[i for i in cols_filter if i in graph.columns]]
    graph.rename(columns=rename_dict, inplace=True)
    graph.loc[:, 'd_euc'] = graph.apply(
        lambda x: coords_pair_dist(traj.loc[x.pid_0].geometry, traj.loc[x.pid_1].geometry), axis=1)

    return graph