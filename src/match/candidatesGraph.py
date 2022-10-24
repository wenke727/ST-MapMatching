
import pandas as pd

def construct_graph( cands,
                     common_attrs=['pid', 'eid', 'mgd'],
                     left_attrs=['dst', 'len_1', 'seg_1'],
                     right_attrs=['src', 'len_0', 'seg_0', 'observ_prob'],
                     rename_dict={
                            'seg_0': 'step_last',
                            'len_0': 'offset_0',
                            'seg_1': 'step_first',
                            'len_1': 'offset_1',
                            'cost': 'd_sht'}
    ):
    """
    Construct the candiadte graph (level, src, dst) for spatial and temporal analysis.
    """
    cands.loc[:, 'mgd'] = 1
    tList = [layer for _, layer in cands.groupby('pid')]
    cands.drop(columns=['mgd'], inplace=True)
    
    # Cartesian product
    graph = []
    for i in range(len(tList)-1):
        a = tList[i][common_attrs + left_attrs]
        b = tList[i+1][common_attrs + right_attrs]
        graph.append(a.merge(b, on='mgd', suffixes=["_0", '_1']).drop(columns='mgd'))
    graph = pd.concat(graph).reset_index(drop=True).rename(columns=rename_dict)

    return graph

