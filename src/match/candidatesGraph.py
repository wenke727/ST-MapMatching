
import pandas as pd

def construct_graph( cands,
                     common_attrs=['pid', 'eid', 'mgd'],
                     left_attrs=['dst', 'len_1', 'seg_1', 'dist'],
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
    gt = []
    for i in range(len(tList)-1):
        a = tList[i][common_attrs + left_attrs]
        b = tList[i+1][common_attrs + right_attrs]
        gt.append(a.merge(b, on='mgd', suffixes=["_0", '_1']).drop(columns='mgd'))
    gt = pd.concat(gt).reset_index(drop=True).rename(columns=rename_dict)

    # (src, dst) on the same edge
    same_edge = gt.eid_0 == gt.eid_1
    cond = (gt.dist - gt.offset_1) < gt.offset_0
    normal = same_edge & cond
    revert = same_edge & (~cond)
    
    # 0: od不一样；1 od 位于同一条edge上，但起点相对终点位置偏前；2 相对偏后
    gt.loc[:, 'flag'] = 0
    gt.loc[normal, 'flag'] = 1
    gt.loc[revert, 'flag'] = 2

    gt.loc[same_edge, ['src', 'dst']] = gt.loc[same_edge, ['dst', 'src']].values
    
    return gt

