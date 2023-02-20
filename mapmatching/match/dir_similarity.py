from geopandas import GeoDataFrame

from .misc import get_shared_line, merge_geom_steps
from ..geo.azimuth import cal_linestring_azimuth_cos_dist


def cal_dir_prob(gt:GeoDataFrame, geom='geometry'):
    # Add: dir_prob
    def _cal_dir_similarity(x):
        return cal_linestring_azimuth_cos_dist(x[geom], x['move_dir'], weight=True)

    # FIXME 可分为两种情况考虑，同一条路段上（flag == 1）；普通情况
    mask = gt.flag == 1
    gt.loc[mask, geom] = gt.loc[mask].apply(lambda x: get_shared_line(x.step_0, x.step_n), axis=1)
    gt.loc[~mask, geom] = gt.loc[~mask].apply(merge_geom_steps, axis=1)
    
    gt.loc[:, 'dir_prob'] = gt.apply(_cal_dir_similarity, axis=1)
    
    filtered_idxs = gt.query("flag == 1").index
    gt.loc[filtered_idxs, 'dir_prob'] = 1

    return gt

