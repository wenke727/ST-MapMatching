from geopandas import GeoDataFrame

from .misc import get_shared_line, merge_step_arrs
from ..geo.azimuth import cal_linestring_azimuth_cos_dist


def cal_dir_prob(gt:GeoDataFrame, geom='geometry'):
    # Add: dir_prob
    def _cal_dir_similarity(x):
        return cal_linestring_azimuth_cos_dist(x[geom], x['move_dir'], weight=True)

    # complete shortest path
    mask = gt.flag == 1
    gt.loc[mask, geom] = gt.loc[mask].apply(lambda x: 
        get_shared_line(x.step_0, x.step_n), axis=1)
    gt.loc[~mask, geom] = gt.loc[~mask].apply(
        merge_step_arrs, axis=1)
    gt.loc[:, 'dir_prob'] = gt.apply(_cal_dir_similarity, axis=1)
    
    return gt

