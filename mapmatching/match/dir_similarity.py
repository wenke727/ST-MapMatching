from geopandas import GeoDataFrame

from ..geo.azimuth import cal_linestring_azimuth_cos_dist
from .candidatesGraph import get_shortest_geometry

def cal_dir_prob(gt:GeoDataFrame, geom='geometry'):
    # TODO: 使用 sub_string 代替现有的情况
    # Add: dir_prob
    def _cal_dir_similarity(x):
        return cal_linestring_azimuth_cos_dist(x[geom], x['move_dir'], weight=True)
    
    gt = get_shortest_geometry(gt, geom, format='array')
    gt.loc[:, 'dir_prob'] = gt.apply(_cal_dir_similarity, axis=1)
    
    return gt

