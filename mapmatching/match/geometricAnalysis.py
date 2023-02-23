import numpy as np
import pandas as pd
import geopandas as gpd
from loguru import logger
from shapely.geometry import box

from ..utils import Timer
from ..geo.query import get_k_neigh_geoms
from ..geo.ops.distance import geom_series_distance


def cal_observ_prob(dist, bias=0, deviation=20, normal=True):
    """The obervation prob is defined as the likelihood that a GPS sampling point `p_i` mathes a candidate point `C_ij`
    computed based on the distance between the two points. 

    Args:
        df (gpd.GeoDataFrame): Distance series or arrays.
        bias (float, optional): GPS measurement error bias. Defaults to 0.
        deviation (float, optional): GPS measurement error deviation. Defaults to 20.
        normal (bool, optional): Min-Max Scaling. Defaults to False.

    Returns:
        _type_: _description_
    """
    observ_prob_factor = 1 / (np.sqrt(2 * np.pi) * deviation)

    def f(x): return observ_prob_factor * \
        np.exp(-np.power(x - bias, 2)/(2 * np.power(deviation, 2)))

    _dist = f(dist)
    if normal:
        _dist /= _dist.max()

    return np.sqrt(_dist)

def analyse_geometric_info(points: gpd.GeoDataFrame,
                           edges: gpd.GeoDataFrame,
                           top_k: int = 5,
                           radius: float = 50,
                           edge_attrs: list = ['src', 'dst', 'way_id', 'dir', 'dist', 'geometry'],
                           pid: str = 'pid',
                           eid: str = 'eid',
                           ll: bool = True,
                           ):
    # TODO improve effeciency: get_k_neigbor_edges 50 %, project_point_to_line_segment 50 %
    cands, _ = get_k_neigh_geoms(points.geometry, edges, 
                                 query_id='pid', project=True, top_k=top_k, 
                                 keep_geom=True, radius=radius)
    cands.loc[:, 'observ_prob'] = cal_observ_prob(cands.dist_p2c)

    return cands
    

if __name__ == "__main__":
    from shapely.geometry import Point, LineString
    
    # edges
    lines = [LineString([[0, i], [10, i]]) for i in range(0, 10)]
    lines += [LineString(([5.2,5.2], [5.8, 5.8]))]
    edges = gpd.GeoDataFrame({'geometry': lines, 
                              'way_id':[i for i in range(10)] + [5]})
    # points
    a, b = Point(1, 1.1), Point(5, 5.1) 
    points = gpd.GeoDataFrame({'geometry': [a, b]}, index=[1, 3])
    
    # candidates
    cands = get_k_neigbor_edges(points, edges, radius=2, top_k=2, ll=False)
    plot_candidates(points, edges, cands)
    
