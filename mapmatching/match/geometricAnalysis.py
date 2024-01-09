import numpy as np
import geopandas as gpd
from ..geo.query import find_nearest_geometries


def cal_observ_prob(dist, bias=0, deviation=20, normal=True):
    """The obervation prob is defined as the likelihood that a GPS sampling point `p_i` 
    matches a candidate point `C_ij` computed based on the distance between the two points. 

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
                           pid: str = 'pid',
                           eid: str = 'eid',
                           bias=0,
                           deviation=20
                           ):
    # TODO improve effeciency: get_k_neigbor_edges 50 %, project_point_to_line_segment 50 %
    cands, _ = find_nearest_geometries(points.geometry, edges, 
                                 query_id=pid, project=True, top_k=top_k, 
                                 keep_geom=True, max_distance=radius)
    if cands is not None:
        # ? deviation 如何取值合适
        cands.loc[:, 'observ_prob'] = cal_observ_prob(cands.dist_p2c, bias, deviation)

    return cands
    
