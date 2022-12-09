import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from .geometricAnalysis import project_point_to_line_segment
from ..graph import GeoDigraph


def project_traj_points_to_network(traj_panos:gpd.GeoDataFrame, path:gpd.GeoDataFrame, net:GeoDigraph, keep_attrs=[['pid', 'geometry']]):
    panos = traj_panos.copy()
    panos.loc[:, 'eid'] = panos.apply(lambda x: path.loc[path.distance(x.geometry).idxmin()].eid, axis=1)

    projectd_edges = net.df_edges.loc[panos['eid'].values]
    edge_geoms = projectd_edges.geometry.values

    df_project = project_point_to_line_segment(
        panos.geometry.values, edge_geoms, keep_cols=['projection', 'len_0', 'distance'])
    df_project.rename(columns={'projection': "projected_point", 'len_0': 'offset', 'distance': 'projected_dist'}, inplace=True)
    df_project.loc[:, 'projected_point'] = df_project.projected_point.apply(lambda x: Point(*x))

    if keep_attrs is None:
        keep_attrs = list(panos)
    panos = pd.concat([panos[keep_attrs], df_project], axis=1)
    panos.sort_values(['eid', 'offset'], inplace=True)

    return panos

