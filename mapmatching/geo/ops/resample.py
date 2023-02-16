import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from ..haversineDistance import cal_points_geom_seq_distacne


def resample_point_seq(points, step=2, last=True):
    # TODO linear referencing
    if points.shape[0] == 1:
        return gpd.GeoDataFrame(points), np.array([points.iloc[0].coords[0]])

    dist, coords = cal_points_geom_seq_distacne(points)
    dxdy = coords[1:] - coords[:-1]

    cum_dist = np.cumsum(dist)
    cum_dist = np.concatenate([[0], cum_dist])
    samples = np.arange(0, cum_dist[-1], step)
    seg_ids = pd.cut(samples, bins=cum_dist, labels=range(len(dist)), right=False)

    samples_lst = []
    samples_coords = []
    for s, idx in zip(samples, seg_ids):
        ratio = (s - cum_dist[idx]) / dist[idx]
        xy = coords[idx] + dxdy[idx] * ratio
        samples_coords.append(xy)
        samples_lst.append({"seg_idx": idx, "offset": s, "geometry": Point(xy)})
    if last:
        samples_lst.append({"seg_idx": len(dist) - 1, "offset": dist[-1], "geometry": Point(coords[-1])})
        samples_coords.append(coords[-1])
    
    df_samples = gpd.GeoDataFrame(samples_lst)

    return df_samples, np.array(samples_coords)

def resample_polyline_seq_to_point_seq(polyline, step=2, last=True):
    coords = np.concatenate(polyline.apply(lambda x: x.coords))

    mask = np.sum(coords[:-1] == coords[1:], axis=1) == 2
    mask = np.concatenate([mask, [False]])
    geoms = gpd.GeoSeries([Point(i) for i in coords[~mask]])

    return resample_point_seq(geoms, step, last)


if __name__ == "__main__":
    df = gpd.read_file('./data/trajs/traj_9.geojson').head(20)

    df_samples, coords = resample_point_seq(df.geometry)

    df.plot(color='r', alpha=.1)

