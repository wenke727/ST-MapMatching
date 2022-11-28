import pandas as pd
import geopandas  as gpd
from ..geo.coord.coordTransfrom_shp import coord_transfer
from ..geo.douglasPeucker import simplify_trajetory_points


def load_points(fn, compress: bool = False, dp_thres: int = None, crs: int = None, in_sys: str = 'wgs', out_sys: str = 'wgs'):
    traj = gpd.read_file(fn, encoding='utf-8')
    if crs is not None:
        traj.set_crs(crs, allow_override=True, inplace=True)

    if 'time' in traj.columns:
        traj.time = pd.to_datetime(
            traj['time'], format='%Y-%m-%d %H:%M:%S')

    traj = coord_transfer(traj, in_sys, out_sys)

    if compress:
        traj_bak = traj.copy()
        traj = traj = simplify_trajetory_points(traj, dp_thres, inplace=True)
    else:
        traj_bak = None
        traj = traj

    return traj, traj_bak
