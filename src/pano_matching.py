#%%
import os
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import LineString, Point

import matplotlib.pyplot as plt
from utils.geo_plot_helper import map_visualize
from load_step import split_line_to_points
from DigraphOSM import load_net_helper, Digraph_OSM
from matching import st_matching, load_trajectory
from setting import SZ_BBOX

net = load_net_helper(bbox=SZ_BBOX, combine_link=True, convert_to_geojson=True)

#%%
from utils.geo_helper import load_postgis

PCL_BBOX = [113.931914,22.573536, 113.944456,22.580613]

roads = load_postgis('roads', PCL_BBOX)


# %%
for id in tqdm(range(0, roads.shape[0])):
    traj = split_line_to_points(roads.iloc[id].geometry, compress=True, config={'dist_max': .5, 'verbose': True})
    path = st_matching(traj, net, plot=False, satellite=False) 
    if path is None:
        continue
    
    fig, ax = map_visualize(roads, scale=0, figsize=(15, 12), color='gray')
    path.plot(ax=ax, color='blue', label='Matching road', alpha=.6)
    roads.iloc[[id]].plot(ax=ax, color='red', linestyle='--', label='Pano path')
    traj.head(1).plot(ax=ax, color='red', marker='*', label='Start', alpha=.6)
    traj.sort_index(ascending=False).head(1).plot(ax=ax, color='green', label='End', alpha=.6)
    
    ax.set_title(f'id: {id}, rid: {roads.iloc[id].RID}')
    ax.legend()
    ax.axis('off')

    fig.savefig(os.path.join("../debug/lxd", f'{roads.iloc[id].RID}.jpg'), dpi=300, bbox_inches='tight', pad_inches=0)

    plt.close()

# %%
net.df_edges[~net.df_edges.oneway].plot()


