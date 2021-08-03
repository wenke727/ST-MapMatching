#%%
import os
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import LineString, Point

import matplotlib.pyplot as plt
from utils.geo_plot_helper import map_visualize
from load_step import split_line_to_points
from DigraphOSM import load_net_helper, Digraph_OSM
from matching import st_matching, load_trajectory, get_candidates
from setting import SZ_BBOX
from utils.geo_helper import gdf_to_geojson, load_postgis

PCL_BBOX = [113.931914,22.573536, 113.944456,22.580613]

roads = load_postgis('roads', PCL_BBOX)

net = load_net_helper(bbox=PCL_BBOX, combine_link=True, reverse_edge=True, overwrite=False, two_way_offeset=True)

# %%
# 剩下的都是只有一个点的轨迹数据
skip_lst = []

for id in tqdm(range(0, roads.shape[0])):
    traj = split_line_to_points(roads.iloc[id].geometry, compress=True, config={'dist_max': .5, 'verbose': False}).reset_index()
    path = st_matching(traj, net, plot=False, satellite=False) 
    if path is None:
        skip_lst.append(id)
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
# net.df_edges[~net.df_edges.oneway].plot()
id = 169 # 129, 149, Done_lst = [49, 114, 149, 157, 169 ]
traj = split_line_to_points(roads.iloc[id].geometry, compress=True, config={'dist_max': .5, 'verbose': True}).reset_index()

path = st_matching(traj, net, plot=True, debug_in_levels=True)

#%%
gdf_to_geojson(traj, f'../input/traj_debug_{id}')


# %%

# %%
from utils.pickle_helper import PickleSaver

saver = PickleSaver()
# saver.save(net.route_planning_memo, "route_planning_memo")

net.route_planning_memo = saver.read('route_planning_memo.pkl')


# %%
for id in skip_lst:
    traj = split_line_to_points(roads.iloc[id].geometry, compress=True, config={'dist_max': .5, 'verbose': True}).reset_index()
    path = st_matching(traj, net, plot=True)

# %%
