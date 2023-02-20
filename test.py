#%%
from mapmatching import build_geograph, ST_Matching
from mapmatching.setting import DATA_FOLDER
from tilemap import plot_geodata, add_basemap

from mapmatching.geo.io import to_geojson
import matplotlib.pyplot as plt

ll = False
net = build_geograph(ckpt='./data/network/Shenzhen_graph_pygeos.ckpt', ll=ll)
matcher = ST_Matching(net=net, ll=ll)


#%%
traj = matcher.load_points("./data/trajs/traj_4.geojson").reset_index(drop=True)
proj_traj = net.align_crs(traj)
plot_geodata(traj)

# %%
from mapmatching.geo.query import get_k_neigh_geoms, plot_candidates
from geopandas import GeoSeries
from mapmatching.geo.ops.point2line import cut_linestring
from mapmatching.match.geometricAnalysis import cal_observ_prob

points = traj
net.df_edges.loc[:, 'dist'] = net.df_edges.length
cands, _ = get_k_neigh_geoms(points['geometry'], net.df_edges[['eid', 'src', 'dst', 'dist', 'geometry']], 
                             normalized=False, radius=50, top_k=5, keep_geom=True)

# plot_candidates(cands)

cands
cands.loc[:, ['seg_0', 'seg_1']] = cands.apply(
    lambda x: cut_linestring(x['edge_geom'], x['offset']), axis=1, result_type='expand')
cands.loc[:, 'observ_prob'] = cal_observ_prob(cands.dist_p2c)
cands.loc[:, 'len_0'] = GeoSeries(cands.seg_0).length
cands.loc[:, 'len_1'] = GeoSeries(cands.seg_1).length
cands

# %%
from mapmatching.match.candidatesGraph import construct_graph

graph = construct_graph(traj, cands.rename(columns={'qid': 'pid'}), dir_trans=True)

# prob, rList, graph = find_matched_sequence(cands, graph, net, dir_trans)

# %%
graph
# %%
