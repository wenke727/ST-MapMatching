#%%
from mapmatching import build_geograph, ST_Matching
from mapmatching.setting import DATA_FOLDER
from tilemap import plot_geodata

net = build_geograph(ckpt = DATA_FOLDER / 'network/Shenzhen_graph_pygeos.ckpt')
matcher = ST_Matching(net=net)

#%%
# TODO max() arg is an empty seq
# traj = matcher.load_points("../LRNC/debug/异常记录/17_770724-d008-13c6-ee9e-f77910.geojson", compress=False)


traj = matcher.load_points("../LRNC/debug/异常记录/26_1060a0-7213-367b-e6b8-17e7e4.geojson", compress=False)
# traj = matcher.load_points("../LRNC/debug/异常记录/29_72437e-5c06-ba42-fc8b-f15dbf.geojson", compress=False)
# traj = matcher.load_points("../LRNC/debug/异常记录/38_69cc38-49a2-3ff2-527b-f796b5.geojson", compress=False)
# traj = matcher.load_points("../LRNC/debug/异常记录/39_ba4ee2-bce1-b948-0271-5e80f1.geojson", compress=False)

path, info = matcher.matching(traj, plot=False, top_k=5)
fig, ax = plot_geodata(traj, reset_extent=False)
path.plot(ax=ax, color='r')

matcher.eval(traj, path, 100, 'lcss')

# %%
