#%%
from mapmatching import build_geograph, ST_Matching
from mapmatching.setting import DATA_FOLDER
from tilemap import plot_geodata

net = build_geograph(ckpt = DATA_FOLDER / 'network/Shenzhen_graph_pygeos.ckpt')
matcher = ST_Matching(net=net)

#%%
# TODO max() arg is an empty seq
# traj = matcher.load_points("../LRNC/debug/异常记录/17_770724-d008-13c6-ee9e-f77910.geojson", compress=False)


# traj = matcher.load_points("../LRNC/debug/24_17a311-19b4-5bfb-1ff9-0cadf6.geojson", compress=False)
traj = matcher.load_points("../LRNC/debug/28_1060a0-7213-367b-e6b8-17e7e4.geojson", compress=False)
# traj = matcher.load_points("../LRNC/debug/44_114065-9768-b039-602a-fc48a0.geojson", compress=False)
# traj = matcher.load_points("../LRNC/debug/45_5ccdd0-494a-be1a-b9f4-2310f0.geojson", compress=False)
# traj = matcher.load_points("../LRNC/debug/47_58405e-e7f3-703f-4b9e-9d683d.geojson", compress=False)

traj = matcher.load_points("./data/tmp/trip_amp_hw_compressed.geojson", compress=False)

path, info = matcher.matching(traj, plot=False, top_k=5)
fig, ax = plot_geodata(traj, reset_extent=False)
path.plot(ax=ax, color='r')

# matcher.eval(traj, path, 10, 'lcss')

# %%
