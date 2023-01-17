#%%
from mapmatching import build_geograph, ST_Matching
from mapmatching.setting import DATA_FOLDER
from tilemap import plot_geodata


net = build_geograph(ckpt='./data/network/Shenzhen_graph_pygeos.ckpt')
matcher = ST_Matching(net=net)

from loguru import logger

# %%
tmp = matcher.load_points("./data/case_5.geojson")

revert = False
if revert:
    tmp = tmp[::-1].reset_index(drop=True)

res = matcher.matching(tmp, simplify=False, details=True, dir_trans=False, debug_in_levels=False, top_k=3)
fig, ax = matcher.plot_result(tmp, res)
path = matcher.transform_res_2_path(res)

print(res['epath'])
print('lcss: ', matcher.eval(tmp, res, metric='lcss', resample=5, eps=10))

res['details']['steps']

# %%
res['details']['rList']

# %%
res['details']['simplified_traj']

# %%
