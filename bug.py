# %%
import pandas as pd
import geopandas as gpd
from mapmatching import build_geograph, ST_Matching

pd.set_option('display.width', 5000)        # 打印结果不换行方法
pd.set_option('display.max_rows', 500)

# %%
net = build_geograph(ckpt='./data/network/GZ_test.ckpt')
matcher = ST_Matching(net=net)

traj = gpd.read_file('./data/traj_others.geojson').set_crs(epsg=4326)
res = matcher.matching(traj, top_k=8, search_radius=80, plot=True,
                       dir_trans=False, details=True, simplify=False, debug_in_levels=True)
graph = res['details']['graph']
res['details']['steps'].query('trans_prob < .85')

# %%
