from DigraphOSM import load_net_helper, Digraph_OSM
from matching import st_matching, load_trajectory
from setting import SZ_BBOX


net = load_net_helper(bbox=SZ_BBOX, combine_link=True, convert_to_geojson=True)

""" matching test"""
traj = load_trajectory("../input/traj_0.geojson")
st_matching(traj, net, plot=True)


