from DigraphOSM import DigraphOSM
from MapMathing import ST_Matching


if __name__ == "__main__":

    traj_compress=True; traj_thres=None; top_k=None; plot=True; dir_trans=True; plot_scale=5
    NET = DigraphOSM("Shenzhen", resume='../cache/Shenzhen.pkl')
    # path = net.route_planning(o=7959990710, d=499265789, plot=True)

    self = ST_Matching(net=NET)

    # github演示数据
    traj = self.load_points("/home/pcl/traffic/MatchGPS2OSM/input/traj_0.geojson")
    path = self.matching(traj, plot=True, dir_trans=True, debug_in_levels=False)
    