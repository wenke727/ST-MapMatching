from DigraphOSM import DigraphOSM
from MapMathing import ST_Matching

if __name__ == "__main__":
    # step 1: 获取/加载道路网络
    from setting import PCL_BBOX
    network = DigraphOSM("PCL", bbox=PCL_BBOX)
    # 预处理 深圳市network, 导入代码如下
    # network = DigraphOSM("Shenzhen", resume='../input/ShenzhenNetwork.pkl')
    
    # step 2: 创建地图匹配 matcher
    matcher = ST_Matching(net=network)

    # step 3: 加载轨迹点集合，以打石一路为例
    traj = matcher.load_points("../input/test/traj_debug_dashiyilu_0.geojson")
    
    # step 4: 开始匹配
    path = matcher.matching(traj, plot=True, top_k=3, dir_trans=True, plot_scale=.01)
    