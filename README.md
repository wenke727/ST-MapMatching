# ST-MapMatching

## 版本

V2.0.0

## 描述

基于时间和空间特性的全局地图匹配算法（ST-Matching），一种针对低采样率的GPS轨迹的新颖全局地图匹配算法。算法的基础路网信息源为 [OSM](https://wiki.openstreetmap.org/wiki/Main_Page)，可通过`DigraphOSM`自动下载。算法匹配过程考虑以下两个方面特征：

1. 道路网络的空间几何和拓扑结构

2. 轨迹的速度/时间约束。基于时空分析，构建候选图，从中确定最佳匹配路径。

输入WGS坐标系的`GPS轨迹点集`，输出途径的路段；

本算法为 MSRA《[Map-Matching for Low-Sampling-Rate GPS Trajectories](https://www.microsoft.com/en-us/research/publication/map-matching-for-low-sampling-rate-gps-trajectories/)》的复现，中文解读可参考 [CSDN文章](https://blog.csdn.net/qq_43281895/article/details/103145327)。

## 调用说明

详见 `demo.py`

```python
from mapmatching import build_geograph, ST_Matching

"""step 1: 获取/加载路网"""
# 方法1：
# 根据 bbox 从 OSM 下载路网，从头解析获得路网数据
# net = build_geograph(bbox=[113.930914, 22.570536, 113.945456, 22.585613],
#                      xml_fn="./data/network/LXD.osm.xml", ll=False)
# 将预处理路网保存为 ckpt
# net.save_checkpoint('./data/network/LXD_graph.ckpt')

# 方法2：
# 使用预处理路网 
net = build_geograph(ckpt='./data/network/LXD_graph.ckpt')

"""step 2: 创建地图匹配 matcher"""
matcher = ST_Matching(net=net, ll=False)

"""step 3: 加载轨迹点集合，以打石一路为例"""
idx = 4
traj = matcher.load_points(f"./data/trajs/traj_{idx}.geojson").reset_index(drop=True)
res = matcher.matching(traj, plot=True, top_k=5, dir_trans=True, 
                       details=False, simplify=True, debug_in_levels=False)

# 后续步骤可按需选择
"""step 4: 将轨迹点映射到匹配道路上"""
path = matcher.transform_res_2_path(res)
proj_traj = matcher.project(traj, path)

"""step 5: eval"""
matcher.eval(traj, res, resample=5, eps=10)
```

### 输入示例

```json
{"type": "FeatureCollection",
"name": "trips",
"crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } },
"features": [
{ "type": "Feature", "properties": { "id": 0 }, "geometry": { "type": "Point", "coordinates": [ 114.042192099217814, 22.530825799254831 ] } },
{ "type": "Feature", "properties": { "id": 1 }, "geometry": { "type": "Point", "coordinates": [ 114.048087551857591, 22.53141414915628 ] } },
{ "type": "Feature", "properties": { "id": 2}, "geometry": { "type": "Point", "coordinates": [ 114.050457097022772, 22.530254493344991 ] } },
{ "type": "Feature", "properties": { "id": 3}, "geometry": { "type": "Point", "coordinates": [ 114.051374300525396, 22.534269663922935 ] } },
{ "type": "Feature", "properties": { "id": 4}, "geometry": { "type": "Point", "coordinates": [ 114.050237176637481, 22.537490331019249 ] } },
{ "type": "Feature", "properties": { "id": 5}, "geometry": { "type": "Point", "coordinates": [ 114.045217471559866, 22.54216729753638 ] } },
{ "type": "Feature", "properties": { "id": 6}, "geometry": { "type": "Point", "coordinates": [ 114.050182240637483, 22.542416259019245 ] } },
{ "type": "Feature", "properties": { "id": 7}, "geometry": { "type": "Point", "coordinates": [ 114.056957680637467, 22.542526131019244 ] } },
{ "type": "Feature", "properties": { "id": 8}, "geometry": { "type": "Point", "coordinates": [ 114.058074914718418, 22.537513356219687 ] } },
{ "type": "Feature", "properties": { "id": 9}, "geometry": { "type": "Point", "coordinates": [ 114.058331080637473, 22.531227627019256 ] } },
{ "type": "Feature", "properties": { "id": 10}, "geometry": { "type": "Point", "coordinates": [ 114.062890768637473, 22.529213307019258 ] } }
]
}
```

注:

1. 示例输入对应`./data/trajs/traj_0.geojson`，在`vscode`中可借助插件`Geo Data Viewer`可视化;
2. 输入轨迹点的坐标系默认为 wgs84, gcj02的轨迹需在调用函数`load_points`明确坐标系`in_sys='gcj'`,

### 输出示例

```python
{
  #  输出状态码，0 为正常输出
  'status': 0, 
  # 匹配路段 index
  'epath': [123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135], 
  # 第一条路被通过的比例(即第一条路上, 第一个轨迹点及之后的部分的占比)
  'step_0': 0.7286440473726905, 
  # 最后一条路被通过的比例(即最后一条路上, 最后一个轨迹点及之前的部分的占比）
  'step_n': 0.8915310605450645，
	# 概率
  'probs': {
    	'prob': 0.9457396931471692, 
      'norm_prob': 0.9861498301181256,
      'dist_prob': 0.9946361835772438,
      'trans_prob': 0.9880031610906268,
      'dir_prob': 0.9933312073337599}
 }
```

可视化效果如下

![](.fig/map_matching_futian.png)

## 环境安装

详见 requirement.txt, 建议`geopandas`使用conda安装

```bash
conda create -n geo python=3.7
conda activate geo
conda install -c conda-forge geopandas==0.12.1
```

## Ref

- batched compression algorithm
  - [轨迹数据压缩的Douglas-Peucker算法](https://zhuanlan.zhihu.com/p/136286488)
  - [基于MapReduce的轨迹压缩并行化方法](