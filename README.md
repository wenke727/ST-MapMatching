# ST-MapMatching

## 版本

1.2.7

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
net = build_geograph(bbox=[113.930914, 22.570536, 113.945456, 22.585613],
                     xml_fn="./data/network/LXD.osm.xml")

"""step 2: 创建地图匹配 matcher"""
matcher = ST_Matching(net=net)

"""step 3: 加载轨迹点集合，以打石一路为例"""
traj = matcher.load_points("./data/trajs/traj_4.geojson")
path, info = matcher.matching(traj, plot=True, top_k=5)
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

```json
{'status': 99,
 'probs': {'prob': 0.07239983608324635,
  'norm_prob': 0.7690839679357104,
  'dist_prob': 0.886956440947009,
  'trans_prob': 0.8333451587857633,
  'dir_prob': 0.9395559018614211,
  'status': 0},
 'eids': [8951,   1403,   1404,   1405,   1406,   1407,   1484,   1482, 1483,   1466, 118095, 118096,   1467,   1468,   1469,   1470, 1471,   1472,   1473,   1474, 117158, 122957, 122958, 117450, 117451, 117452, 117453, 122849, 122850, 117121, 117122,  96813, 96814,  96815,  96816,  96817,  96789, 122869,    771,  23786, 23808, 122874, 117741, 117742, 117743, 123309, 123310, 123311, 123312, 123313, 123314, 123315, 123316, 123317, 123318,   1536, 76069,   1537, 117105,    650,    651,    652,    653,    654, 655,    656,    657,    658, 124425, 124426, 124427, 124428, 118053, 118054,   1582,   1581,   1580,   1420,    531,   1645,  139265],
 'step_0': [[114.0421798,  22.5309295], [114.0425387,  22.530972 ]],
 'step_n': [[114.0630578,  22.5293481], [114.0630362,  22.5293596], [114.0628913,  22.5293602]])}
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
  - [基于MapReduce的轨迹压缩并行化方法](http://www.xml-data.org/JSJYY/2017-5-1282.htm)
