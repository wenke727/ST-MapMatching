# Changelog

## [Unreleased]

- [ ] 聚光搜索, 流程梳理
- [ ] 后处理的流程加速，`transform_node_seq_to_polyline`

## [V1.1.9] - 2022-10-31

### Added

- `GeoDigraph`
  - `transform_node_seq_to_edge_seq`，通过 eid 快速访问edge，加快还原轨迹的速度
  - `transform_edge_seq_to_polyline`
- candidatesGraph
  - 新增`same_edge`的判断


### Changed

- `GeoDigraph` 
  - search 函数，增加 `eid_list` 和`geometry`属性
  - get_edge, get_node, _get_feature函数，增加 reset_index 控制
- `postprocess`
  - 将原来 edge的访问索引`(src, dst)` 更改为 `eid`

- `spatialAnalysis`
  - _trans_prob, 新增(o, d) 位于同一 edge 上的处理方式


### Deprecated

- `GeoDigraph`
  - transform_node_seq_to_polyline
  - transform_node_seq_to_df_edge


## [V1.1.8] - 2022-10-27

### Changed

- `project_point_to_line_segment`接口修改，输入为`points, edges, keep_cols`
- `postprocess`轨迹还原处理模块重构
- `parse_xml_to_graph`增加 index 和 eid 顺序一致性的判断

### Fixed

- `sindex.query`查询问题修正，index 的值为 item 在数据框中的序号，而非其真正索引的 `index`

## [V1.1.7] - 2022-10-24

### Added

- `db_process` 增加 PostgreSQL 支持

### Changed

- `ST-MapMatching` 重构，将各个模块的功能进行拆分

  ```log
  ├── __int__.py
  ├── candidatesGraph.py
  ├── geometricAnalysis.py
  ├── postprocess.py
  ├── spatialAnalysis.py
  ├── temporalAnalysis.py
  ├── topologicalAnalysis.py
  ├── visualization.py
  └── viterbi.py
  ```

- astar 的 memo 经测试，耗时可以降低 2 个数量级
- 统一 `parse_osm_xml`的 df_edges `eid` 和 `index`值 

## [V1.1.6] - 2022-10-19

### Changed

- 将 `spatial_analysis`相关部分从 ST_Matching 中抽离


## [V1.1.5] - 2022-10-19

### Changed

- 将 `get_k_neigbor_edges`(即原 get_candidates) 从 ST_Matching 中抽象出来
  - 针对一般场景，耗时在 `十毫秒` 级别
  - sindex.query(bbox, predicate='intersects'), 经过测试，`feature` 在 `bbox` 内部/相交 都是支持的，详见 `./test/0_shapely_空间关系.ipynb`

- `ST_Matching` 适配 `get_k_neigbor_candidates` 算法


## [V1.1.4] - 2022-10-18

### Added

- `GeoDigraph` 增加 对象序列化功能

  - save_checkpoint
  - load_checkpoint

- `osmnet` 模块

  - 新增 `build_graph` 模块，通过 osm 文件快速获得 路网的图信息

- `utils` 中 `serialization` 子模块

  新增 `save_checkpoint`， `load_checkpoint` 

### Changed

- `matching` 适配新路网格式，主要是边 的起点 `(s, e)` -> `(src, dst)`


## [V1.1.4] - 2022-10-13

### Changed

- 将原来 DigraphOSM 分为 `OSM 预处理的部分` 和 `图论算法部分`
  - 其中图论部分抽象为 `GeoDigraph`
  - OSM预处理模块按照模块拆分至 `osmnet `模块
- `graph`模块：将图论相关模块抽象
  - `base.py`: Node, Digraph
  - `geograph.py`: 原来 DigraphOSM 图论通用模块下沉至本模块 
  - `bfs`：A* 算法
- `osmnet`模块
  - osm_io: 数据加载模块
  - parse_osm_xml: 修改 edge 和 node 提取的属性
  - twoway_edge: 针对识别双向道路，并进行一定程度的偏移

## [V1.1.3] - 2022-10-10

### Added

- 新增`osmnet` 模块

  - combine_edges：将和 edge 合并相关的代码抽象合并在一起

    为了提升最短路算法搜索的步长，将节点的入度和出度均为1的边和上下游合并

    使用默认的 way 的方式会导致 graph 不齐全，`(o=7959990710, d=499265789)` :arrow_left: 拆分为最小单元，然后再次合并

  - downloader：osm xml 下载模块

  - parse_osm_xml

    使用 pyosmium 模块解析 OSM 文件，加快提取 graph 的处理速度



## [V1.1.2] - 2022-10-08

### Added

- 增加基于瓦片编号规则的 `fishnet`

### Fixed

- DigraphOSM
  - 考虑 节点数据 没有 `traffic_signals`情况
- parallel_process
  - 支持 tuple 作为 输入
  - total 控制进度条的长度


## [v1.1.1] - 2022-05-09

- 增加测试轨迹输入，存储路径为 `./input`
- MapMathching
  matching 增加仅有一个 candidate points 的边界条件
- azimuth_diff
  修复角度夹角大于 180 时计算错误的情况
- point_to_polyline_process
  返回值增加`点`到`折线`的`距离`

## [reconsitution] - 2022-05-09

- 封装`DigraphOSM` 和 `ST_Matching`
- geo_helper 增加 `point_to_polyline_process`
- osm_helper 增加 `download_osm_xml`, `parse_xml_to_topo`, `combine_links_parallel_helper`

## [V1.1] - 2022-01-25

- 修复预测bug

## [1.0.11] - 2021-12-23

### Changed

- matching.py
  - find_matched_sequence
    - 优化针对某些场景，gt下标出错的情景
    - f_score中查找最大值优化

- setting.py
  - filters 中去掉 service（相当于连接线）

## [1.0.10] - 2021-09-23

### Changed

- `DigraphOSM.py`
  - net.df_edges: change `index` -> `eid`
  - to_csv

## [1.0.09] - 2021-09-23

### Added

- `azimuth_helper.py`
  - azimuth_cos_similarity_for_linestring(geom, dir, weight=True)

### Changed

- `matching.py`
  - add log for candidates

## [1.0.08] - 2021-09-13

### Changed

- `matching.py`
  - `matching_debug_subplot` add transition probability
- `douglasPeucker.py`
  - `dfs` change the passes parameter from `array` to `subscript`
    - to cut down the continuity point and raise the compression rate.
  - `dp_compress_for_points` for trajectory points compression.
- `azimuthAngle`
  - fix speail case: [(0, 0), (0,1)]

## [1.0.07] - 2021-09-08

### Changed

- DigraphOSM
  - upload_topo_data_to_db: upload `indegree` and `outdegree` osm point version to db
- matching
  - move gba trajectory tesing to `matching_testing.py`
  - `find_matched_sequence` delete the `drop_dulplicates` part, for it would ignore a choice beteen levels.
  - `get_candidates` add `shrink` parameter to check candidates in consecutive levels is the same or not, and then delete the dulplicate level.
  - `combine_link_and_path` add parameter `net`

## [1.0.06] - 2021-07-27

### Changed

- matching.py
  - cal_trans_prob
    - Adjust the caculate functions, + 'np.abs'
    - adjust the ratio, when the ratio is bigger than 1, than `factor = 1/ factor`
      ![case0](./.fig/bug_v_bigger_than_1_case0.png)
    - Add azimuth to transmission probability.
      ![azimuth](./.fig/bug2.jpg)

## [1.0.05] - 2021-07-27

### Added

- Digraph_OSM
  - Topo edge data: add `dir`

### Changed

- matching.py
  - get_candidates
    take `dir` into accounts.
  - cal_relative_offset
    Change calculation logic
    - caculate the closest line
    - caculate the accumulative distance from the start point to the foot point
  - cal_trans_prob
    `_od_in_same_link`: Analyzed the cases that the cotinuos candidates are located on the same link

## [1.0.04] - 2021-07-24

### Added

- Digraph_OSM
  - node_sequence_to_edge

### Changed

- `mathing.py`
  - linestring_combine_helper
    In `cal_trans_prob`, there is no need to obtain geometry.
  - get_path
    Reconstruct path according to the resualt of the  `find_matched_sequence`.
  - the find best sequence: add the observ probility
  - matching_debug_subplot
    add observ prob, v, f details.

## [1.0.03] - 2021-07-22

### Added

- Douglas-Peucker / trajectory segmentation (Batched Compression Techniques)
- load path_set of direction API for matching testing.
- `mathing.py`
  - add visulization
  - completed the pano matching frame
- `geo_helper.py`
  - foot related functions
  - edge_parallel_offset
    - Add judgment conditions: `is_ring`
  - io related functions: load/upload postgis
- `geo_plot_helper.py`
  - adaptive_zoom_level
- `log_helper.py`
  - log to file/std

### Changed

- DigraphOSM
  - Encapsulated it as a separated module.
  - `add_reverse_edge` add `is_ring` judgement condition to determine add the reverse edge or not.

## [1.0.02] - 2021-07-19

### Added

- Digraph_OSM
  - download_map from the Internet
  - way_filters['auto']
  - add_reverse_edge: add reverse edge of two-way road. The order of reverse edge is [-1, -2, .., -n-1], where n in the number of line segments.
  - A star 增加终止条件; Add `max_layer` parameter to control the deepest level.

- PickleSaver
  - Save the object to file, and load the object from the file.

- setting.py
  - Define `way_filters['auto']`

### Changed

- matching.py
  - cal_relative_offset
    Update the calculation process
  - st_matching
    plot with more details

## [1.0.01] - 2021-07-15

### Added

- Digraph_OSM
  - 创建类，实现自动从OSM解析路网
  - 合并出入度都为1的edge

- matching.py
  - 实现空间匹配算法
  - 路网匹配，可视化debug

- coordTransfrom_shp
  实现geopandas的坐标转换

- foot_helper
  和垂足相关的脚本
