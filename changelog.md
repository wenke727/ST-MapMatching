# Changelog

## [Unreleased]

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
