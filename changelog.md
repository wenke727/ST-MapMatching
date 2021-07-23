# Changelog

## [Unreleased]

- bidirectional A* algm

## [1.0.03] - 2021-07-22

### Added

- trajectory_segmentation
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
