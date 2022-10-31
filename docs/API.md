# API 设计文档

## 地图匹配模块

|            模块             |             函数              | 输入                                                         | 输出         | 说明                                                         |
| :-------------------------: | :---------------------------: | :----------------------------------------------------------- | ------------ | ------------------------------------------------------------ |
|   *candidate<br />Graph*    |        construct_graph        | cands<br />common_attrs<br />left_attrs<br />right_attrs<br />rename_dict | gt           | Construct the candiadte graph (level, src, dst) for spatial and temporal analysis.<br />针对 od 落在同一个 edge 上时，将 od 对调 |
|    *geometric Analysis*     |       _filter_candidate       | df_candidates<br />top_k<br />pid=‘eid’<br />edge_keys       | df_cands     | 过滤cands<br />1 按照距离顺序排序，并针对每一个道路保留最近的一个路段<br />2 针对每一个节点，保留 top_k 个记录 |
|                             |      get_k_neigbor_edges      | points<br />edges<br />top_k<br />radius<br />               | df_cands     | [sindex.query_bulk](https://geopandas.org/en/stable/docs/reference/api/geopandas.sindex.SpatialIndex.query_bulk.html#geopandas.sindex.SpatialIndex.query_bulk)，返回的是tree geom 的 整数 index |
|                             |        cal_observ_prob        | dist<br />bias<br />deviation<br />normal=True               | observe_prob | 正态分布                                                     |
|                             | project_point_to_line_segment | points<br />edges<br />keeps_colsFai                         |              |                                                              |
|                             |    analyse_geometric_info     |                                                              |              |                                                              |
|   *spatial<br />Analysis*   |       cal_traj_distance       |                                                              |              |                                                              |
|                             |     _move_dir_similarity      |                                                              |              |                                                              |
|                             |          _trans_prob          |                                                              |              |                                                              |
|                             |     analyse_spatial_info      |                                                              |              |                                                              |
| *topological<br />Analysis* |              --               |                                                              |              |                                                              |
|  *temporal<br />Analysis*   |        cos_similarity         |                                                              |              |                                                              |
|          *viterbi*          |   process_viterbi_pipeline    |                                                              |              |                                                              |
|        *postprocess*        |           get_path            |                                                              |              |                                                              |
|                             |         get_one_step          |                                                              |              |                                                              |
|                             |        get_connectors         |                                                              |              |                                                              |
|       *visualization*       |         plot_matching         |                                                              |              |                                                              |
|                             |     matching_debug_level      |                                                              |              |                                                              |
|                             |    matching_debug_subplot     |                                                              |              |                                                              |
|                             |                               |                                                              |              |                                                              |

