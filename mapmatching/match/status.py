from enum import IntEnum

class STATUS:
    SUCCESS       = 0 # 成功匹配
    SAME_LINK     = 1 # 所有轨迹点位于同一条线上
    ONE_POINT     = 2 # 所有轨迹点位于同一个点上
    NO_CANDIDATES = 3 # 轨迹点无法映射到候选边上
    FAILED        = 4 # 匹配结果，prob低于阈值
    UNKNOWN       = 99

class CANDS_EDGE_TYPE:
    NORMAL = 0         # od 不一样
    SAME_SRC_FIRST = 1 # od 位于同一条edge上，但起点相对终点位置偏前
    SAME_SRC_LAST  = 2 # 相对偏后
