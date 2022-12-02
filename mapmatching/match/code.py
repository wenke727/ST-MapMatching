from enum import IntEnum

class STATUS(IntEnum):
    SUCCESS       = 0 # 成功匹配
    SAME_LINK     = 1 # 所有轨迹点位于同一条线上
    ONE_POINT     = 2 # 所有轨迹点位于同一个点上
    NO_CANDIDATES = 3 # 轨迹点无法映射到候选边上
    FAILED        = 4 # 匹配结果，prob低于阈值
    UNKNOWN       = 99
