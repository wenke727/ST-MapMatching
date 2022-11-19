
def merge_intervals(intervals):
    res = []
    for i in intervals:
        merge_intervals_helper(res, i[0], i[1])
        
    return res


def merge_intervals_helper(intervals, start, end, height=None):
    """merge intervals

    Args:
        intervals ([type]): [description]
        start ([type]): [description]
        end ([type]): [description]
        height ([type], optional): [description]. Defaults to None.
    """
    if start is None or height ==0 or start == end: 
        return 

    if not intervals:
        intervals.append( [start, end, height] )
        return
    
    _, prev_end, prev_height = intervals[-1]
    if prev_height == height and prev_end == start:
        intervals[-1][1] = end

        return  
    intervals.append([start, end, height])


def insert_intervals(intervals, newInterval):
    res = []
    insertPos = 0
    newInterval = newInterval.copy()
    for interval in intervals:
        if interval[1] < newInterval[0]:
            res.append(interval)
            insertPos += 1
        elif interval[0] > newInterval[1]:
            res.append(interval)
        else:
            newInterval[0] = min(interval[0], newInterval[0])
            newInterval[1] = max(interval[1], newInterval[1])
            newInterval[2] = interval[2]
    
    res.insert(insertPos, newInterval)

    return res

