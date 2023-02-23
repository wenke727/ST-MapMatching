import heapq
import numpy as np
import pandas as pd
from loguru import logger
from collections import defaultdict

from .spatialAnalysis import get_trans_prob_bet_layers
from ..utils import Timer, timeit


def cal_prob_func(x, y, mode):
    if mode == '+':
        return x +  y
    elif mode == '*':
        return x *  y

def merge_k_heapq(arrays, count=100):
    queue, res = [], []
    eid_set = set()
    
    for eid, arr in arrays.items():
        if len(arr) == 0:
            continue
        heapq.heappush(queue, (arr[0][0], eid))
    
    while queue and count:
        _, eid = heapq.heappop(queue)
        prob, keys = heapq.heappop(arrays[eid])
        count -= 1
        
        if arrays[eid]:
            heapq.heappush(queue, (arrays[eid][0][0], eid))
        if eid not in eid_set:
            res.append(list(keys) + [-prob])
        eid_set.add(eid)
        
    return res

def prune_layer(df_layer, prune=True, trim_factor=.75, use_pandas=False):
    # prune -> pick the most likely one
    _max_prob = df_layer['prob'].max()
    
    if use_pandas:
        df = df_layer[['prob']].sort_values('prob', ascending=False)\
                               .head(100 if prune else 5)\
                               .query(f"prob > {_max_prob * trim_factor}")\
                               .groupby('eid_1')\
                               .head(1).reset_index()
    else:
        thred = _max_prob * trim_factor
        arrs = defaultdict(list)
        for row in df_layer[['prob']].itertuples():
            idx, prob = getattr(row, "Index"), getattr(row, "prob")
            if prob < thred:
                continue
            heapq.heappush(arrs[idx[2]], (-prob, idx))

        records = merge_k_heapq(arrs, 100 if prune else 5)   
        df = pd.DataFrame(records, columns=['pid_0', 'eid_0', 'eid_1', 'prob'])

    return df

def reconstruct_path(f_score, prev_path):
    epath = []
    state = None
    end_probs = []

    for idx in range(len(f_score) - 1, 0, -1):
        if state is None:
            state = get_max_state(f_score, idx)
            if state is None:
                continue
            end_probs.append(f_score[idx][state])

        cur = (idx, state)
        if idx not in prev_path or state not in prev_path[idx]:
            state = None
            continue
        prev = prev_path[idx].get(state)
        if not epath or cur != epath[-1]:
            epath.append(cur)
        epath.append(prev)
        state = prev[1]

    epath = epath[::-1]

    return epath, sum(end_probs) / len(end_probs)

def get_max_state(f_score, idx):
    f = f_score[idx]
    if len(f) == 0:
        return None
    return max(f, key=f.get)

def print_level(df_layer):
    f = lambda x: sorted(df_layer.index.get_level_values(x).unique())
    return f"{f(1)} -> {f(2)}"

def find_matched_sequence(cands, gt, net, dir_trans=True, mode='*', trim_factor=0.75, trim_layer=5, level='trace'):
    # Initialize
    times = []
    timer = Timer()

    gt_beam = []
    layer_ids = np.sort(cands.pid.unique())
    start_prob = cands.query("pid == 0").set_index('eid')['observ_prob'].to_dict()
    f_score = [start_prob]
    prev_path = defaultdict(dict)
    prev_path[0] = {st: None for st in start_prob}
    prev_states = list(start_prob.keys())

    for idx, lvl in enumerate(layer_ids[:-1]):
        df_layer = gt.query(f"pid_0 == @lvl and eid_0 in @prev_states")
        if df_layer.empty:
            print(f"Matching traj break at idx: {idx}, level:  {lvl}")
            df_layer = gt.query(f"pid_0 == @lvl")
            prev_probs = 0 if mode == '+' else 1
        else:
            prev_probs = np.array(
                [f_score[-1][i] for i in df_layer.index.get_level_values(1)])
        
        # timer.start()
        df_layer = get_trans_prob_bet_layers(df_layer, net, dir_trans)
        # ti mes.append(timer.stop())
        df_layer.loc[:, 'prob'] = cal_prob_func(prev_probs, df_layer.trans_prob * df_layer.observ_prob, mode)
        _df = prune_layer(df_layer, idx >= trim_layer, trim_factor)

        # post-process
        _df = _df.set_index('eid_1')
        for name, item in _df.iterrows():
            prev_path[idx + 1][name] = (idx, int(item.eid_0))
        prev_states = list(_df.index.unique())
        f_score.append(_df['prob'].to_dict())
        gt_beam.append(df_layer)

    # epath
    epath, end_prob = reconstruct_path(f_score, prev_path)
    epath = ((layer_ids[idx], eid) for idx, eid in epath)
    rList = cands.set_index(['pid', 'eid'])\
                 .loc[epath, ['src', 'dst']].reset_index()
    
    gt_beam = pd.concat(gt_beam)
    ratio = gt_beam.shape[0] / gt.shape[0]
    _log = f"Route planning time cost: {np.sum(times):.3f} s, trim ratio: {(1 - ratio) * 100:.1f} %"
    getattr(logger, level)(_log)
    
    return end_prob, rList, gt_beam


""" normal """
def viterbi_decode(nodes, trans):
    """
    Viterbi算法求最优路径
    其中 nodes.shape=[seq_len, num_labels],
        trans.shape=[num_labels, num_labels].
    """
    # 获得输入状态序列的长度，以及观察标签的个数
    seq_len, num_labels = len(nodes), len(trans)
    # 简单起见，先不考虑发射概率，直接用起始0时刻的分数
    scores = nodes[0].reshape((-1, 1)) # (num_labels, 1)
    
    paths = []
    # 递推求解上一时刻t-1到当前时刻t的最优
    for t in range(1, seq_len):
        # scores 表示起始0到t-1时刻的每个标签的最优分数
        scores_repeat = np.repeat(scores, num_labels, axis=1) # (num_labels, num_labels)
        
        # observe当前时刻t的每个标签的观测分数
        observe = nodes[t].reshape((1, -1)) # (1, num_labels)
        observe_repeat = np.repeat(observe, num_labels, axis=0) # (num_labels, num_labels)
        
        # 从t-1时刻到t时刻最优分数的计算，这里需要考虑转移分数trans
        M = scores_repeat + trans + observe_repeat
        
        # 寻找到t时刻的最优路径
        scores = np.max(M, axis=0).reshape((-1, 1))
        idxs = np.argmax(M, axis=0)
        
        # 路径保存
        paths.append(idxs.tolist())
        
    best_path = [0] * seq_len
    best_path[-1] = np.argmax(scores)
    
    # 最优路径回溯
    for i in range(seq_len-2, -1, -1):
        idx = best_path[i+1]
        best_path[i] = paths[i][idx]

def get_trans_prob(trans_prob, layer_id):
    return trans_prob[layer_id]

def decode(observations, states, start_prob, trans_prob, emit_prob, mode='+'):
    def _formula(x, y):
        if mode == '+':
            return x +  y
        elif mode == '*':
            return x *  y
            
    V = [{}]
    path = {}

    # Initialize
    for st in states:
        if st not in start_prob:
            continue
        V[0][st] = start_prob[st]
        path[st] = [(observations[0], st)]

    # Run Viterbi when t > 0
    for t in range(1, len(observations)):
        V.append({})
        newpath = {}

        for curr_st in states:
            paths_to_curr_st = []
            for prev_st in V[t-1]:
                _trans_prob = get_trans_prob(trans_prob, t-1)
                if (prev_st, curr_st) not in _trans_prob:
                    continue
                
                v = V[t-1][prev_st]
                _v = _trans_prob[(prev_st, curr_st)]
                _e = emit_prob[curr_st][observations[t]]
                paths_to_curr_st.append(( _formula(v, _v * _e), prev_st))
            
            if not paths_to_curr_st:
                continue
            
            cur_prob, prev_state = max(paths_to_curr_st)
            V[t][curr_st] = cur_prob
            newpath[curr_st] = path[prev_state] + [(observations[t], curr_st)]

        # No need to keep the old paths
        path = newpath

    prob, end_state = max([(V[-1][st], st) for st in states if st in V[-1]])
    
    return prob, path[end_state]

def prepare_viterbi_input(cands, gt):
    states = cands.eid.unique()
    observations = cands.pid.unique()
    start_prob = cands.query("pid == 0").set_index('eid')['observ_prob'].to_dict()
    # start_prob = {key:1 for key in start_prob}

    observ_dict = cands[['pid', 'eid', 'observ_prob']].set_index(['eid'])
    emit_prob = {i: observ_dict.loc[[i]].set_index('pid')['observ_prob'].to_dict() for i in states}

    # BUG cands 坐标不连续的问题, 莫非是中断
    trans_prob = [gt.loc[i]['f'].to_dict() for i in observations[:-1] ]
    
    return states, observations, start_prob, trans_prob, emit_prob
        
def process_viterbi_pipeline(cands, gt):
    states, observations, start_prob, trans_prob, emit_prob = prepare_viterbi_input(cands, gt)
    prob, rList = decode(observations, states, start_prob, trans_prob, emit_prob)

    rList = cands.set_index(['pid', 'eid']).loc[rList][[ 'src', 'dst']].reset_index()
    
    return prob, rList


if __name__ == "__main__":
    import sys
    sys.path.append('../')
    from utils.serialization import load_checkpoint
    
    fn = "../debug/traj_0_data_for_viterbi.pkl"
    fn = "../../debug/traj_1_data_for_viterbi.pkl"
    # fn = Path(__file__).parent / fn
    data = load_checkpoint(fn)

    cands = data['cands']
    gt = data['graph']
    rList = data['rList']

    res = process_viterbi_pipeline(cands, gt)
