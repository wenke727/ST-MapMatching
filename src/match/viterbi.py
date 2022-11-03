import imp
import numpy as np
import geopandas as gpd
from loguru import logger


def find_matched_sequence(cands:gpd.GeoDataFrame, graph:gpd.GeoDataFrame):
    prev_dict, f_score = {}, {}
    layer_ids = graph.index.get_level_values(0).unique().sort_values().values

    for i, item in cands.query(f"pid=={layer_ids[0]}").iterrows():
        f_score[i] = item.observ_prob

    for cur_layer_id in layer_ids:
        df_layer     = graph.loc[cur_layer_id]
        nxt_layer_id = df_layer.pid_1.unique()[0]
        cur_eids     = cands.query(f"pid == {cur_layer_id}").eid
        nxt_eids     = cands.query(f"pid == {nxt_layer_id}").eid

        for j, nxt in nxt_eids.iteritems():
            _max = -np.inf
            for i, cur in cur_eids.iteritems():
                info = df_layer.loc[cur].loc[nxt].to_dict()
                _f = info['f']
                if _f > 1.001:
                    logger.warning(f"level {i}->{j}({cur}, {nxt}), F value {_f:.3f}, exceed 1.\n\t{info}")
                    
                alt = f_score[i] + _f
                if alt > _max:
                    _max = alt
                    prev_dict[j] = i
                f_score[j] = _max
            
    rList = []
    c = max(f_score, key=lambda x: (f_score.get(x), x))
    for i in range(len(layer_ids), 0, -1):
        rList.append(c)
        c = prev_dict[c]
    rList.append(c)
    rList = cands.loc[rList[::-1]][['pid', 'eid', 'src', 'dst']]

    logger.info(f'max score: {c}, f_score: {f_score}\n{rList}')
    
    # [8951, 1407, 1483, 118096, 117450, 23786, 123310, 123317, 658, 1581, 139265]
    return rList


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
    
    # tmp = gt[['d_euc', 'first_step_len', 'last_step_len', 'cost', 'path', 'geometry', 'v', 'first_step', 'last_step', 'move_dir', 'f_dir']]
    
    return states, observations, start_prob, trans_prob, emit_prob
        

def process_viterbi_pipeline(cands, gt):
    states, observations, start_prob, trans_prob, emit_prob = prepare_viterbi_input(cands, gt)
    _, rList = decode(observations, states, start_prob, trans_prob, emit_prob)

    rList = cands.set_index(['pid', 'eid']).loc[rList][[ 'src', 'dst']].reset_index()
    
    return rList


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
