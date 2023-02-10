import geopandas as gpd
from loguru import logger

def plot_topo_helper(seg, pos, neg, matcher):
    fig, ax = matcher.plot_result(seg, pos)
    neg_path = matcher.transform_res_2_path(neg)
    neg_path.plot(ax=ax, color='b', label = 'revert', linestyle=':')
    ax.legend()

    return

def check_each_step(matcher, traj:gpd.GeoDataFrame, idx:int, factor=1.2, plot=False):
    flag = False
    seg = traj.iloc[idx: idx + 2]
    net = matcher.net

    pos = matcher.matching(seg.reset_index(drop=True))
    neg = matcher.matching(seg[::-1].reset_index(drop=True))

    if neg['status'] == 1 and pos['status'] == 4 or\
        neg['probs']['prob'] > pos['probs']['prob'] * factor:
        eids = neg['epath']
        way_ids = net.get_edge(eids, 'way_id').unique()

        for idx in way_ids:
            # TODO 上游确定是单向的，且仅有一个的情况下才增加
            status = net.add_reverse_way(way_id=idx)
            if status:
                if not flag:
                    flag = True
                logger.info(f"add {idx}")
            else:
                logger.info(f"{idx} exist")
                pass

        if plot:
            plot_topo_helper(seg, pos, neg, matcher)
    
    return True

def check_steps(matcher, res, prob_thred=.75, factor=1.2):
    flag = True
    traj = res['details']['simplified_traj']
    steps = res['details']['steps']

    if steps is None:
        # FIXME 更精炼
        logger.warning("Steps is None")
        if res['status'] == 4:
            _res = matcher.matching(traj[::-1].reset_index(drop=True))
            if _res['status'] == 1:
                eids = _res['epath']
                way_ids = matcher.net.get_edge(eids, 'way_id').unique()

                for idx in way_ids:
                    # TODO 上游确定是单向的，且仅有一个的情况下才增加
                    status = matcher.net.add_reverse_way(way_id=idx)
                    if status:
                        if not flag:
                            flag = True
                        logger.info(f"add {idx}")
                    else:
                        logger.info(f"{idx} exist")
        
        return flag

    cand_steps = steps.query(f'trans_prob < {prob_thred}')
    for i, item in cand_steps.iterrows():
        flag |= check_each_step(matcher, traj, i, factor)

    return flag
