import geopandas as gpd
from .matching import ST_Matching


def plot_topo_helper(seg, pos, neg, matcher):
    fig, ax = matcher.plot_result(seg, pos)
    neg_path = matcher.transform_res_2_path(neg)
    neg_path.plot(ax=ax, color='b', label = 'revert', linestyle=':')
    ax.legend()

    return

def check_each_step(matcher:ST_Matching, traj:gpd.GeoDataFrame, idx:int, factor=1.2, verbose=True):
    seg = traj.iloc[idx: idx + 2]
    net = matcher.net

    pos = matcher.matching(seg.reset_index(drop=True))
    neg = matcher.matching(seg[::-1].reset_index(drop=True))


    if neg['probs']['prob'] > pos['probs']['prob'] * factor:
        eids = neg['epath']
        way_ids = net.get_edge(eids, 'way_id').unique()

        for idx in way_ids:
            # TODO 上游确定是单向的，且仅有一个的情况下才增加
            status = net.add_reverse_way(way_id=idx)
            if status:
                print(f"add {idx}")
            else:
                print(f"{idx} exist")

        if verbose:
            plot_topo_helper(seg, pos, neg, matcher)

def check_steps(matcher, res, prob_thred=.75, factor=1.2):
    traj = res['details']['simplified_traj']
    steps = res['details']['steps']

    cand_steps = steps.query(f'trans_prob < {prob_thred}')
    for i, item in cand_steps.iterrows():
        check_each_step(matcher, traj, i, factor)

