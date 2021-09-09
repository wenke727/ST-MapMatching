from tqdm import tqdm
from load_step import df_path, split_line_to_points
from matching import st_matching, get_candidates, cal_observ_prob, cal_trans_prob, find_matched_sequence, get_path


def code_testing_single(id, net, debuf_with_step=False):
    traj = split_line_to_points(df_path.iloc[id].geometry, compress=True, config={'dist_max': 8, 'verbose': True})

    if not debuf_with_step:
        st_matching(traj, net, plot=True, save_fn=id)
    else:
        # step 1: candidate prepararation
        df_candidates = get_candidates(traj, net.df_edges, georadius=50, plot=True, verbose=False)

        # step 2.1: Spatial analysis, obervation prob
        observ_prob_dict = cal_observ_prob(df_candidates)

        # step 2.2: Spatial analysis, transmission prob
        tList, gt, graph_t = cal_trans_prob(df_candidates, net)


        # step 4: find matched sequence
        rList = find_matched_sequence(gt, df_candidates, tList)
        path = get_path(rList, gt, net, True)

        graph_t

    return 


def code_testing(net, start=0, end=100, debuf_with_step=False):
    for id in tqdm( range(start, end) ):
        traj = split_line_to_points(df_path.iloc[id].geometry, compress=True, config={'dist_max': 8, 'verbose': True})
        _ = st_matching(traj, net, plot=True, save_fn=f"{id:03d}", satellite=True)


