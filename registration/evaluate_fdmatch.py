import glob
from tqdm import tqdm
import torch
import numpy as np
import os, sys
cwd = os.getcwd()
sys.path.append(cwd)
from lib.utils import natural_key, square_distance


def partition_arg_topK(matrix, K, axis=0):
    """ find index of K smallest entries along a axis
    perform topK based on np.argpartition
    :param matrix: to be sorted
    :param K: select and sort the top K items
    :param axis: 0 or 1. dimension to be sorted.
    :return:
    """
    a_part = np.argpartition(matrix, K, axis=axis)
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        a_sec_argsort_K = np.argsort(matrix[a_part[0:K, :], row_index], axis=axis)
        return a_part[0:K, :][a_sec_argsort_K, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        a_sec_argsort_K = np.argsort(matrix[column_index, a_part[:, 0:K]], axis=axis)
        return a_part[:, 0:K][column_index, a_sec_argsort_K]


def knn_point_np(k, reference_pts, query_pts):
    '''
    :param k: number of k in k-nn search
    :param reference_pts: (N, 3) float32 array, input points
    :param query_pts: (M, 3) float32 array, query points
    :return:
        val: (batch_size, npoint, k) float32 array, L2 distances
        idx: (batch_size, npoint, k) int32 array, indices to input points
    '''

    N, _ = reference_pts.shape
    M, _ = query_pts.shape
    reference_pts = reference_pts.reshape(1, N, -1).repeat(M, axis=0)
    query_pts = query_pts.reshape(M, 1, -1).repeat(N, axis=1)
    dist = np.sum((reference_pts - query_pts) ** 2, -1)
    idx = partition_arg_topK(dist, K=k, axis=1)
    val = np.take_along_axis(dist, idx, axis=1)
    return np.sqrt(val), idx


def blend_anchor_motion (query_loc, reference_loc, reference_flow , knn=3, search_radius=0.1) :
    '''approximate flow on query points
    this function assume query points are sub- or un-sampled from reference locations
    @param query_loc:[m,3]
    @param reference_loc:[n,3]
    @param reference_flow:[n,3]
    @param knn:
    @return:
        blended_flow:[m,3]
    '''

    dists, idx = knn_point_np(knn, reference_loc, query_loc)
    dists[dists < 1e-10] = 1e-10
    mask = dists>search_radius
    dists[mask] = 1e+10
    weight = 1.0 / dists
    weight = weight / np.sum(weight, -1, keepdims=True)  # [B,N,3]
    blended_flow = np.sum (reference_flow [idx] * weight.reshape ([-1, knn, 1]), axis=1, keepdims=False)

    mask = mask.sum(axis=1)<3

    return blended_flow, mask


def compute_nrfmr(data, recall_thr=0.04):


    s_pcd, t_pcd = data['src_raw_pcd'], data['tgt_pcd']
    s_deformed_pcd = data['src_pcd']
    sflow_list = s_deformed_pcd - s_pcd
    metric_index_list = data['metric_index_list']

    rot = data['rot']  # B,3,3
    trans = data['trans']


    nrfmr = 0.


    # get the metric points' transformed position
    metric_index = metric_index_list
    sflow = sflow_list
    s_pcd_raw_i = s_pcd
    metric_pcd = s_pcd_raw_i[metric_index]
    metric_sflow = sflow[metric_index]
    metric_pcd_deformed = metric_pcd + metric_sflow
    metric_pcd_wrapped_gt = (torch.matmul(rot, metric_pcd_deformed.T) + trans).T


    # use the match prediction as the motion anchor
    s_pcd_matched = data['src_corr_pts']
    distance = square_distance(s_pcd_matched[None, ::], s_deformed_pcd[None, ::])[0]
    idx = torch.min(distance, dim=-1)[1]
    s_pcd_matched = s_pcd[idx]

    t_pcd_matched = data['tgt_corr_pts']
    motion_pred = t_pcd_matched - s_pcd_matched
    metric_motion_pred, valid_mask = blend_anchor_motion(
        metric_pcd.numpy(), s_pcd_matched.numpy(), motion_pred.numpy(), knn=3, search_radius=0.1)
    metric_pcd_wrapped_pred = metric_pcd + torch.from_numpy(metric_motion_pred).to(metric_pcd)

    dist = torch.sqrt(torch.sum((metric_pcd_wrapped_pred - metric_pcd_wrapped_gt)**2, dim=1))

    r = (dist < recall_thr).float().sum() / len(dist)

    return r



def test_thr():

    IR =0.
    NR_FMR =0.

    inlier_thr = recall_thr = 0.04
    n_sample = 0.
    desc = sorted(glob.glob('./snapshot/fdmatch_ripoint_transformer_test/4DLoMatch/*.pth'), key=natural_key)
    print(len(desc))
    id_list = []
    fmr_list = []
    ir_list = []
    num = 0
    for eachfile in tqdm(desc):
        print(eachfile)
        data = torch.load(eachfile)
        src_pcd, tgt_pcd = data['src_pcd'], data['tgt_pcd']
        src_raw_pcd = data['src_raw_pcd']
        src_feats, tgt_feats = data['src_point_desc'], data['tgt_point_desc']
        rot, trans = data['rot'], data['trans']
        src_corr_pts, tgt_corr_pts = data['src_corr_pts'], data['tgt_corr_pts']
        confidence = data['confidence']
        src_node, tgt_node = data['src_nodes'], data['tgt_nodes']
        src_node_desc, tgt_node_desc = data['src_node_desc'], data['tgt_node_desc']
        #n_points = 3000
        #prob = confidence / torch.sum(confidence)
        #if prob.shape[0] > n_points:
        #    sel_idx = np.random.choice(prob.shape[0], n_points, replace=False, p=prob.numpy())
            #sel_idx = torch.topk(confidence, k=n_points)[1]
        #    src_corr_pts, tgt_corr_pts = src_corr_pts[sel_idx], tgt_corr_pts[sel_idx]
        #    confidence = confidence[sel_idx]

        rot_src = torch.matmul(src_corr_pts, rot.T) + trans.T
        dist = torch.sqrt(torch.sum((rot_src - tgt_corr_pts) ** 2, dim=-1))
        ir = torch.sum(torch.lt(dist, inlier_thr).float()) / src_corr_pts.shape[0]


        nrfmr = compute_nrfmr(data, recall_thr=recall_thr)

        IR += ir
        NR_FMR += nrfmr

        n_sample += src_corr_pts.shape[0]

        if nrfmr < 0.05:
            id_list.append(eachfile.split('/')[-1].split('.')[0])
            fmr_list.append(nrfmr)
            ir_list.append(ir)
        num+=1

    IRate = IR / len(desc)
    NR_FMR = NR_FMR / len(desc)
    n_sample = n_sample / len(desc)
    return IRate, NR_FMR, n_sample


def evaluate():
        # for thr in [ 0.1 ]:
    import time
    start = time.time()
    ir, fmr, nspl = test_thr()
    print("NFMR:", fmr, " Inlier rate:", ir, "Number sample:", nspl)
    print("time costs:", time.time() - start)


if __name__ == '__main__':
    evaluate()
