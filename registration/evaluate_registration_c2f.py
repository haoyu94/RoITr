import os, sys, glob, torch, argparse
cwd = os.getcwd()
sys.path.append(cwd)
import numpy as np
from lib.utils import setup_seed, natural_key
from tqdm import tqdm
from registration.benchmark_utils import ransac_pose_estimation_correspondences, get_inlier_ratio_correspondence, get_scene_split, write_est_trajectory
from registration.benchmark import benchmark
from lib.utils import square_distance
from visualizer.visualizer import Visualizer
from visualizer.plot import draw_distance_geo_feat
from dataset.common import collect_local_neighbors, get_square_distance_matrix, point2node_sampling
from lib.utils import weighted_procrustes

setup_seed(0)


def extract_correspondence(dist, major='row'):
    if major == 'row':
        top2 = np.partition(dist, axis=1, kth=1)[:, :2]
        row_inds = np.arange(dist.shape[0])
        d0 = top2[:, 0]
        d1 = top2[:, 1]
        col_inds = np.argmin(dist, axis=1)
        nn = np.where(d0 < d1, d0, d1)
        nn2 = np.where(d0 > d1, d0, d1)
        weights = -nn
    elif major == 'col':
        top2 = np.partition(dist, axis=0, kth=1)[:2, :]
        col_inds = np.arange(dist.shape[1])
        d0 = top2[0, :]
        d1 = top2[1, :]
        row_inds = np.argmin(dist, axis=0)
        nn = np.where(d0 < d1, d0, d1)
        nn2 = np.where(d0 > d1, d0, d1)
        weights = -nn
    else:
        raise NotImplementedError

    return row_inds, col_inds, weights



def benchmark_registration(desc, exp_dir, whichbenchmark, n_points, ransac_with_mutual=False, inlier_ratio_threshold=0.05):
    gt_folder = f'configs/benchmarks/{whichbenchmark}'
    exp_dir = f'{exp_dir}/{whichbenchmark}/{n_points}'
    if (not os.path.exists(exp_dir)):
        os.makedirs(exp_dir)

    results = dict()
    results['w_mutual'] = {'inlier_ratios': [], 'distances': []}
    results['wo_mutual'] = {'inlier_ratios': [], 'distances': []}
    tsfm_est = []
    inlier_ratio_list = []

    coarse_sample = 256
    idx = 0
    for eachfile in tqdm(desc):

        #if idx < 1320:
        #    idx += 1
        #    continue
        #else:
        #    idx += 1
        ######################################################
        # 1. take the nodes and descriptors
        print(eachfile)
        data = torch.load(eachfile)
        src_pcd, tgt_pcd = data['src_pcd'], data['tgt_pcd']
        src_nodes, tgt_nodes = data['src_nodes'], data['tgt_nodes']
        src_feats, tgt_feats = data['src_node_desc'], data['tgt_node_desc']
        src_point_feats, tgt_point_feats = data['src_point_desc'], data['tgt_point_desc']
        rot, trans = data['rot'], data['trans']
        src_corr_pts, tgt_corr_pts = data['src_corr_pts'], data['tgt_corr_pts']
        confidence = data['confidence']
        ######################################################
        # 2. run ransac
        prob = confidence / torch.sum(confidence)
        print(confidence.shape[0])
        if prob.shape[0] > n_points:
            sel_idx = np.random.choice(prob.shape[0], n_points, replace=False, p=prob.numpy())
            #mute the previous line and unmute the following line for changing the sampling strategy to top-k
            #sel_idx = torch.topk(confidence, k=n_points)[1]
            src_corr_pts, tgt_corr_pts = src_corr_pts[sel_idx], tgt_corr_pts[sel_idx]
            confidence = confidence[sel_idx]

        correspondences = torch.from_numpy(np.arange(src_corr_pts.shape[0])[:, np.newaxis]).expand(-1, 2)
        tsfm_est.append(ransac_pose_estimation_correspondences(src_corr_pts, tgt_corr_pts, correspondences))
        ######################################################
        # 3. calculate inlier ratios
        cur_inlier_ratio = get_inlier_ratio_correspondence(src_corr_pts, tgt_corr_pts, rot, trans, inlier_distance_threshold=0.1)
        inlier_ratio_list.append(cur_inlier_ratio)
        idx += 1

    tsfm_est = np.array(tsfm_est)

    ########################################
    # wirte the estimated trajectories
    write_est_trajectory(gt_folder, exp_dir, tsfm_est)

    ########################################
    # evaluate the results, here FMR and Inlier ratios are all average twice
    inlier_ratio_list = np.array(inlier_ratio_list)
    benchmark(exp_dir, gt_folder)
    split = get_scene_split(whichbenchmark)

    inliers = []
    fmrs = []
    inlier_ratio_thres = 0.05
    for ele in split:
        c_inliers = inlier_ratio_list[ele[0]:ele[1]]
        inliers.append(np.mean(c_inliers))
        fmrs.append((np.array(c_inliers) > inlier_ratio_thres).mean())
    with open(os.path.join(exp_dir, 'result'), 'a') as f:
        f.write(f'Inlier ratio: {np.mean(inliers):.3f} : +- {np.std(inliers):.3f}\n')
        f.write(f'Feature match recall: {np.mean(fmrs):.3f} : +- {np.std(fmrs):.3f}\n')

    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', default=None, type=str, help='path to precomputed features and scores')
    parser.add_argument('--benchmark', default='3DLoMatch', type=str, help='[3DMatch, 3DLoMatch]')
    parser.add_argument('--n_points', default=1000, type=int, help='number of points used by RANSAC')
    parser.add_argument('--exp_dir', default='est_traj', type=str, help='export final results')
    args = parser.parse_args()
    desc = sorted(glob.glob(f'{args.source_path}/*.pth'), key=natural_key)
    benchmark_registration(desc, args.exp_dir, args.benchmark, args.n_points, ransac_with_mutual=False)

