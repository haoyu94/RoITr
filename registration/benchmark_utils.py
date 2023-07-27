import os, torch, glob
import numpy as np
import open3d as o3d
from lib.utils import to_o3d_pcd
from registration.benchmark import read_trajectory, write_trajectory


def to_array(tensor):
    """
    Conver tensor to array
    """
    if(not isinstance(tensor,np.ndarray)):
        if(tensor.device == torch.device('cpu')):
            return tensor.numpy()
        else:
            return tensor.cpu().numpy()
    else:
        return tensor


def to_tensor(array):
    """
    Convert array to tensor
    """
    if(not isinstance(array,torch.Tensor)):
        return torch.from_numpy(array).float()
    else:
        return array


def to_o3d_feats(embedding):
    """
    Convert tensor/array to open3d features
    embedding:  [N, 3]
    """
    feats = o3d.registration.Feature()
    feats.data = to_array(embedding).T
    return feats



def mutual_selection(score_mat):
    """
    Return a {0,1} matrix, the element is 1 if and only if it's maximum along both row and column

    Args: np.array()
        score_mat:  [B,N,N]
    Return:
        mutuals:    [B,N,N]
    """
    score_mat = to_array(score_mat)
    if (score_mat.ndim == 2):
        score_mat = score_mat[None, :, :]

    mutuals = np.zeros_like(score_mat)
    for i in range(score_mat.shape[0]):  # loop through the batch
        c_mat = score_mat[i]
        flag_row = np.zeros_like(c_mat)
        flag_column = np.zeros_like(c_mat)

        max_along_row = np.argmax(c_mat, 1)[:, None]
        max_along_column = np.argmax(c_mat, 0)[None, :]
        np.put_along_axis(flag_row, max_along_row, 1, 1)
        np.put_along_axis(flag_column, max_along_column, 1, 0)
        mutuals[i] = (flag_row.astype(np.bool)) & (flag_column.astype(np.bool))
    return mutuals.astype(np.bool)


def get_inlier_ratio_correspondence(src_node, tgt_node, rot, trans, inlier_distance_threshold=0.1):
    '''
    Compute inlier ratios based on input torch tensors
    '''
    src_node = torch.matmul(src_node, rot.T) + trans.T
    dist = torch.norm(src_node - tgt_node, dim=-1)
    inliers = dist < inlier_distance_threshold
    inliers_num = torch.sum(inliers)
    return inliers_num / src_node.shape[0]


def get_inlier_ratio(src_pcd, tgt_pcd, src_feat, tgt_feat, rot, trans, inlier_distance_threshold=0.1):
    """
    Compute inlier ratios with and without mutual check, return both
    """
    src_pcd = to_tensor(src_pcd)
    tgt_pcd = to_tensor(tgt_pcd)
    src_feat = to_tensor(src_feat)
    tgt_feat = to_tensor(tgt_feat)
    rot, trans = to_tensor(rot), to_tensor(trans)

    results =dict()
    results['w']=dict()
    results['wo']=dict()

    if(torch.cuda.device_count()>=1):
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    src_pcd = (torch.matmul(rot, src_pcd.transpose(0,1)) + trans).transpose(0,1)
    scores = torch.matmul(src_feat.to(device), tgt_feat.transpose(0,1).to(device)).cpu()

    ########################################
    # 1. calculate inlier ratios wo mutual check
    _, idx = scores.max(-1)
    dist = torch.norm(src_pcd- tgt_pcd[idx],dim=1)
    results['wo']['distance'] = dist.numpy()

    c_inlier_ratio = (dist < inlier_distance_threshold).float().mean()
    results['wo']['inlier_ratio'] = c_inlier_ratio

    ########################################
    # 2. calculate inlier ratios w mutual check
    selection = mutual_selection(scores[None,:,:])[0]
    row_sel, col_sel = np.where(selection)
    dist = torch.norm(src_pcd[row_sel]- tgt_pcd[col_sel],dim=1)
    results['w']['distance'] = dist.numpy()

    c_inlier_ratio = (dist < inlier_distance_threshold).float().mean()
    results['w']['inlier_ratio'] = c_inlier_ratio

    return results


def ransac_pose_estimation(src_pcd, tgt_pcd, src_feat, tgt_feat, mutual=False, distance_threshold=0.05, ransac_n=3):
    """
    RANSAC pose estimation with two checkers
    We follow D3Feat to set ransac_n = 3 for 3DMatch and ransac_n = 4 for KITTI.
    For 3DMatch dataset, we observe significant improvement after changing ransac_n from 4 to 3.
    """
    if (mutual):
        if (torch.cuda.device_count() >= 1):
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        src_feat, tgt_feat = to_tensor(src_feat), to_tensor(tgt_feat)
        scores = torch.matmul(src_feat.to(device), tgt_feat.transpose(0, 1).to(device)).cpu()
        selection = mutual_selection(scores[None, :, :])[0]
        row_sel, col_sel = np.where(selection)
        corrs = o3d.utility.Vector2iVector(np.array([row_sel, col_sel]).T)
        src_pcd = to_o3d_pcd(src_pcd)
        tgt_pcd = to_o3d_pcd(tgt_pcd)
        result_ransac = o3d.registration.registration_ransac_based_on_correspondence(
            source=src_pcd, target=tgt_pcd, corres=corrs,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            criteria=o3d.registration.RANSACConvergenceCriteria(50000, 1000))
    else:
        src_pcd = to_o3d_pcd(src_pcd)
        tgt_pcd = to_o3d_pcd(tgt_pcd)
        src_feats = to_o3d_feats(src_feat)
        tgt_feats = to_o3d_feats(tgt_feat)

        result_ransac = o3d.registration.registration_ransac_based_on_feature_matching(
            src_pcd, tgt_pcd, src_feats, tgt_feats, distance_threshold,
            o3d.registration.TransformationEstimationPointToPoint(False), ransac_n,
            [o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
             o3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
            o3d.registration.RANSACConvergenceCriteria(50000, 1000))

    return result_ransac.transformation



def ransac_pose_estimation_correspondences(src_pcd, tgt_pcd, correspondences, mutual=False, distance_threshold=0.05,
                                           ransac_n=3):
    '''
    Run RANSAC estimation based on input correspondences
    :param src_pcd:
    :param tgt_pcd:
    :param correspondences:
    :param mutual:
    :param distance_threshold:
    :param ransac_n:
    :return:
    '''

    # ransac_n = correspondences.shape[0]

    if mutual:
        raise NotImplementedError
    else:
        # src_pcd = src_pcd.cuda()
        # tgt_pcd = tgt_pcd.cuda()
        # correspondences = correspondences.cuda()
        src_pcd = to_o3d_pcd(to_array(src_pcd))
        tgt_pcd = to_o3d_pcd(to_array(tgt_pcd))
        correspondences = o3d.utility.Vector2iVector(to_array(correspondences))

        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_correspondence(src_pcd, tgt_pcd,
                                                                                               correspondences,
                                                                                               distance_threshold,
                                                                                               o3d.pipelines.registration.TransformationEstimationPointToPoint(
                                                                                                   False), ransac_n, [
                                                                                                   o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                                                                                                       0.9),
                                                                                                   o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                                                                                                       distance_threshold)],
                                                                                               o3d.pipelines.registration.RANSACConvergenceCriteria(
                                                                                                   50000, 1000))
        '''
        result_ransac = o3d.registration.registration_ransac_based_on_correspondence(
            source=src_pcd, target=tgt_pcd, corres=correspondences,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            criteria=o3d.registration.RANSACConvergenceCriteria(50000, 1000))
        '''
    return result_ransac.transformation


def get_scene_split(whichbenchmark):
    """
    Just to check how many valid fragments each scene has
    """
    assert whichbenchmark in ['3DMatch','3DLoMatch']
    folder = f'configs/benchmarks/{whichbenchmark}/*/gt.log'

    scene_files=sorted(glob.glob(folder))
    split=[]
    count=0
    for eachfile in scene_files:
        gt_pairs, gt_traj = read_trajectory(eachfile)
        split.append([count,count+len(gt_pairs)])
        count+=len(gt_pairs)
    return split


def write_est_trajectory(gt_folder, exp_dir, tsfm_est):
    """
    Write the estimated trajectories
    """
    scene_names=sorted(os.listdir(gt_folder))
    count=0
    for scene_name in scene_names:
        gt_pairs, gt_traj = read_trajectory(os.path.join(gt_folder,scene_name,'gt.log'))
        est_traj = []
        for i in range(len(gt_pairs)):
            est_traj.append(tsfm_est[count])
            count+=1

        # write the trajectory
        c_directory=os.path.join(exp_dir,scene_name)
        if not os.path.exists(c_directory):
            os.makedirs(c_directory)
        write_trajectory(np.array(est_traj),gt_pairs,os.path.join(c_directory,'est.log'))

