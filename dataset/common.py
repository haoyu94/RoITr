import pickle
import open3d as o3d
import numpy as np
import torch
from lib.utils import to_o3d_pcd
import open3d as o3d
import math, copy


def get_square_distance_matrix(src, tgt):
    '''
    Calculate the squared Euclidean distance matrix between two point sets
    :param src: source point cloud in numpy.ndarray of shape[N, 3]
    :param tgt: target point cloud in numpy.ndarray of shape[M, 3]
    :return: matrix distance
    '''

    distance = -2. * np.matmul(src, tgt.transpose())
    distance += np.sum(src ** 2, axis=-1, keepdims=False)[:, np.newaxis]
    distance += np.sum(tgt ** 2, axis=-1, keepdims=False)[np.newaxis, :]
    distance = np.clip(distance, a_min=1e-12, a_max=None)
    return distance


def get_batched_square_distance_matrix(src, tgt):
    '''
    Calculate the squared Euclidean distance matrix between two point sets
    :param src: source point cloud in numpy.ndarray of shape[B, N, 3]
    :param tgt: target point cloud in numpy.ndarray of shape[B, M, 3]
    :return: matrix distance
    '''

    distance = -2. * np.matmul(src, tgt.transpose(0, 2, 1))
    distance += np.sum(src ** 2, axis=-1, keepdims=False)[:, :, np.newaxis]
    distance += np.sum(tgt ** 2, axis=-1, keepdims=False)[:, np.newaxis, :]
    distance = np.clip(distance, a_min=1e-12, a_max=None)

    return distance


def load_info(path):
    '''
    Read a dictionary from a pickle file including the information of dataset
    :param path: path to the pickle file
    :return: loaded info
    '''
    with open(path, 'rb') as f:
        return pickle.load(f)

def collate_fn(list_data, config):
    '''
    Function used in pytorch dataloader. It mainly puts different data into a single dictionary
    :param list_data: data
    :param config: configuration
    :return: inputs: a dictionary contains all the input data
    '''
    batched_src_points_list = []
    batched_src_lengths_list = []
    batched_tgt_points_list = []
    batched_tgt_lengths_list = []

    batched_src_normals_list = []
    batched_tgt_normals_list = []

    batched_src_feat_list = []
    batched_tgt_feat_list = []

    batched_rot = []
    batched_trans = []

    batched_raw_src_pcd = []
    batched_metric_index = []

    for ind, (src_pcd, tgt_pcd, src_normals, tgt_normals, src_feats, tgt_feats, rot, trans, raw_src_pcd, metric_index) in enumerate(list_data):
        batched_src_points_list.append(src_pcd)
        batched_tgt_points_list.append(tgt_pcd)

        batched_src_normals_list.append(src_normals)
        batched_tgt_normals_list.append(tgt_normals)

        batched_src_feat_list.append(src_feats)
        batched_tgt_feat_list.append(tgt_feats)

        batched_src_lengths_list.append(src_pcd.shape[0])
        batched_tgt_lengths_list.append(tgt_pcd.shape[0])

        batched_rot.append(rot[np.newaxis, :])
        batched_trans.append(trans[np.newaxis, :])

        batched_raw_src_pcd.append(raw_src_pcd)
        batched_metric_index.append(metric_index)

    batched_src_points = torch.from_numpy(np.concatenate(batched_src_points_list, axis=0))
    batched_tgt_points = torch.from_numpy(np.concatenate(batched_tgt_points_list, axis=0))

    batched_src_normals = torch.from_numpy(np.concatenate(batched_src_normals_list, axis=0))
    batched_tgt_normals = torch.from_numpy(np.concatenate(batched_tgt_normals_list, axis=0))

    batched_src_feats = torch.from_numpy(np.concatenate(batched_src_feat_list, axis=0))
    batched_tgt_feats = torch.from_numpy(np.concatenate(batched_tgt_feat_list, axis=0))

    batched_rot = torch.from_numpy(np.concatenate(batched_rot, axis=0))
    batched_trans = torch.from_numpy(np.concatenate(batched_trans, axis=0))

    batched_raw_src_pcd = torch.from_numpy(np.concatenate(batched_raw_src_pcd, axis=0))
    if batched_metric_index[0] is None:
        batched_metric_index = None
    else:
        batched_metric_index = torch.from_numpy(np.concatenate(batched_metric_index, axis=0))

    ##################################
    # Return batch data
    ##################################
    inputs = {
        'src_points': batched_src_points,
        'src_normals': batched_src_normals,
        'tgt_points': batched_tgt_points,
        'tgt_normals': batched_tgt_normals,
        'src_feats': batched_src_feats,
        'tgt_feats': batched_tgt_feats,
        'rot': batched_rot,
        'trans': batched_trans,
        'raw_src_pcd': batched_raw_src_pcd,
        'metric_index': batched_metric_index
    }
    return inputs


def sample_gt_node_correspondence(overlap_map, corr_num, thres=0.1):
    '''
    Sample ground truth node correspondences for traingin
    :param: overlap_map: score map between nodes from a pair of point clouds in numpy.ndarray of shape [M1, M2]
    :param: corr_num: the number of required correspondences (M)
    :return: selected node indices from source and target point clouds
    '''
    overlap_mask = (overlap_map > thres).astype(np.float32)
    overlap_map_ = overlap_map * overlap_mask

    src_sel_inds, tgt_sel_inds = np.nonzero(overlap_map_)

    cur_corr_num = src_sel_inds.shape[0]
    if cur_corr_num == 0:
        return None, None

    if cur_corr_num >= corr_num:
        sel_inds = np.random.choice(cur_corr_num, corr_num, replace=False)
    else:
        sel_inds = np.arange(cur_corr_num)
        num_rest = corr_num - cur_corr_num
        sel_inds_ = np.random.choice(cur_corr_num, num_rest, replace=True)
        sel_inds = np.concatenate((sel_inds, sel_inds_), axis=0)
        np.random.shuffle(sel_inds)

    return src_sel_inds[sel_inds], tgt_sel_inds[sel_inds]


def calc_gt_patch_correspondence(src_patch, tgt_patch, rot, trans, distance_thres=0.0375):
    '''
    Calculate the ground truth point correspondences between overlapping patches
    :param: src_patch: patches in numpy.ndarray of shape [M, N, 3]
    :param: tgt_patch: patches in numpy.ndarray of shape [M, N, 3]
    :param: rot: ground truth rotation from source to target in numpy.ndarray of shape [3, 3]
    :param: trans: ground truth translation from source to target in numpy.ndarray of shape [3, 1]
    :param: distance_thres: distance threshold to determine point correspondences
    :return: gt_patch_corr: ground truth corresponding score matrix in numpy.ndarray of shape[M, N + 1, N + 1], where the last row and column are slacking used in Optimal Transport
    '''
    trans_src_patch = np.matmul(src_patch, rot.T) + (trans.T)[np.newaxis, :]
    distance = np.sqrt(get_batched_square_distance_matrix(trans_src_patch, tgt_patch))
    gt_patch_corr = (distance < distance_thres).astype(np.float32)
    new_row = np.zeros(shape=(src_patch.shape[0], 1, tgt_patch.shape[1]))
    new_col = np.zeros(shape=(src_patch.shape[0], src_patch.shape[1] + 1, 1))
    gt_patch_corr = np.concatenate((gt_patch_corr, new_row), axis=1)
    gt_patch_corr = np.concatenate((gt_patch_corr, new_col), axis=2)
    gt_patch_corr[:, :-1, -1] = np.clip(1. - gt_patch_corr[:, :-1, :-1].sum(-1), a_min=0., a_max=None)
    gt_patch_corr[:, -1, :-1] = np.clip(1. - gt_patch_corr[:, :-1, :-1].sum(-2), a_min=0., a_max=None)
    return gt_patch_corr


def random_point_subsampling(pcd, num_patches=2048):
    '''
    Reference: https://github.com/XuyangBai/PPF-FoldNet/blob/63af940632ad2350141e332314aab99f41766378/input_preparation.py#L52
    :param pcd: point cloud of numpy.ndarray in shape[N, 3]
    :param num_patches: number of sub-sampled nodes (patches)
    :return: sub-sampled point indices and corresponding nodes in shape[num_patches, 3]
    '''
    num_points = pcd.shape[0]
    if num_points >= num_patches:
        inds = np.random.choice(range(num_points), num_patches, replace=False)
    else:
        inds = np.random.choice(range(num_points), num_patches, replace=True)
    return inds, pcd[inds, :]


def farthest_point_subsampling(pcd, num_patches=2048):
    '''
    Farthest point sampling (FPS)
    :param pcd: point cloud in numpy.ndarray of shape [n, 3]
    :param num_patches: number of sub-sampled nodes (m)
    :return: inds: the list of indices of sub-sampled nodes
             nodes: sub-sampled nodes in numpy.ndarray of shape [m, 3]
    '''

    num_points = pcd.shape[0]
    inds = np.zeros(num_patches, dtype=np.int32)
    #inds[0] = 0
    inds[0] = np.random.randint(num_points)
    cur_point = pcd[inds[0], :]
    distances = np.linalg.norm(cur_point[np.newaxis, :] - pcd, axis=-1)

    for i in range(1, num_patches):
        farthest_id = np.argmax(distances)
        farthest_point = pcd[farthest_id, :]
        inds[i] = farthest_id
        cur_distances = np.linalg.norm(farthest_point[np.newaxis, :] - pcd, axis=-1)
        distances = np.minimum(distances, cur_distances)

    return inds, pcd[inds, :]


def collect_local_neighbors(points, nodes, vicinity=0.3, num_points_per_patch=1024):
    '''
    Grouping points to patches according to given sub-sampled nodes.
    :param points: raw points of numpy.ndarray in shape[N, 3]
    :param nodes: sub-sampled nodes of numpy.ndarray in shape[M, 3]
    :param vicinity: radius for ball query
    :param num_points_per_patch: truncated number of each patch
    :return: indices of numpy.ndarray in shape[M, num_points_per_patch]
             patches of numpy.ndarray in shape[M, num_points_per_patch, 3]
    '''
    pcd = to_o3d_pcd(points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    inds = np.empty(shape=(0, num_points_per_patch))
    patches = np.empty(shape=(0, num_points_per_patch, 3))
    masks = np.empty(shape=(0, num_points_per_patch))

    for i in range(nodes.shape[0]):
        node = nodes[i]

        [_, _, dist] = kdtree.search_knn_vector_3d(node, 2)

        [k, idx, _] = kdtree.search_radius_vector_3d(node, max(vicinity, math.sqrt(dist[1] + 1e-6)))

        idx = np.asarray(idx)
        idx = idx[1:]
        assert idx.shape[0] > 0
        k -= 1
        if k >= num_points_per_patch:
            idx = np.random.choice(idx, num_points_per_patch, replace=False)
            mask = np.ones(num_points_per_patch)
        else:
            idx1 = idx.copy()
            idx2 = np.random.choice(idx, num_points_per_patch - k, replace=True)
            idx = np.concatenate((idx1, idx2), axis=0)
            mask = np.zeros(num_points_per_patch)
            mask[:k] = 1
            shuffle = np.arange(num_points_per_patch)
            np.random.shuffle(shuffle)
            idx = idx[shuffle]
            mask = mask[shuffle]

        inds = np.concatenate((inds, idx[np.newaxis, :]), axis=0)
        patches = np.concatenate((patches, points[idx, :][np.newaxis, :]), axis=0)
        masks = np.concatenate((masks, mask[np.newaxis, :]), axis=0)

    return inds.astype(np.int32), patches, masks


def point2node_sampling(points, nodes, node_id=None, num_points_per_patch=1024):
    '''
    Point to node sampling, where points will be assigned to their nearest nodes
    :param points: raw points of numpy.ndarray in shape[N, 3]
    :param nodes: sub-sampled nodes of numpy.ndarray in shape[M, 3]
    :param num_points_per_patch: truncated number of each patch
    :return: indices of numpy.ndarray in shape[M, num_points_per_patch]
             patches of numpy.ndarray in shape[M, num_points_per_patch, 3]
             masks of numpy.ndarray in shape[M, num_points_per_patch], indicating whether a repeated sample
    '''
    inds = np.empty(shape=(0, num_points_per_patch))
    patches = np.empty(shape=(0, num_points_per_patch, 3))
    masks = np.empty(shape=(0, num_points_per_patch))
    distance = get_square_distance_matrix(points, nodes)
    assigned_node_idx = np.argmin(distance, axis=-1) # [N, ]

    for i in range(nodes.shape[0]):
        if node_id is not None:
            assigned_node_idx[node_id[i]] = i
        point_mask = (assigned_node_idx == i)
        idx = np.nonzero(point_mask)[0]
        k = idx.shape[0]
        assert k != 0
        if k >= num_points_per_patch:
            idx = np.random.choice(idx, num_points_per_patch, replace=False)
            mask = np.ones(num_points_per_patch)
        else:
            idx1 = idx.copy()
            idx2 = np.random.choice(idx, num_points_per_patch - k, replace=True)
            idx = np.concatenate((idx1, idx2), axis=0)
            mask = np.zeros(num_points_per_patch)
            mask[:k] = 1
            shuffle = np.arange(num_points_per_patch)
            np.random.shuffle(shuffle)
            idx = idx[shuffle]
            mask = mask[shuffle]

        inds = np.concatenate((inds, idx[np.newaxis, :]), axis=0)
        patches = np.concatenate((patches, points[idx, :][np.newaxis, :]), axis=0)
        masks = np.concatenate((masks, mask[np.newaxis, :]), axis=0)

    return inds.astype(np.int32), patches, masks


def normal_redirect(points, normals, view_point):
    '''
    Make direction of normals towards the view point
    '''
    vec_dot = np.sum((view_point - points) * normals, axis=-1)
    mask = (vec_dot < 0.)
    redirected_normals = normals.copy()
    redirected_normals[mask] *= -1.
    return redirected_normals


def build_ppf_patch(points, node_inds, neighbors, point_normals=None, view_point=None):
    '''
    Generating patches consisting of point-pair features(ppfs) from input point clouds.
    :param points: raw points of numpy.ndarray in shape[N, 3]
    :param node_inds: indices of sub-sampled nodes of numpy.ndarray in shape[M, 3]
    :param neighbors:  indices of numpy.ndarray in shape[M, num_points_per_patch]
    :return: ppf encoded local patches of numpy.ndarray in shape [M, num_points_per_patch, 4]
             the normal of each node
    '''
    nodes = points[node_inds, :]
    pcd = to_o3d_pcd(points)

    if point_normals is None:
        ############################################
        # Normal estimation
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=33))
        point_normals = np.asarray(pcd.normals)

    if view_point is not None:
        vec_dot = np.sum((view_point - points) * point_normals, axis=-1)
        mask = (vec_dot < 0.)
        point_normals[mask] *= -1.

    node_normals = point_normals[node_inds]

    local_ppf_patch = calc_ppf_cpu(nodes, node_normals, points, point_normals, neighbors)
    return local_ppf_patch, node_normals


def calc_ppf_cpu(point1, normal1, point2, normal2, neighbors):
    '''
    Given a representative point and a patch of points, calculate point-pair feature ppfs for each of the point in this patch
    :param point1: xyz coordinates of the representative point of numpy.ndarray in shape[n, 3]
    :param normal1: normal of the representative point of numpy.ndarray in shape[n, 3]
    :param point2: xyz coordinates of a patch of points of numpy.ndarray in shape[m, 3]
    :param normal2: normal of representative point of numpy.ndarray in shape[m, 3]
    :param neighbors: patches of point indices of numpy.ndarray in shape[n, num_point_per_patch]
    :return: point-pair feature(ppfs) of each point in this patch, of numpy.ndarray in shape[n, num_point_per_patch, 4]
    '''
    patches = np.take(point2, neighbors, axis=0) #[n, num_point_per_patch, 3]
    patch_normals = np.take(normal2, neighbors, axis=0) #[n, num_point_per_patch, 3]

    d = patches - point1[:, np.newaxis, :] #[n, num_point_per_patch, 3]
    len_d = np.linalg.norm(d, axis=-1, keepdims=True)
    # angle(n1, d)
    y = np.sum(normal1[:, np.newaxis, :].repeat(patch_normals.shape[1], axis=1) * d, axis=-1, keepdims=True)  #[n, num_point_per_patch, ]
    x = np.linalg.norm(np.cross(normal1[:, np.newaxis, :].repeat(patch_normals.shape[1], axis=1), d), axis=-1, keepdims=True)

    angle1 = np.arctan2(x, y) / np.pi

    # angle(n2, d)
    y = np.sum(patch_normals * d, axis=-1, keepdims=True) # [n, num_point_per_patch, ]
    x = np.linalg.norm(np.cross(patch_normals, d), axis=-1, keepdims=True)
    angle2 = np.arctan2(x, y) / np.pi

    # angle(n1, n2)
    y = np.sum(normal1[:, np.newaxis, :].repeat(patch_normals.shape[1], axis=1) * patch_normals, axis=-1, keepdims=True) #[n, num_point_per_patch, ]
    x = np.linalg.norm(np.cross(normal1[:, np.newaxis, :].repeat(patch_normals.shape[1], axis=1), patch_normals), axis=-1, keepdims=True)
    angle3 = np.arctan2(x, y) / np.pi

    return np.concatenate((len_d, angle1, angle2, angle3), axis=-1)


def calc_ppf_gpu(point1, normal1, point2, normal2):
    '''
    TBD
    :param point1:
    :param normal1:
    :param point2:
    :param normal2:
    :return:
    '''
    '''
    Given a representative point and a patch of points, calculate point-pair feature ppfs for each of the point in this patch
    :param point1: xyz coordinates of the representative point of numpy.ndarray in shape[1, 3]
    :param normal1: normal of the representative point of numpy.ndarray in shape[1, 3]
    :param point2: xyz coordinates of a patch of points of numpy.ndarray in shape[num_point_per_patch, 3]
    :param normal2: normal of representative point of numpy.ndarray in shape[num_point_per_patch, 3]
    :return: point-pair feature(ppfs) of each point in this patch, of numpy.ndarray in shape[num_point_per_patch, 4]
    '''
    d = point2 - point1[np.newaxis, :]  # [num_point_per_patch, 3]
    len_d = np.linalg.norm(d, axis=1)
    # angle(n1, d)
    y = np.sum(normal1[np.newaxis, :].repeat(normal2.shape[0], axis=0) * d, axis=1)  # [num_point_per_patch, ]
    x = np.linalg.norm(np.cross(normal1[np.newaxis, :].repeat(normal2.shape[0], axis=0), d), axis=1)

    angle1 = np.arctan2(x, y) / np.pi

    # angle(n2, d)
    y = np.sum(normal2 * d, axis=1)  # [num_point_per_patch, ]
    x = np.linalg.norm(np.cross(normal2, d), axis=1)
    angle2 = np.arctan2(x, y) / np.pi

    # angle(n1, n2)
    y = np.sum(normal1[np.newaxis, :].repeat(normal2.shape[0], axis=0) * normal2, axis=1)  # [num_point_per_patch, ]
    x = np.linalg.norm(np.cross(normal1[np.newaxis, :].repeat(normal2.shape[0], axis=0), normal2), axis=1)
    angle3 = np.arctan2(x, y) / np.pi

    return np.array([len_d, angle1, angle2, angle3]).T


def calc_patch_overlap_ratio(src_nodes, src_patches, src_masks, tgt_nodes, tgt_patches, tgt_masks, rot, trans, overlap_control=0.0375):
    '''
    Calculate patch-level pairwise overlap ratios.
    Args:
        src_nodes: Source down-sampled nodes of numpy.ndarray in shape [M1, 3]
        src_patches: Source patches of numpy.ndarray in shape [M1, K, 3]
        src_masks: Source patch masks to indicate whether a point is repeatedly sampled, of numpy.ndarray in shape [M1, K]
        tgt_nodes: Target down-sampled nodes of numpy.ndarray in shape [M2, 3]
        tgt_patches: Target patches of numpy.ndarray in shape [M2, K, 3]
        tgt_masks: Target patch masks to indicate whether a point is repeatedly sampled, of numpy.ndarray in shape [M2, K]
        rot: Rotation matrix of numpy.ndarray in shape [3, 3]
        trans: Translation matrix of numpy.ndarray in shape [3, 1]

    Returns:
        Row- and column- major overlap ratio matrices
    '''
    src_distance = np.linalg.norm(src_nodes[:, np.newaxis, :] - src_patches, axis=-1) # [M1, K]
    src_distance_thres = np.max(src_distance, axis=-1) #[M1]
    tgt_distance = np.linalg.norm(tgt_nodes[:, np.newaxis, :] - tgt_patches, axis=-1)
    tgt_distance_thres = np.max(tgt_distance, axis=-1) #[M2]

    transformed_src_nodes = (np.matmul(rot, src_nodes.T) + trans).T
    transformed_src_patches = (np.matmul(rot[np.newaxis, :], src_patches.transpose(0, 2, 1)) + trans[np.newaxis, :]).transpose(0, 2, 1) #[M1, K, 3]
    distance = np.sqrt(get_square_distance_matrix(transformed_src_nodes, tgt_nodes)) # [M1, M2]
    row_major_overlap_ratio_matrix = np.zeros_like(distance)
    col_major_overlap_ratio_matrix = np.zeros_like(distance)
    #########################################################################################
    # Row Major
    #########################################################################################
    row_major_dist_mask = np.less(distance, src_distance_thres[:, np.newaxis]) #[M1, M2]
    row_major_possible_corr = np.nonzero(row_major_dist_mask)

    sel_src_patches = transformed_src_patches[row_major_possible_corr[0], :, :] #[X, K, 3]
    sel_src_masks = src_masks[row_major_possible_corr[0], :] #[X, K]
    sel_tgt_patches = tgt_patches[row_major_possible_corr[1], :, :] #[X, K, 3]

    patchwise_mask = np.sqrt(get_batched_square_distance_matrix(sel_src_patches, sel_tgt_patches)) < overlap_control #[X, K, K]
    patchwise_mask = (np.sum(patchwise_mask, axis=-1) > 0) #[X, K]
    row_major_overlap_ratio = np.sum(patchwise_mask * sel_src_masks, axis=-1) / np.sum(sel_src_masks, axis=-1) #[X, ]
    row_major_overlap_ratio_matrix[row_major_possible_corr[0], row_major_possible_corr[1]] = row_major_overlap_ratio #[M1, M2]

    #########################################################################################
    # Column Major
    #########################################################################################
    col_major_dist_mask = np.less(distance, tgt_distance_thres[np.newaxis, :]) #[M1, M2]
    col_major_possible_corr = np.nonzero(col_major_dist_mask)

    sel_src_patches = transformed_src_patches[col_major_possible_corr[0], :, :] #[X, K, 3]
    sel_tgt_patches = tgt_patches[col_major_possible_corr[1], :, :] #[X, K, 3]
    sel_tgt_masks = tgt_masks[col_major_possible_corr[1], :] #[X, K]

    patchwise_mask = np.sqrt(get_batched_square_distance_matrix(sel_src_patches, sel_tgt_patches)) < overlap_control #[X, K, K]
    patchwise_mask = (np.sum(patchwise_mask, axis=-2) > 0) #[X, K]
    col_major_overlap_ratio = np.sum(patchwise_mask * sel_tgt_masks, axis=-1) / np.sum(sel_tgt_masks, axis=-1) #[X, ]
    col_major_overlap_ratio_matrix[col_major_possible_corr[0], col_major_possible_corr[1]] = col_major_overlap_ratio #[M1, M2]

    return row_major_overlap_ratio_matrix, col_major_overlap_ratio_matrix


def uniform_2_sphere(num: int = None):
    """Uniform sampling on a 2-sphere
    Source: https://gist.github.com/andrewbolster/10274979
    Args:
        num: Number of vectors to sample (or None if single)
    Returns:
        Random Vector (np.ndarray) of size (num, 3) with norm 1.
        If num is None returned value will have size (3,)
    """
    if num is not None:
        phi = np.random.uniform(0.0, 2 * np.pi, num)
        cos_theta = np.random.uniform(-1.0, 1.0, num)
    else:
        phi = np.random.uniform(0.0, 2 * np.pi)
        cos_theta = np.random.uniform(-1.0, 1.0)

    theta = np.arccos(cos_theta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.stack((x, y, z), axis=-1)


def random_crop(points, p_keep):
    '''

    :param points:
    :param p_keep:
    :return:
    '''

    rand_xyz = uniform_2_sphere()
    centroid = np.mean(points[:, :3], axis=0)
    points_centered = points[:, :3] - centroid

    dist_from_plane = np.dot(points_centered, rand_xyz)

    if p_keep == 0.5:
        mask = dist_from_plane > 0
    else:
        mask = dist_from_plane > np.percentile(dist_from_plane, (1.0 - p_keep) * 100)

    return points[mask, :]


def x_axis_crop(pcd, min_overlap=0.3):
    x_min = np.min(pcd, axis=0)[0]
    x_max = np.max(pcd, axis=0)[0]
    interval = x_max - x_min
    left_max = np.random.uniform(0.6, 1.)
    right_min = np.random.uniform(0., 0.4)
    left_min = 0.
    right_max = 1.

    left_min = x_min + interval * left_min
    left_max = x_min + interval * left_max
    right_min = x_min + interval * right_min
    right_max = x_min + interval * right_max

    left_mask = np.logical_and(pcd[:, 0] >= left_min, pcd[:, 0] <= left_max)
    right_mask = np.logical_and(pcd[:, 0] >= right_min, pcd[:, 0] <= right_max)

    if np.random.rand(1)[0] > 0.5:
        src_pcd = copy.deepcopy(pcd[left_mask, :])
        tgt_pcd = copy.deepcopy(pcd[right_mask, :])
    else:
        src_pcd = copy.deepcopy(pcd[right_mask, :])
        tgt_pcd = copy.deepcopy(pcd[left_mask, :])
    return src_pcd, tgt_pcd

