import random, time, re
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from typing import Optional
from cpp_wrappers.pointops.functions.pointops import knnquery
import glob, os


def read_entries(split, data_root, shuffle=False):
    '''
    Read all the names of data files into a single list
    :param split: ['train', 'val', 'test']
    :param data_root: Directory of data
    :param shuffle: Whether to shuffle the resulted data list
    :return: Data list
    '''
    #print(os.path.join(data_root, split, "*/*.npz"))
    entries = glob.glob(os.path.join(data_root, split, "*/*.npz"))
    #print(entries)
    if shuffle:
        random.shuffle(entries)

    return entries


def setup_seed(seed):
    '''
    fix random seed for deterministic training
    :param seed: selected seed for deterministic training
    :return: None
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def natural_key(string_):
    """
    Sort strings by numbers in the name
    """
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def to_tsfm(rot, trans):
    '''
    Transfer rotation and translation to transformation
    :param rot: rotation matrix of numpy.ndarray in shape [3, 3]
    :param trans: translation vector of numpy.ndarray in shape[3, 1]
    :return: Transformation matrix of numpy.ndarray in shape[4, 4]
    '''
    tsfm = np.eye(4)
    tsfm[:3, :3] = rot
    tsfm[:3, 3] = trans.flatten()
    return tsfm


def to_o3d_pcd(pcd):
    '''
    Transfer a point cloud of numpy.ndarray to open3d point cloud
    :param pcd: point cloud of numpy.ndarray in shape[N, 3]
    :return: open3d.geometry.PointCloud()
    '''
    pcd_ = o3d.geometry.PointCloud()
    pcd_.points = o3d.utility.Vector3dVector(pcd)
    return pcd_


def get_correspondences(src_pcd, tgt_pcd, trans, search_voxel_size, K=None):
    '''
    Get correspondences between a pair of point clouds, given the ground truth transformation
    :param src_pcd: source point cloud of open3d.geomerty.PointCloud in shape[N, 3]
    :param tgt_pcd: target point cloud of open3d.geomerty.PointCloud in shape[M, 3]
    :param trans: transformation matrix of numpy.ndarray in shape[4, 4]
    :param search_voxel_size: distrance threshold within which two points are considered as a correspondence
    :param K: if K is not None, only return K corresponding points for each point
    :return: correspondences of torch.tensor in shape[?, 2]
    '''

    src_pcd.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(tgt_pcd)

    correspondences = []
    for i, point in enumerate(src_pcd.points):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            correspondences.append([i, j])

    correspondences = np.array(correspondences)
    correspondences = torch.from_numpy(correspondences)
    return correspondences


def matching_descriptors(src_desc, tgt_desc, mutual=False, major=None):
    '''
    Matching based on descriptors, return correspondences
    :param src_desc: descriptors of source point cloud
    :param tgt_desc: descriptors of target point cloud
    :param mutual: wheter to perform mutual selection
    :return: Extracted correspondences of numpy.ndarray in shape[n, 2]
    '''
    assert major in ['row', 'col'] or major is None
    distances = square_distance(torch.from_numpy(src_desc[np.newaxis, :]), torch.from_numpy(tgt_desc[np.newaxis, :]))[0].numpy()
    row_idx = np.arange(src_desc.shape[0])
    row_major_idx = np.argmin(distances, axis=1)
    col_idx = np.arange(tgt_desc.shape[0])
    col_major_idx = np.argmin(distances, axis=0)
    if not mutual:
        if major == 'row':
            correspondence = np.concatenate((row_idx[:, np.newaxis], row_major_idx[:, np.newaxis]), axis=1)
        elif major == 'col':
            correspondence = np.concatenate((col_major_idx[:, np.newaxis], col_idx[:, np.newaxis]), axis=1)
        else:
            row_major_mask = np.zeros_like(distances)
            row_major_mask[row_idx, row_major_idx] = 1
            col_major_mask = np.zeros_like(distances)
            col_major_mask[col_major_idx, col_idx] = 1
            mask = np.logical_or(row_major_mask > 0, col_major_mask > 0)
            correspondence = np.nonzero(mask)
            correspondence = np.concatenate((correspondence[0][:, np.newaxis], correspondence[1][:, np.newaxis]),
                                            axis=-1)
        return correspondence
    else:
        row_major_mask = np.zeros_like(distances)
        row_major_mask[row_idx, row_major_idx] = 1
        col_major_mask = np.zeros_like(distances)
        col_major_mask[col_major_idx, col_idx] = 1
        mask = np.logical_and(row_major_mask > 0, col_major_mask > 0)
        correspondence = np.nonzero(mask)
        correspondence = np.concatenate((correspondence[0][:, np.newaxis], correspondence[1][:, np.newaxis]), axis=-1)
        return correspondence


def square_distance(src, tgt, normalized=False):
    '''
    Calculate Euclidean distance between every two points, for batched point clouds in torch.tensor
    :param src: source point cloud in shape [B, N, 3]
    :param tgt: target point cloud in shape [B, M, 3]
    :return: Squared Euclidean distance matrix in torch.tensor of shape[B, N, M]
    '''
    B, N, _ = src.shape
    _, M, _ = tgt.shape
    if normalized:
        dist = 2.0 - 2.0 * torch.matmul(src, tgt.permute(0, 2, 1).contiguous())
    else:
        dist = -2. * torch.matmul(src, tgt.permute(0, 2, 1).contiguous())
        dist += torch.sum(src ** 2, dim=-1).unsqueeze(-1)
        dist += torch.sum(tgt ** 2, dim=-1).unsqueeze(-2)

    dist = torch.clamp(dist, min=1e-12, max=None)
    return dist


def weighted_procrustes(src_points, tgt_points, weights=None, weight_thresh=0., eps=1e-5, return_transform=False):
    r"""
    Compute rigid transformation from `src_points` to `tgt_points` using weighted SVD.

    Modified from [PointDSC](https://github.com/XuyangBai/PointDSC/blob/master/models/common.py).

    :param src_points: torch.Tensor (batch_size, num_corr, 3) or (num_corr, 3)
    :param tgt_points: torch.Tensor (batch_size, num_corr, 3) or (num_corr, 3)
    :param weights: torch.Tensor (batch_size, num_corr) or (num_corr,) (default: None)
    :param weight_thresh: float (default: 0.)
    :param eps: float (default: 1e-5)
    :param return_transform: bool (default: False)

    :return R: torch.Tensor (batch_size, 3, 3) or (3, 3)
    :return t: torch.Tensor (batch_size, 3) or (3,)
    :return transform: torch.Tensor (batch_size, 4, 4) or (4, 4)
    """
    if src_points.ndim == 2:
        src_points = src_points.unsqueeze(0)
        tgt_points = tgt_points.unsqueeze(0)
        if weights is not None:
            weights = weights.unsqueeze(0)
        squeeze_first = True
    else:
        squeeze_first = False

    batch_size = src_points.shape[0]
    if weights is None:
        weights = torch.ones_like(src_points[:, :, 0])
    weights = torch.where(torch.lt(weights, weight_thresh), torch.zeros_like(weights), weights)
    weights_norm = weights / (torch.sum(weights, dim=1, keepdim=True) + eps)

    src_centroid = torch.sum(src_points * weights_norm.unsqueeze(2), dim=1, keepdim=True)
    tgt_centroid = torch.sum(tgt_points * weights_norm.unsqueeze(2), dim=1, keepdim=True)
    src_points_centered = src_points - src_centroid
    tgt_points_centered = tgt_points - tgt_centroid

    W = torch.diag_embed(weights)
    H = src_points_centered.permute(0, 2, 1) @ W @ tgt_points_centered
    U, _, V = torch.svd(H)  # H = USV^T
    Ut, V = U.transpose(1, 2), V
    eye = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
    eye[:, -1, -1] = torch.sign(torch.det(V @ Ut))
    R = V @ eye @ Ut

    t = tgt_centroid.permute(0, 2, 1) - R @ src_centroid.permute(0, 2, 1)
    t = t.squeeze(2)

    if return_transform:
        transform = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
        transform[:, :3, :3] = R
        transform[:, :3, 3] = t
        if squeeze_first:
            transform = transform.squeeze(0)
        return transform
    else:
        if squeeze_first:
            R = R.squeeze(0)
            t = t.squeeze(0)
        return R, t


def sinkhorn(log_alpha, n_iters: int = 5, slack: bool = True, eps: float = -1) -> torch.Tensor:
    """ Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column sum to <=1
    Args:
        log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
        n_iters (int): Number of normalization iterations
        slack (bool): Whether to include slack row and column
        eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.
    Returns:
        log(perm_matrix): Doubly stochastic matrix (B, J, K)
    Modified from original source taken from:
        Learning Latent Permutations with Gumbel-Sinkhorn Networks
        https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
    """

    # Sinkhorn iterations
    prev_alpha = None
    if slack:
        zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        log_alpha_padded = zero_pad(log_alpha[:, None, :, :])

        log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

        for i in range(n_iters):
            # Row normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                    log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                dim=1)

            # Column normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                    log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                dim=2)

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()

        log_alpha = log_alpha_padded[:, :-1, :-1]
    else:
        for i in range(n_iters):
            # Row normalization (i.e. each row sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True))

            # Column normalization (i.e. each column sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True))

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha).clone()

    return log_alpha



def interpolate(weights, points):
    '''
    Do interpolation based on provided weights
    :param weights: interpolation weights in torch.tensor of shape [b, n, m]
    :param points: points to be interpolated, in torch.tensor of shape [b, m, 3]
    :return: Interpolated coordinates in torch.tensor of shape[b, n, 3]
    '''
    weights = torch.unsqueeze(weights, dim=-1).expand(-1, -1, -1, 3) # [b, n, m] -> [b, n, m, 3]
    points = torch.unsqueeze(points, dim=1).expand(-1, weights.shape[1], -1, -1) #[b, m, 3] -> [b, n, m, 3]
    interpolation = torch.sum(weights * points, dim=-2)
    return interpolation


def soft_assignment(src_xyz, src_feats, tgt_xyz, tgt_feats):
    '''
    Differentiablely compute correspondences between points, return with weights.
    :param src_xyz: Torch tensor in shape[b, n, 3]
    :param src_feats: Torch tensor in shape[b, n, c]
    :param tgt_xyz: Torch tensor in shape[b, n, 3]
    :param tgt_feats: Torch tensor in shape[b, n, c]
    :return: src2tgt_assignment_confidence: confidence of each corresponding point, torch.tensor in shape[b, n]
             src2tgt_interpolated_xyz: interpolated xyz coordinates in tgt space, torch.tensor in shape[b, n, 3]
             tgt2src_assignment_confidence: confidence of each corresponding point, torch.tensor in shape[b, n]
             tgt2src_interpolated_xyz: interpolated xyz coordinates in src space, torch.tensor in shape[b, n]
    '''
    feat_distance = torch.sqrt(square_distance(src_feats, tgt_feats))
    feat_similarity = 1. / (1e-8 + feat_distance) #similarity matrix in shape [b, n, n]
    # calculate src's corresponding weights and confidence in tgt
    src2tgt_assignment_weights = feat_similarity / torch.sum(feat_similarity, dim=-1, keepdim=True) #row-normalized similarity matrix in shape [b, n, n]
    src2tgt_assignment_max_sim = torch.max(feat_similarity, dim=-1)[0] #row-major max similarity in shape [b, n]
    src2tgt_assignment_confidence = src2tgt_assignment_max_sim / torch.sum(src2tgt_assignment_max_sim, dim=-1, keepdim=True) #normalized confidence of softassignment in shape [b, n]
    src2tgt_interpolated_xyz = interpolate(src2tgt_assignment_weights, tgt_xyz)
    # calculate tgt's corresponding weights and confidence in src
    tgt2src_assignment_weights = feat_similarity / torch.sum(feat_similarity, dim=1, keepdim=True)  # column-normalized similarity matrix in shape [b, n, n]
    tgt2src_assignment_max_sim = torch.max(feat_similarity, dim=1)[0]  # row-major max similarity in shape [b, n]
    tgt2src_assignment_confidence = tgt2src_assignment_max_sim / torch.sum(tgt2src_assignment_max_sim, dim=-1, keepdim=True)  # normalized confidence of softassignment in shape [b, n]
    tgt2src_interpolated_xyz = interpolate(tgt2src_assignment_weights, src_xyz)
    return src2tgt_assignment_confidence, src2tgt_interpolated_xyz, tgt2src_assignment_confidence, tgt2src_interpolated_xyz


def get_geometric_structure_embeddings(points, angle_k=3):

    batch_size, num_point, _ = points.shape

    dist_map = torch.sqrt(square_distance(points, points))  # (B, N, N)

    knn_indices = dist_map.topk(k=angle_k + 1, dim=2, largest=False)[1]  # (B, N, k)
    knn_indices = knn_indices[:, :, 1:]
    knn_indices = knn_indices.unsqueeze(3).expand(batch_size, num_point, angle_k, 3)  # (B, N, k, 3)
    expanded_points = points.unsqueeze(1).expand(batch_size, num_point, num_point, 3)  # (B, N, N, 3)
    knn_points = torch.gather(expanded_points, dim=2, index=knn_indices)  # (B, N, k, 3)
    ref_vectors = knn_points - points.unsqueeze(2)  # (B, N, k, 3)
    anc_vectors = points.unsqueeze(1) - points.unsqueeze(2)  # (B, N, N, 3)
    ref_vectors = ref_vectors.unsqueeze(2).expand(batch_size, num_point, num_point, angle_k, 3)  # (B, N, N, k, 3)
    anc_vectors = anc_vectors.unsqueeze(3).expand(batch_size, num_point, num_point, angle_k, 3)  # (B, N, N, k, 3)
    sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)  # (B, N, N, k)
    cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)  # (B, N, N, k)
    angles = torch.atan2(sin_values, cos_values)  # (B, N, N, k)

    return dist_map, angles


def k_nearest_neighbors(query, ref, k):
    '''
    Get k nearest neighbors
    query: query points in shape [N, c]
    ref: ref points in shape[M, c]
    k: number of nearest neighbors
    '''
    dist = square_distance(query[None, ::], ref[None, ::])
    _, knn_ids = torch.topk(dist, largest=False, dim=-1, sorted=True) #[N, k]
    return knn_ids



def calc_ppf_gpu(points, point_normals, patches, patch_normals):
    '''
    Calculate ppf gpu
    points: [n, 3]
    point_normals: [n, 3]
    patches: [n, nsamples, 3]
    patch_normals: [n, nsamples, 3]
    '''
    points = torch.unsqueeze(points, dim=1).expand(-1, patches.shape[1], -1)
    point_normals = torch.unsqueeze(point_normals, dim=1).expand(-1, patches.shape[1], -1)
    vec_d = patches - points #[n, n_samples, 3]
    d = torch.sqrt(torch.sum(vec_d ** 2, dim=-1, keepdim=True)) #[n, n_samples, 1]
    # angle(n1, vec_d)
    y = torch.sum(point_normals * vec_d, dim=-1, keepdim=True)
    x = torch.cross(point_normals, vec_d, dim=-1)
    x = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True))
    angle1 = torch.atan2(x, y) / np.pi

    # angle(n2, vec_d)
    y = torch.sum(patch_normals * vec_d, dim=-1, keepdim=True)
    x = torch.cross(patch_normals, vec_d, dim=-1)
    x = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True))
    angle2 = torch.atan2(x, y) / np.pi

    # angle(n1, n2)
    y = torch.sum(point_normals * patch_normals, dim=-1, keepdim=True)
    x = torch.cross(point_normals, patch_normals, dim=-1)
    x = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True))
    angle3 = torch.atan2(x, y) / np.pi

    ppf = torch.cat([d, angle1, angle2, angle3], dim=-1) #[n, samples, 4]
    return ppf


def group_all(feats):
    '''
    all-to-all grouping
    feats: [n, c]
    out: grouped feat: [n, n, c]
    '''
    grouped_feat = torch.unsqueeze(feats, dim=0)
    grouped_feat = grouped_feat.expand(feats.shape[0], -1, -1) #[n, n, c]
    return grouped_feat


def index_select(data: torch.Tensor, index: torch.LongTensor, dim: int) -> torch.Tensor:
    r"""Advanced index select.
    Returns a tensor `output` which indexes the `data` tensor along dimension `dim`
    using the entries in `index` which is a `LongTensor`.
    Different from `torch.index_select`, `index` does not has to be 1-D. The `dim`-th
    dimension of `data` will be expanded to the number of dimensions in `index`.
    For example, suppose the shape `data` is $(a_0, a_1, ..., a_{n-1})$, the shape of `index` is
    $(b_0, b_1, ..., b_{m-1})$, and `dim` is $i$, then `output` is $(n+m-1)$-d tensor, whose shape is
    $(a_0, ..., a_{i-1}, b_0, b_1, ..., b_{m-1}, a_{i+1}, ..., a_{n-1})$.
    Args:
        data (Tensor): (a_0, a_1, ..., a_{n-1})
        index (LongTensor): (b_0, b_1, ..., b_{m-1})
        dim: int
    Returns:
        output (Tensor): (a_0, ..., a_{dim-1}, b_0, ..., b_{m-1}, a_{dim+1}, ..., a_{n-1})
    """
    output = data.index_select(dim, index.view(-1))

    if index.ndim > 1:
        output_shape = data.shape[:dim] + index.shape + data.shape[dim:][1:]
        output = output.view(*output_shape)

    return output


def point_to_node_partition(
    points: torch.Tensor,
    nodes: torch.Tensor,
    point_limit: int,
    return_count: bool = False,
):
    r"""Point-to-Node partition to the point cloud.
    Fixed knn bug.
    Args:
        points (Tensor): (N, 3)
        nodes (Tensor): (M, 3)
        point_limit (int): max number of points to each node
        return_count (bool=False): whether to return `node_sizes`
    Returns:
        point_to_node (Tensor): (N,)
        node_sizes (LongTensor): (M,)
        node_masks (BoolTensor): (M,)
        node_knn_indices (LongTensor): (M, K)
        node_knn_masks (BoolTensor) (M, K)
    """
    sq_dist_mat = square_distance(nodes[None, ::], points[None, ::])[0]  # (M, N)

    point_to_node = sq_dist_mat.min(dim=0)[1]  # (N,)
    node_masks = torch.zeros(nodes.shape[0], dtype=torch.bool).cuda()  # (M,)
    node_masks.index_fill_(0, point_to_node, True)

    matching_masks = torch.zeros_like(sq_dist_mat, dtype=torch.bool)  # (M, N)
    point_indices = torch.arange(points.shape[0]).cuda()  # (N,)
    matching_masks[point_to_node, point_indices] = True  # (M, N)
    sq_dist_mat.masked_fill_(~matching_masks, 1e12)  # (M, N)

    node_knn_indices = sq_dist_mat.topk(k=point_limit, dim=1, largest=False)[1]  # (M, K)
    node_knn_node_indices = index_select(point_to_node, node_knn_indices, dim=0)  # (M, K)
    node_indices = torch.arange(nodes.shape[0]).cuda().unsqueeze(1).expand(-1, point_limit)  # (M, K)
    node_knn_masks = torch.eq(node_knn_node_indices, node_indices)  # (M, K)
    node_knn_indices.masked_fill_(~node_knn_masks, points.shape[0])

    if return_count:
        unique_indices, unique_counts = torch.unique(point_to_node, return_counts=True)
        node_sizes = torch.zeros(nodes.shape[0], dtype=torch.long).cuda()  # (M,)
        node_sizes.index_put_([unique_indices], unique_counts)
        return point_to_node, node_sizes, node_masks, node_knn_indices, node_knn_masks
    else:
        return point_to_node, node_masks, node_knn_indices, node_knn_masks


def get_node_occlusion_score(
        ref_knn_point_ids: torch.Tensor,
        src_knn_point_ids: torch.Tensor,
        ref_points: torch.Tensor,
        src_points: torch.Tensor,
        rot: torch.Tensor,
        trans: torch.Tensor,
        ref_masks: Optional[torch.Tensor] = None,
        src_masks: Optional[torch.Tensor] = None,
        ref_knn_masks: Optional[torch.Tensor] = None,
        src_knn_masks: Optional[torch.Tensor] = None,
        overlap_thres: Optional[float] = 0.0375,
):
    r"""
    Compute the occlusion scores for each node. Scores are in range of [0, 1], 0 for completely occluded,
    while 1 for completely visible, depending on vicinity points.
    Args:
        ref_knn_point_ids: torch.Tensor (M, K)
        src_knn_point_ids: torch.Tensor (N, K)
        ref_points: torch.Tensor (N1, 3)
        src_points: torch.Tensor (N2, 3)
        rot: torch.Tensor (3, 3)
        trans: torch.Tensor (3, 1)
        ref_masks (optional): torch.BoolTensor (M,) (default: None)
        src_masks (optional): torch.BoolTensor (N,) (default: None)
        ref_knn_masks (optional): torch.BoolTensor (M, K) (default: None)
        src_knn_masks (optional): torch.BoolTensor (N, K) (default: None)
    Returns:
        ref_overlap_score: torch.Tensor (M,)
        src_overlap_score: torch.Tensor (N,)
    """
    src_points = torch.matmul(src_points, rot.T) + trans.T
    ref_o, src_o = torch.from_numpy(np.array([ref_points.shape[0]])).to(ref_points).int(), torch.from_numpy(np.array([src_points.shape[0]])).to(src_points).int()

    _, ref_dist = knnquery(1, src_points, ref_points, src_o, ref_o)
    _, src_dist = knnquery(1, ref_points, src_points, ref_o, src_o)

    ref_overlap = torch.lt(ref_dist, overlap_thres).float().squeeze(1) #(M, )
    src_overlap = torch.lt(src_dist, overlap_thres).float().squeeze(1) #(N, )

    M, K = ref_knn_point_ids.shape
    N, _ = src_knn_point_ids.shape
    ref_knn_point_ids = ref_knn_point_ids.view(-1).contiguous()
    src_knn_point_ids = src_knn_point_ids.view(-1).contiguous()

    ref_knn_overlap = ref_overlap[ref_knn_point_ids].reshape((M, K))
    src_knn_overlap = src_overlap[src_knn_point_ids].reshape((N, K))

    ref_overlap_score = torch.sum(ref_knn_overlap * ref_knn_masks, dim=1) / (torch.sum(ref_knn_masks, dim=1) + 1e-10)
    src_overlap_score = torch.sum(src_knn_overlap * src_knn_masks, dim=1) / (torch.sum(src_knn_masks, dim=1) + 1e-10)

    ref_overlap_score = ref_overlap_score * ref_masks
    src_overlap_score = src_overlap_score * src_masks
    return ref_overlap_score, src_overlap_score


def get_node_correspondences(
    ref_nodes: torch.Tensor,
    src_nodes: torch.Tensor,
    ref_knn_points: torch.Tensor,
    src_knn_points: torch.Tensor,
    rot: torch.Tensor,
    trans: torch.Tensor,
    pos_radius: float,
    ref_masks: Optional[torch.Tensor] = None,
    src_masks: Optional[torch.Tensor] = None,
    ref_knn_masks: Optional[torch.Tensor] = None,
    src_knn_masks: Optional[torch.Tensor] = None,
):
    r"""Generate ground-truth superpoint/patch correspondences.
    Each patch is composed of at most k nearest points of the corresponding superpoint.
    A pair of points match if the distance between them is smaller than `self.pos_radius`.
    Args:
        ref_nodes: torch.Tensor (M, 3)
        src_nodes: torch.Tensor (N, 3)
        ref_knn_points: torch.Tensor (M, K, 3)
        src_knn_points: torch.Tensor (N, K, 3)
        rot: torch.Tensor (3, 3)
        trans: torch.Tensor (3, 1)
        pos_radius: float
        ref_masks (optional): torch.BoolTensor (M,) (default: None)
        src_masks (optional): torch.BoolTensor (N,) (default: None)
        ref_knn_masks (optional): torch.BoolTensor (M, K) (default: None)
        src_knn_masks (optional): torch.BoolTensor (N, K) (default: None)
    Returns:
        corr_indices: torch.LongTensor (C, 2)
        corr_overlaps: torch.Tensor (C,)
    """
    src_nodes = torch.matmul(src_nodes, rot.T) + trans.T
    src_knn_points = torch.matmul(src_knn_points, rot.T) + (trans.T)[None, ::]

    # generate masks
    if ref_masks is None:
        ref_masks = torch.ones(size=(ref_nodes.shape[0],), dtype=torch.bool).cuda()
    if src_masks is None:
        src_masks = torch.ones(size=(src_nodes.shape[0],), dtype=torch.bool).cuda()
    if ref_knn_masks is None:
        ref_knn_masks = torch.ones(size=(ref_knn_points.shape[0], ref_knn_points.shape[1]), dtype=torch.bool).cuda()
    if src_knn_masks is None:
        src_knn_masks = torch.ones(size=(src_knn_points.shape[0], src_knn_points.shape[1]), dtype=torch.bool).cuda()

    node_mask_mat = torch.logical_and(ref_masks.unsqueeze(1), src_masks.unsqueeze(0))  # (M, N)

    # filter out non-overlapping patches using enclosing sphere
    ref_knn_dists = torch.linalg.norm(ref_knn_points - ref_nodes.unsqueeze(1), dim=-1)  # (M, K)
    ref_knn_dists.masked_fill_(~ref_knn_masks, 0.0)
    ref_max_dists = ref_knn_dists.max(1)[0]  # (M,)
    src_knn_dists = torch.linalg.norm(src_knn_points - src_nodes.unsqueeze(1), dim=-1)  # (N, K)
    src_knn_dists.masked_fill_(~src_knn_masks, 0.0)
    src_max_dists = src_knn_dists.max(1)[0]  # (N,)
    dist_mat = torch.sqrt(square_distance(ref_nodes[None, ::], src_nodes[None, ::])[0])  # (M, N)
    intersect_mat = torch.gt(ref_max_dists.unsqueeze(1) + src_max_dists.unsqueeze(0) + pos_radius - dist_mat, 0)
    intersect_mat = torch.logical_and(intersect_mat, node_mask_mat)
    sel_ref_indices, sel_src_indices = torch.nonzero(intersect_mat, as_tuple=True)

    # select potential patch pairs
    ref_knn_masks = ref_knn_masks[sel_ref_indices]  # (B, K)
    src_knn_masks = src_knn_masks[sel_src_indices]  # (B, K)
    ref_knn_points = ref_knn_points[sel_ref_indices]  # (B, K, 3)
    src_knn_points = src_knn_points[sel_src_indices]  # (B, K, 3)

    point_mask_mat = torch.logical_and(ref_knn_masks.unsqueeze(2), src_knn_masks.unsqueeze(1))  # (B, K, K)

    # compute overlaps
    dist_mat = square_distance(ref_knn_points, src_knn_points) # (B, K, K)
    dist_mat.masked_fill_(~point_mask_mat, 1e12)
    point_overlap_mat = torch.lt(dist_mat, pos_radius ** 2)  # (B, K, K)
    ref_overlap_counts = torch.count_nonzero(point_overlap_mat.sum(-1), dim=-1).float()  # (B,)
    src_overlap_counts = torch.count_nonzero(point_overlap_mat.sum(-2), dim=-1).float()  # (B,)
    ref_overlaps = ref_overlap_counts / ref_knn_masks.sum(-1).float()  # (B,)
    src_overlaps = src_overlap_counts / src_knn_masks.sum(-1).float()  # (B,)
    overlaps = (ref_overlaps + src_overlaps) / 2  # (B,)

    overlap_masks = torch.gt(overlaps, 0)
    ref_corr_indices = sel_ref_indices[overlap_masks]
    src_corr_indices = sel_src_indices[overlap_masks]
    corr_indices = torch.stack([ref_corr_indices, src_corr_indices], dim=1)
    #corr_indices = torch.stack([src_corr_indices, ref_corr_indices], dim=1)
    corr_overlaps = overlaps[overlap_masks]

    return corr_indices, corr_overlaps


########################
# utils classes
########################

class AverageMeter(object):
    '''
    A class computes and stores the average and current values
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.sq_sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.sq_sum += val ** 2 * n
        self.var = self.sq_sum / self.count - self.avg ** 2


class Timer(object):
    '''
    A simple timer
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.avg = 0.

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.avg = self.total_time / self.calls
        if average:
            return self.avg
        else:
            return self.diff


class Logger(object):
    '''
    A simple logger
    '''

    def __init__(self, path):
        self.path = path
        self.fw = open(self.path + '/log', 'a')

    def write(self, text):
        self.fw.write(text)
        self.fw.flush()

    def close(self):
        self.fw.close()
