# Reference: https://github.com/qinzheng93/GeoTransformer

import torch
from torch import nn
from typing import Optional
from lib.utils import square_distance
import numpy as np


class LearnableLogOptimalTransport(nn.Module):
    '''
    Optimal Transport Layer with learnable thresholds
    Reference: GeoTransformer, Zheng Qin et al.
    '''
    def __init__(self, num_iter, inf=1e6):
        super(LearnableLogOptimalTransport, self).__init__()
        self.num_iter = num_iter
        self.register_parameter('alpha', torch.nn.Parameter(torch.tensor(1.)))
        self.inf = inf

    def log_sinkhorn_normalization(self, scores, log_mu, log_nu):
        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
        for _ in range(self.num_iter):
            u = log_mu - torch.logsumexp(scores + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(scores + u.unsqueeze(2), dim=1)
        return scores + u.unsqueeze(2) + v.unsqueeze(1)

    def forward(self, scores, row_masks, col_masks):
        r"""
        Optimal transport with Sinkhorn.
        :param scores: torch.Tensor (B, M, N)
        :param row_masks: torch.Tensor (B, M)
        :param col_masks: torch.Tensor (B, N)
        :return matching_scores: torch.Tensor (B, M+1, N+1)
        """
        batch_size, num_row, num_col = scores.shape
        ninf = torch.tensor(-self.inf).cuda()

        padded_row_masks = torch.zeros(batch_size, num_row + 1, dtype=torch.bool).cuda()
        padded_row_masks[:, :num_row] = ~row_masks
        padded_col_masks = torch.zeros(batch_size, num_col + 1, dtype=torch.bool).cuda()
        padded_col_masks[:, :num_col] = ~col_masks

        padded_col = self.alpha.expand(batch_size, num_row, 1)
        padded_row = self.alpha.expand(batch_size, 1, num_col + 1)
        padded_scores = torch.cat([torch.cat([scores, padded_col], dim=-1), padded_row], dim=1)

        padded_score_masks = torch.logical_or(padded_row_masks.unsqueeze(2), padded_col_masks.unsqueeze(1))
        padded_scores[padded_score_masks] = ninf

        num_valid_row = row_masks.float().sum(1)
        num_valid_col = col_masks.float().sum(1)
        norm = -torch.log(num_valid_row + num_valid_col)  # (B,)

        log_mu = torch.empty(batch_size, num_row + 1).cuda()
        log_mu[:, :num_row] = norm.unsqueeze(1)
        log_mu[:, num_row] = torch.log(num_valid_col) + norm
        log_mu[padded_row_masks] = ninf

        log_nu = torch.empty(batch_size, num_col + 1).cuda()
        log_nu[:, :num_col] = norm.unsqueeze(1)
        log_nu[:, num_col] = torch.log(num_valid_row) + norm
        log_nu[padded_col_masks] = ninf

        outputs = self.log_sinkhorn_normalization(padded_scores, log_mu, log_nu)
        outputs = outputs - norm.unsqueeze(1).unsqueeze(2)

        return outputs

    def __repr__(self):
        format_string = self.__class__.__name__ + '(num_iter={})'.format(self.num_iter)
        return format_string


class AdaptiveSuperPointMatching(nn.Module):
    def __init__(self, min_num_correspondences, similarity_threshold=0.75):
        super().__init__()
        self.min_num_correspondences = min_num_correspondences
        self.similarity_threshold = similarity_threshold

    def forward(self, src_feats, tgt_feats, src_masks=None, tgt_masks=None):
        """Extract superpoint correspondences.

        Args:
            src_feats (Tensor): features of the superpoints in source point cloud.
            tgt_feats (Tensor): features of the superpoints in target point cloud.
            src_masks (BoolTensor, optional): masks of the superpoints in source point cloud (False if empty).
            tgt_masks (BoolTensor, optional): masks of the superpoints in target point cloud (False if empty).

        Returns:
            src_corr_indices (LongTensor): indices of the corresponding superpoints in source point cloud.
            tgt_corr_indices (LongTensor): indices of the corresponding superpoints in target point cloud.
            corr_scores (Tensor): scores of the correspondences.
        """
        # remove empty patch
        if src_masks is not None:
            src_feats = src_feats[src_masks]

        if tgt_masks is not None:
            tgt_feats = tgt_feats[tgt_masks]

        # select proposals
        similarity_mat = torch.sqrt(square_distance(src_feats[None, :], tgt_feats[None, :], normalized=True)[0])
        min_num_correspondences = min(self.min_num_correspondences, similarity_mat.numel())
        masks = torch.le(similarity_mat, self.similarity_threshold)
        if masks.sum() < min_num_correspondences:
            corr_distances, corr_indices = similarity_mat.view(-1).topk(k=min_num_correspondences, largest=False)
            src_corr_indices = torch.div(corr_indices, similarity_mat.shape[1], rounding_mode="floor")
            tgt_corr_indices = corr_indices % similarity_mat.shape[1]
        else:
            src_corr_indices, tgt_corr_indices = torch.nonzero(masks, as_tuple=True)
            corr_distances = similarity_mat[src_corr_indices, tgt_corr_indices]
        corr_scores = torch.exp(-corr_distances)

        # recover original indices
        if src_masks is not None:
            src_valid_indices = torch.nonzero(src_masks, as_tuple=True)[0]
            src_corr_indices = src_valid_indices[src_corr_indices]

        if tgt_masks is not None:
            tgt_valid_indices = torch.nonzero(tgt_masks, as_tuple=True)[0]
            tgt_corr_indices = tgt_valid_indices[tgt_corr_indices]

        return src_corr_indices, tgt_corr_indices, corr_scores

    def extra_repr(self) -> str:
        param_strings = [
            f"min_num_correspondences={self.min_num_correspondences}",
            f"similarity_threshold={self.similarity_threshold:g}",
        ]
        format_string = ", ".join(param_strings)
        return format_string


class CoarseMatching(nn.Module):
    def __init__(self, num_correspondences, dual_normalization=True):
        super(CoarseMatching, self).__init__()
        self.num_correspondences = num_correspondences
        self.dual_normalization = dual_normalization

    def forward(self, ref_feats, src_feats, ref_masks=None, src_masks=None, ref_occ=None, src_occ=None):
        r"""Extract superpoint correspondences.
        Args:
            ref_feats (Tensor): features of the superpoints in reference point cloud.
            src_feats (Tensor): features of the superpoints in source point cloud.
            ref_masks (BoolTensor=None): masks of the superpoints in reference point cloud (False if empty).
            src_masks (BoolTensor=None): masks of the superpoints in source point cloud (False if empty).
        Returns:
            ref_corr_indices (LongTensor): indices of the corresponding superpoints in reference point cloud.
            src_corr_indices (LongTensor): indices of the corresponding superpoints in source point cloud.
            corr_scores (Tensor): scores of the correspondences.
        """
        if ref_masks is None:
            ref_masks = torch.ones(size=(ref_feats.shape[0],), dtype=torch.bool).cuda()
        if src_masks is None:
            src_masks = torch.ones(size=(src_feats.shape[0],), dtype=torch.bool).cuda()
        # remove empty patch
        ref_indices = torch.nonzero(ref_masks, as_tuple=True)[0]
        src_indices = torch.nonzero(src_masks, as_tuple=True)[0]
        ref_feats = ref_feats[ref_indices]
        src_feats = src_feats[src_indices]

        # select top-k proposals
        matching_scores = torch.exp(-square_distance(ref_feats[None, ::], src_feats[None, ::]))[0]

        if self.dual_normalization:
            ref_matching_scores = matching_scores / (matching_scores.sum(dim=1, keepdim=True) + 1e-8)
            src_matching_scores = matching_scores / (matching_scores.sum(dim=0, keepdim=True) + 1e-8)
            matching_scores = ref_matching_scores * src_matching_scores

        num_correspondences = min(self.num_correspondences, matching_scores.numel())
        corr_scores, corr_indices = matching_scores.view(-1).topk(k=num_correspondences, largest=True)
        ref_sel_indices = corr_indices // matching_scores.shape[1]
        src_sel_indices = corr_indices % matching_scores.shape[1]
        # recover original indices
        ref_corr_indices = ref_indices[ref_sel_indices]
        src_corr_indices = src_indices[src_sel_indices]
        return ref_corr_indices, src_corr_indices, corr_scores


class GTCoarseCorrGenerator(nn.Module):
    def __init__(self, num_targets, overlap_threshold):
        super(GTCoarseCorrGenerator, self).__init__()
        self.num_targets = num_targets
        self.overlap_threshold = overlap_threshold

    @torch.no_grad()
    def forward(self, gt_corr_indices, gt_corr_overlaps):
        r"""Generate ground truth superpoint (patch) correspondences.
        Randomly select "num_targets" correspondences whose overlap is above "overlap_threshold".
        Args:
            gt_corr_indices (LongTensor): ground truth superpoint correspondences (N, 2)
            gt_corr_overlaps (Tensor): ground truth superpoint correspondences overlap (N,)
        Returns:
            gt_ref_corr_indices (LongTensor): selected superpoints in reference point cloud.
            gt_src_corr_indices (LongTensor): selected superpoints in source point cloud.
            gt_corr_overlaps (LongTensor): overlaps of the selected superpoint correspondences.
        """
        gt_corr_masks = torch.gt(gt_corr_overlaps, self.overlap_threshold)
        gt_corr_overlaps = gt_corr_overlaps[gt_corr_masks]
        gt_corr_indices = gt_corr_indices[gt_corr_masks]

        if gt_corr_indices.shape[0] > self.num_targets:
            indices = np.arange(gt_corr_indices.shape[0])
            sel_indices = np.random.choice(indices, self.num_targets, replace=False)
            sel_indices = torch.from_numpy(sel_indices).cuda()
            gt_corr_indices = gt_corr_indices[sel_indices]
            gt_corr_overlaps = gt_corr_overlaps[sel_indices]

        gt_ref_corr_indices = gt_corr_indices[:, 0]
        gt_src_corr_indices = gt_corr_indices[:, 1]

        return gt_ref_corr_indices, gt_src_corr_indices, gt_corr_overlaps


class FineMatching(nn.Module):
    def __init__(
        self,
        k: int,
        mutual: bool = True,
        confidence_threshold: float = 0.05,
        use_dustbin: bool = False,
        use_global_score: bool = False,
        correspondence_threshold: int = 3
    ):
        r"""Point Matching.
        Args:
            k (int): top-k selection for matching.
            mutual (bool=True): mutual or non-mutual matching.
            confidence_threshold (float=0.05): ignore matches whose scores are below this threshold.
            use_dustbin (bool=False): whether dustbin row/column is used in the score matrix.
            use_global_score (bool=False): whether use patch correspondence scores.
            correspondence_threshold (int=3): minimal number of correspondences for each patch correspondence.
        """
        super(FineMatching, self).__init__()
        self.k = k
        self.mutual = mutual
        self.confidence_threshold = confidence_threshold
        self.use_dustbin = use_dustbin
        self.use_global_score = use_global_score
        self.correspondence_threshold = correspondence_threshold

    def compute_correspondence_matrix(self, score_mat, ref_knn_masks, src_knn_masks):
        r"""Compute matching matrix and score matrix for each patch correspondence."""
        mask_mat = torch.logical_and(ref_knn_masks.unsqueeze(2), src_knn_masks.unsqueeze(1))

        batch_size, ref_length, src_length = score_mat.shape
        batch_indices = torch.arange(batch_size).cuda()

        # correspondences from reference side
        ref_topk_scores, ref_topk_indices = score_mat.topk(k=self.k, dim=2)  # (B, N, K)
        ref_batch_indices = batch_indices.view(batch_size, 1, 1).expand(-1, ref_length, self.k)  # (B, N, K)
        ref_indices = torch.arange(ref_length).cuda().view(1, ref_length, 1).expand(batch_size, -1, self.k)  # (B, N, K)
        ref_score_mat = torch.zeros_like(score_mat)
        ref_score_mat[ref_batch_indices, ref_indices, ref_topk_indices] = ref_topk_scores
        ref_corr_mat = torch.gt(ref_score_mat, self.confidence_threshold)

        # correspondences from source side
        src_topk_scores, src_topk_indices = score_mat.topk(k=self.k, dim=1)  # (B, K, N)
        src_batch_indices = batch_indices.view(batch_size, 1, 1).expand(-1, self.k, src_length)  # (B, K, N)
        src_indices = torch.arange(src_length).cuda().view(1, 1, src_length).expand(batch_size, self.k, -1)  # (B, K, N)
        src_score_mat = torch.zeros_like(score_mat)
        src_score_mat[src_batch_indices, src_topk_indices, src_indices] = src_topk_scores
        src_corr_mat = torch.gt(src_score_mat, self.confidence_threshold)

        # merge results from two sides
        if self.mutual:
            corr_mat = torch.logical_and(ref_corr_mat, src_corr_mat)
        else:
            corr_mat = torch.logical_or(ref_corr_mat, src_corr_mat)

        if self.use_dustbin:
            corr_mat = corr_mat[:, -1:, -1]

        corr_mat = torch.logical_and(corr_mat, mask_mat)

        return corr_mat


    def extract_correspondences(self, ref_knn_points, src_knn_points, score_mat, corr_mat):
        # extract dense correspondences
        batch_indices, ref_indices, src_indices = torch.nonzero(corr_mat, as_tuple=True)
        global_ref_corr_points = ref_knn_points[batch_indices, ref_indices]
        global_src_corr_points = src_knn_points[batch_indices, src_indices]
        global_corr_scores = score_mat[batch_indices, ref_indices, src_indices]
        return global_ref_corr_points, global_src_corr_points, global_corr_scores

    def forward(
        self,
        ref_knn_points,
        src_knn_points,
        ref_knn_masks,
        src_knn_masks,
        score_mat,
        global_scores,
    ):
        r"""Point Matching Module forward propagation.
        Args:
            ref_knn_points (Tensor): (B, K, 3)
            src_knn_points (Tensor): (B, K, 3)
            ref_knn_masks (BoolTensor): (B, K)
            src_knn_masks (BoolTensor): (B, K)
            score_mat (Tensor): (B, K, K) or (B, K + 1, K + 1), log likelihood
            global_scores (Tensor): (B,)
        Returns:
            ref_corr_points: torch.LongTensor (C, 3)
            src_corr_points: torch.LongTensor (C, 3)
            corr_scores: torch.Tensor (C,)
        """
        score_mat = torch.exp(score_mat)

        corr_mat = self.compute_correspondence_matrix(score_mat, ref_knn_masks, src_knn_masks)  # (B, K, K)

        if self.use_dustbin:
            score_mat = score_mat[:, :-1, :-1]
        if self.use_global_score:
            score_mat = score_mat * global_scores.view(-1, 1, 1)
        score_mat = score_mat * corr_mat.float()

        ref_corr_points, src_corr_points, corr_scores = self.extract_correspondences(
            ref_knn_points, src_knn_points, score_mat, corr_mat
        )

        return ref_corr_points, src_corr_points, corr_scores



class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(SinusoidalPositionalEmbedding, self).__init__()
        if d_model % 2 != 0:
            raise ValueError(f'Sinusoidal positional encoding with odd d_model: {d_model}')
        self.d_model = d_model
        div_indices = torch.arange(0, d_model, 2).float()
        div_term = torch.exp(div_indices * (-np.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, emb_indices):
        r"""Sinusoidal Positional Embedding.
        Args:
            emb_indices: torch.Tensor (*)
        Returns:
            embeddings: torch.Tensor (*, D)
        """
        input_shape = emb_indices.shape
        omegas = emb_indices.view(-1, 1, 1) * self.div_term.view(1, -1, 1)  # (-1, d_model/2, 1)
        sin_embeddings = torch.sin(omegas)
        cos_embeddings = torch.cos(omegas)
        embeddings = torch.cat([sin_embeddings, cos_embeddings], dim=2)  # (-1, d_model/2, 2)
        embeddings = embeddings.view(*input_shape, self.d_model)  # (*, d_model)
        embeddings = embeddings.detach()
        return embeddings


