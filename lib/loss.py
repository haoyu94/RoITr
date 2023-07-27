# Reference: https://github.com/qinzheng93/GeoTransformer

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import square_distance

def weighted_circle_loss(
    pos_masks,
    neg_masks,
    feat_dists,
    pos_margin,
    neg_margin,
    pos_optimal,
    neg_optimal,
    log_scale,
    pos_scales=None,
    neg_scales=None,
):
    # get anchors that have both positive and negative pairs
    row_masks = (torch.gt(pos_masks.sum(-1), 0) & torch.gt(neg_masks.sum(-1), 0)).detach()
    col_masks = (torch.gt(pos_masks.sum(-2), 0) & torch.gt(neg_masks.sum(-2), 0)).detach()

    # get alpha for both positive and negative pairs
    pos_weights = feat_dists - 1e5 * (~pos_masks).float()  # mask the non-positive
    pos_weights = pos_weights - pos_optimal  # mask the uninformative positive
    pos_weights = torch.maximum(torch.zeros_like(pos_weights), pos_weights)
    if pos_scales is not None:
        pos_weights = pos_weights * pos_scales
    pos_weights = pos_weights.detach()

    neg_weights = feat_dists + 1e5 * (~neg_masks).float()  # mask the non-negative
    neg_weights = neg_optimal - neg_weights  # mask the uninformative negative
    neg_weights = torch.maximum(torch.zeros_like(neg_weights), neg_weights)
    if neg_scales is not None:
        neg_weights = neg_weights * neg_scales
    neg_weights = neg_weights.detach()

    loss_pos_row = torch.logsumexp(log_scale * (feat_dists - pos_margin) * pos_weights, dim=-1)
    loss_pos_col = torch.logsumexp(log_scale * (feat_dists - pos_margin) * pos_weights, dim=-2)

    loss_neg_row = torch.logsumexp(log_scale * (neg_margin - feat_dists) * neg_weights, dim=-1)
    loss_neg_col = torch.logsumexp(log_scale * (neg_margin - feat_dists) * neg_weights, dim=-2)

    loss_row = F.softplus(loss_pos_row + loss_neg_row) / log_scale
    loss_col = F.softplus(loss_pos_col + loss_neg_col) / log_scale
    loss = (loss_row[row_masks].mean() + loss_col[col_masks].mean()) / 2

    return loss


class WeightedCircleLoss(nn.Module):
    def __init__(self, pos_margin, neg_margin, pos_optimal, neg_optimal, log_scale):
        super(WeightedCircleLoss, self).__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.pos_optimal = pos_optimal
        self.neg_optimal = neg_optimal
        self.log_scale = log_scale

    def forward(self, pos_masks, neg_masks, feat_dists, pos_scales=None, neg_scales=None):
        return weighted_circle_loss(
            pos_masks,
            neg_masks,
            feat_dists,
            self.pos_margin,
            self.neg_margin,
            self.pos_optimal,
            self.neg_optimal,
            self.log_scale,
            pos_scales=pos_scales,
            neg_scales=neg_scales,
        )


class CoarseMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(CoarseMatchingLoss, self).__init__()
        self.weighted_circle_loss = WeightedCircleLoss(
            cfg.coarse_loss_positive_margin,
            cfg.coarse_loss_negative_margin,
            cfg.coarse_loss_positive_optimal,
            cfg.coarse_loss_negative_optimal,
            cfg.coarse_loss_log_scale,
        )
        self.positive_overlap = cfg.coarse_loss_positive_overlap

    def forward(self, output_dict):
        tgt_feats = output_dict['tgt_node_feats']
        src_feats = output_dict['src_node_feats']

        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']


        gt_tgt_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]

        feat_dists = torch.sqrt(square_distance(tgt_feats[None, ::], src_feats[None, ::])[0])

        #print(tgt_feats)
        overlaps = torch.zeros_like(feat_dists)
        overlaps[gt_tgt_node_corr_indices, gt_src_node_corr_indices] = gt_node_corr_overlaps

        pos_masks = torch.gt(overlaps, self.positive_overlap)
        neg_masks = torch.eq(overlaps, 0)
        pos_scales = torch.sqrt(overlaps * pos_masks.float())

        loss = self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales)

        return loss


class FineMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(FineMatchingLoss, self).__init__()
        self.positive_radius = cfg.fine_loss_positive_radius

    def forward(self, output_dict, data_dict):
        tgt_node_corr_knn_points = output_dict['tgt_node_corr_knn_points']
        src_node_corr_knn_points = output_dict['src_node_corr_knn_points']
        tgt_node_corr_knn_masks = output_dict['tgt_node_corr_knn_masks']
        src_node_corr_knn_masks = output_dict['src_node_corr_knn_masks']
        matching_scores = output_dict['matching_scores']
        rot = data_dict['rot'][0]
        trans = data_dict['trans'][0]
        #src_node_corr_knn_points = apply_transform(src_node_corr_knn_points, transform)

        src_node_corr_knn_points = torch.matmul(src_node_corr_knn_points, rot.T) + (trans.T)[None, ::]
        dists = square_distance(tgt_node_corr_knn_points, src_node_corr_knn_points)  # (B, N, M)
        gt_masks = torch.logical_and(tgt_node_corr_knn_masks.unsqueeze(2), src_node_corr_knn_masks.unsqueeze(1))
        gt_corr_map = torch.lt(dists, self.positive_radius ** 2)
        gt_corr_map = torch.logical_and(gt_corr_map, gt_masks)
        slack_row_labels = torch.logical_and(torch.eq(gt_corr_map.sum(2), 0), tgt_node_corr_knn_masks)
        slack_col_labels = torch.logical_and(torch.eq(gt_corr_map.sum(1), 0), src_node_corr_knn_masks)
        labels = torch.zeros_like(matching_scores, dtype=torch.bool)
        labels[:, :-1, :-1] = gt_corr_map
        labels[:, :-1, -1] = slack_row_labels
        labels[:, -1, :-1] = slack_col_labels

        loss = -matching_scores[labels].mean()

        return loss


class OverallLoss(nn.Module):
    def __init__(self, cfg):
        super(OverallLoss, self).__init__()
        self.coarse_loss = CoarseMatchingLoss(cfg)
        self.fine_loss = FineMatchingLoss(cfg)
        self.weight_coarse_loss = cfg.coarse_loss_weight
        self.weight_fine_loss = cfg.fine_loss_weight
        self.weight_occ_loss = cfg.occ_loss_weight

    def forward(self, output_dict, data_dict):
        coarse_loss = self.coarse_loss(output_dict)
        fine_loss = self.fine_loss(output_dict, data_dict)

        loss = self.weight_coarse_loss * coarse_loss + self.weight_fine_loss * fine_loss

        return {
            'loss': loss,
            'c_loss': coarse_loss,
            'f_loss': fine_loss,
            'o_loss': 0. * fine_loss
        }


class Evaluator(nn.Module):
    def __init__(self, cfg):
        super(Evaluator, self).__init__()
        self.acceptance_overlap = cfg.eval_acceptance_overlap
        self.acceptance_radius = cfg.eval_acceptance_radius

    @torch.no_grad()
    def evaluate_coarse(self, output_dict):
        tgt_length_c = output_dict['tgt_nodes'].shape[0]
        src_length_c = output_dict['src_nodes'].shape[0]
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        masks = torch.gt(gt_node_corr_overlaps, self.acceptance_overlap)
        gt_node_corr_indices = gt_node_corr_indices[masks]
        gt_tgt_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]
        gt_node_corr_map = torch.zeros(tgt_length_c, src_length_c).cuda()
        gt_node_corr_map[gt_tgt_node_corr_indices, gt_src_node_corr_indices] = 1.0

        tgt_node_corr_indices = output_dict['tgt_node_corr_indices']
        src_node_corr_indices = output_dict['src_node_corr_indices']

        precision = gt_node_corr_map[tgt_node_corr_indices, src_node_corr_indices].mean()

        return precision

    @torch.no_grad()
    def evaluate_fine(self, output_dict, data_dict):
        rot, trans = data_dict['rot'][0], data_dict['trans'][0]
        tgt_corr_points = output_dict['tgt_corr_points']
        src_corr_points = output_dict['src_corr_points']
        if src_corr_points.shape[0] == 0:
            precision = 0.
        else:
            src_corr_points = torch.matmul(src_corr_points, rot.T) + trans.T
            corr_distances = torch.linalg.norm(tgt_corr_points - src_corr_points, dim=1)
            precision = torch.lt(corr_distances, self.acceptance_radius).float().mean()
        return precision

    def forward(self, output_dict, data_dict):
        c_precision = self.evaluate_coarse(output_dict)
        f_precision = self.evaluate_fine(output_dict, data_dict)
        return {
            'PIR': c_precision,
            'IR': f_precision
        }
