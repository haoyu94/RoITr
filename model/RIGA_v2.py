import torch

from model.model import *
from model.modules import CoarseMatching, AdaptiveSuperPointMatching, GTCoarseCorrGenerator, FineMatching, LearnableLogOptimalTransport
import torch.nn.functional as F
from lib.utils import to_o3d_pcd
import open3d as o3d


class RIGA_v2(nn.Module):
    '''
    The RoITr pipeline
    '''
    def __init__(self, config):
        super(RIGA_v2, self).__init__()
        self.config = config
        # backbone network
        self.with_cross_pos_embed = config.with_cross_pos_embed
        self.benchmark = config.benchmark

        if self.benchmark in ['3DMatch', '3DLoMatch']:

            self.coarse_matching = CoarseMatching(num_correspondences=config.num_est_coarse_corr, dual_normalization=True)
            self.factor = 1
        else:
            
            self.coarse_matching = AdaptiveSuperPointMatching(min_num_correspondences=config.num_est_coarse_corr, similarity_threshold=0.75)
            self.factor = 2

        self.backbone = RIPointTransformer(transformer_architecture=config.transformer_architecture, with_cross_pos_embed=self.with_cross_pos_embed, factor=self.factor)
        # learnable Optimal Transport Layer
        self.OT = LearnableLogOptimalTransport(num_iter=100)
        # the number of correspondences used for each point cloud pair during training

        self.mode = config.mode # current phase, should be in ['train', 'val', 'test']
        self.point_per_patch = config.point_per_patch
        self.matching_radius = config.matching_radius
        
        # coarse level final descriptor projection
        self.coarse_proj = nn.Linear(256*self.factor, 256*self.factor)
        # fine level final descriptor projection
        self.fine_proj = nn.Linear(64*self.factor, 256*self.factor)



        self.coarse_generator = GTCoarseCorrGenerator(num_targets=config.num_gt_coarse_corr, overlap_threshold=config.coarse_overlap_threshold)
        self.fine_matching = FineMatching(config.fine_matching_topk,
                                          mutual=config.fine_matching_mutual, confidence_threshold=config.fine_matching_confidence_threshold,
                                          use_dustbin=config.fine_matching_use_dustbin,
                                          use_global_score=config.fine_matching_use_global_score,
                                          correspondence_threshold=config.fine_matching_correspondence_threshold)

        self.fine_matching_use_dustbin = config.fine_matching_use_dustbin

        self.optimal_transport = LearnableLogOptimalTransport(num_iter=100)


    def forward(self, src_pcd, tgt_pcd, src_feats, tgt_feats, src_normals, tgt_normals, rot, trans, src_raw_pcd):
        src_o, tgt_o = torch.from_numpy(np.array([src_raw_pcd.shape[0]])).to(src_raw_pcd).int(), torch.from_numpy(np.array([tgt_pcd.shape[0]])).to(tgt_pcd).int()
        output_dict = {}
        # 1. get descriptors
        src_node_xyz, src_node_feats, src_pcd, src_point_feats, tgt_node_xyz, tgt_node_feats, tgt_pcd, tgt_point_feats = self.backbone([src_raw_pcd, src_feats, src_o, src_normals], [tgt_pcd, tgt_feats, tgt_o, tgt_normals], src_pcd)

        src_node_feats = F.normalize(self.coarse_proj(src_node_feats), p=2, dim=1)
        tgt_node_feats = F.normalize(self.coarse_proj(tgt_node_feats), p=2, dim=1)

        src_point_feats = self.fine_proj(src_point_feats)
        tgt_point_feats = self.fine_proj(tgt_point_feats)

        output_dict['src_points'] = src_pcd
        output_dict['tgt_points'] = tgt_pcd
        output_dict['src_nodes'] = src_node_xyz
        output_dict['tgt_nodes'] = tgt_node_xyz

        output_dict['src_point_feats'] = src_point_feats
        output_dict['tgt_point_feats'] = tgt_point_feats
        output_dict['src_node_feats'] = src_node_feats
        output_dict['tgt_node_feats'] = tgt_node_feats


        # 2. get ground truth node correspondences
        _, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(src_pcd, src_node_xyz, point_limit=self.point_per_patch)
        _, tgt_node_masks, tgt_node_knn_indices, tgt_node_knn_masks = point_to_node_partition(tgt_pcd, tgt_node_xyz, point_limit=self.point_per_patch)


        src_padded_points = torch.cat([src_pcd, torch.zeros_like(src_pcd[:1])], dim=0)
        tgt_padded_points = torch.cat([tgt_pcd, torch.zeros_like(tgt_pcd[:1])], dim=0)
        src_node_knn_points = index_select(src_padded_points, src_node_knn_indices, dim=0)
        tgt_node_knn_points = index_select(tgt_padded_points, tgt_node_knn_indices, dim=0)

        gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
            tgt_node_xyz,
            src_node_xyz,
            tgt_node_knn_points,
            src_node_knn_points,
            rot, trans,
            self.matching_radius,
            ref_masks=tgt_node_masks,
            src_masks=src_node_masks,
            ref_knn_masks=tgt_node_knn_masks,
            src_knn_masks=src_node_knn_masks
        )
        gt_tgt_node_occ_score, gt_src_node_occ_score = get_node_occlusion_score(tgt_node_knn_indices,
                                                                                src_node_knn_indices,
                                                                                tgt_padded_points,
                                                                                src_padded_points,
                                                                                rot, trans,
                                                                                ref_masks=tgt_node_masks,
                                                                                src_masks=src_node_masks,
                                                                                ref_knn_masks=tgt_node_knn_masks,
                                                                                src_knn_masks=src_node_knn_masks)

        output_dict['gt_node_corr_indices'] = gt_node_corr_indices
        output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps
        output_dict['gt_tgt_node_occ'] = gt_tgt_node_occ_score
        output_dict['gt_src_node_occ'] = gt_src_node_occ_score

        # 3. select topk node correspondences
        with torch.no_grad():

            tgt_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(tgt_node_feats, src_node_feats, tgt_node_masks, src_node_masks)
            output_dict['src_node_corr_indices'] = src_node_corr_indices
            output_dict['tgt_node_corr_indices'] = tgt_node_corr_indices

            if self.training:
                tgt_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_generator(gt_node_corr_indices, gt_node_corr_overlaps.contiguous())

        # 4. Generate batched node points & feats
        src_node_corr_knn_indices = src_node_knn_indices[src_node_corr_indices]  # (P, K)
        tgt_node_corr_knn_indices = tgt_node_knn_indices[tgt_node_corr_indices]  # (P, K)

        src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices]  # (P, K)
        tgt_node_corr_knn_masks = tgt_node_knn_masks[tgt_node_corr_indices]  # (P, K)

        src_node_corr_knn_points = src_node_knn_points[src_node_corr_indices]  # (P, K, 3)
        tgt_node_corr_knn_points = tgt_node_knn_points[tgt_node_corr_indices]  # (P, K, 3)

        src_padded_point_feats = torch.cat([src_point_feats, torch.zeros_like(src_point_feats[:1])], dim=0)
        tgt_padded_point_feats = torch.cat([tgt_point_feats, torch.zeros_like(tgt_point_feats[:1])], dim=0)

        src_node_corr_knn_feats = index_select(src_padded_point_feats, src_node_corr_knn_indices, dim=0)  # (P, K, C)
        tgt_node_corr_knn_feats = index_select(tgt_padded_point_feats, tgt_node_corr_knn_indices, dim=0)  # (P, K, C)

        output_dict['src_node_corr_knn_points'] = src_node_corr_knn_points
        output_dict['tgt_node_corr_knn_points'] = tgt_node_corr_knn_points
        output_dict['src_node_corr_knn_masks'] = src_node_corr_knn_masks
        output_dict['tgt_node_corr_knn_masks'] = tgt_node_corr_knn_masks

        # 5. Optimal transport
        matching_scores = torch.einsum('bnd,bmd->bnm', tgt_node_corr_knn_feats,
                                       src_node_corr_knn_feats)  # (P, K, K)
        matching_scores = matching_scores / src_point_feats.shape[1] ** 0.5
        matching_scores = self.optimal_transport(matching_scores, tgt_node_corr_knn_masks, src_node_corr_knn_masks)

        output_dict['matching_scores'] = matching_scores # row: tgt, col: src

        # 6. Generate final correspondences during testing
        with torch.no_grad():
            if not self.fine_matching_use_dustbin:
                matching_scores = matching_scores[:, :-1, :-1]

            tgt_corr_points, src_corr_points, corr_scores = self.fine_matching(
                tgt_node_corr_knn_points,
                src_node_corr_knn_points,
                tgt_node_corr_knn_masks,
                src_node_corr_knn_masks,
                matching_scores,
                node_corr_scores
            )

            output_dict['tgt_corr_points'] = tgt_corr_points
            output_dict['src_corr_points'] = src_corr_points
            output_dict['corr_scores'] = corr_scores

        return output_dict


def create_model(config):
    model = RIGA_v2(config)
    return model

