import os
import torch
from tqdm import tqdm
from lib.trainer import Trainer
from lib.utils import to_o3d_pcd
from visualizer.visualizer import Visualizer, create_visualizer
from visualizer.feature_space import visualize_feature_space
import open3d as o3d
import numpy as np

class Tester(Trainer):
    '''
    Tester
    '''

    def __init__(self, config):
        Trainer.__init__(self, config)

    def test(self):
        print('Starting to evaluate on test datasets...')
        os.makedirs(f'{self.snapshot_dir}/{self.config.benchmark}', exist_ok=True)


        num_iter = len(self.loader['test'])

        c_loader_iter = self.loader['test'].__iter__()
        self.model.eval()
        with torch.no_grad():
            for idx in tqdm(range(num_iter)):
                torch.cuda.synchronize()
                inputs = c_loader_iter.next()

                #######################################
                # Load inputs to device
                for k, v in inputs.items():
                    if v is None:
                        pass
                    elif type(v) == list:
                        inputs[k] = [items.to(self.device) for items in v]
                    else:
                        inputs[k] = v.to(self.device)
                ##################
                # forward pass
                ##################
                rot, trans = inputs['rot'][0], inputs['trans'][0]
                src_pcd, tgt_pcd = inputs['src_points'].contiguous(), inputs['tgt_points'].contiguous()
                src_normals, tgt_normals = inputs['src_normals'].contiguous(), inputs[
                    'tgt_normals'].contiguous()
                src_feats, tgt_feats = inputs['src_feats'].contiguous(), inputs['tgt_feats'].contiguous()
                src_raw_pcd = inputs['raw_src_pcd'].contiguous()


                outputs = self.model.forward(src_pcd, tgt_pcd, src_feats, tgt_feats, src_normals, tgt_normals,
                                             rot, trans, src_raw_pcd)

                data = dict()
                data['src_raw_pcd'] = src_raw_pcd.cpu()
                data['src_pcd'], data['tgt_pcd'] = src_pcd.cpu(), tgt_pcd.cpu()
                data['src_nodes'], data['tgt_nodes'] = outputs['src_nodes'].cpu(), outputs['tgt_nodes'].cpu()
                data['src_node_desc'], data['tgt_node_desc'] = outputs['src_node_feats'].cpu().detach(), outputs['tgt_node_feats'].cpu().detach()
                data['src_point_desc'], data['tgt_point_desc'] = outputs['src_point_feats'].cpu().detach(), outputs['tgt_point_feats'].cpu().detach()
                data['src_corr_pts'], data['tgt_corr_pts'] = outputs['src_corr_points'].cpu(), outputs['tgt_corr_points'].cpu()
                data['confidence'] = outputs['corr_scores'].cpu().detach()
                data['gt_tgt_node_occ'] = outputs['gt_tgt_node_occ'].cpu()
                data['gt_src_node_occ'] = outputs['gt_src_node_occ'].cpu()
                data['rot'], data['trans'] = rot.cpu(), trans.cpu()
                if self.config.benchmark == '4DMatch' or self.config.benchmark == '4DLoMatch':
                    data['metric_index_list'] = inputs['metric_index']
                torch.save(data, f'{self.snapshot_dir}/{self.config.benchmark}/{idx}.pth')
                ###########################################################

    





def get_trainer(config):
    '''
    Get corresponding trainer according to the config file
    :param config:
    :return:
    '''

    if config.dataset == 'tdmatch' or config.dataset == 'fdmatch':
        return Tester(config)
    else:
        raise NotImplementedError
