import sys
import numpy as np
import random
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from lib.utils import read_entries, to_o3d_pcd
import open3d as o3d
from dataset.common import normal_redirect


class FDMatch(Dataset):
    def __init__(self, configs, split, data_augmentation=True):
        super(FDMatch, self).__init__()

        assert split in ['train', 'val', 'test']
        self.entries = read_entries(configs.split[split], configs.data_root, shuffle=False)

        self.base_dir = configs.data_root
        self.data_augmentation = data_augmentation
        self.config = configs

        self.max_points = 30000
        self.rot_factor = 1.
        self.augment_noise = configs.augment_noise

        self.overlap_radius = configs.fine_loss_positive_radius

        self.split = configs.split
        self.view_point = np.array([0., 0., 0.])

    def __getitem__(self, index):

        entry = np.load(self.entries[index])

        # get transformation
        rot = entry['rot']
        trans = entry['trans']
        s2t_flow = entry['s2t_flow']
        src_pcd = entry['s_pc']
        tgt_pcd = entry['t_pc']
        if "metric_index" in entry:
            metric_index = entry['metric_index'].squeeze()
        else:
            metric_index = None
        #correspondences = entry['correspondences'] # obtained with search radius 0.015m

        src_pcd_deformed = src_pcd + s2t_flow

        # if we get too many points, we do some downsampling
        if (src_pcd.shape[0] > self.max_points):
            idx = np.random.permutation(src_pcd.shape[0])[:self.max_points]
            src_pcd = src_pcd[idx]
            src_pcd_deformed = src_pcd_deformed[idx]
        if (tgt_pcd.shape[0] > self.max_points):
            idx = np.random.permutation(tgt_pcd.shape[0])[:self.max_points]
            tgt_pcd = tgt_pcd[idx]

        # add gaussian noise
        if self.data_augmentation:
            # rotate the point cloud
            euler_ab = np.random.rand(3) * np.pi * 2 / self.rot_factor  # anglez, angley, anglex
            rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
            if (np.random.rand(1)[0] > 0.5):
                src_pcd = np.matmul(rot_ab, src_pcd.T).T
                src_pcd_deformed = np.matmul(rot_ab, src_pcd_deformed.T).T
                rot = np.matmul(rot, rot_ab.T)
            else:
                tgt_pcd = np.matmul(rot_ab, tgt_pcd.T).T
                rot = np.matmul(rot_ab, rot)
                trans = np.matmul(rot_ab, trans)

            src_pcd += (np.random.rand(src_pcd.shape[0], 3) - 0.5) * self.augment_noise
            tgt_pcd += (np.random.rand(tgt_pcd.shape[0], 3) - 0.5) * self.augment_noise

        if trans.ndim == 1:
            trans = trans[:, None]

        rot = rot.astype(np.float32)
        trans = trans.astype(np.float32)

        ##################################################################################################
        # Normal estimation
        o3d_src_pcd = to_o3d_pcd(src_pcd)
        o3d_tgt_pcd = to_o3d_pcd(tgt_pcd)
        o3d_src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=33))
        src_normals = np.asarray(o3d_src_pcd.normals)
        src_normals = normal_redirect(src_pcd, src_normals, view_point=self.view_point)
        o3d_tgt_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=33))
        tgt_normals = np.asarray(o3d_tgt_pcd.normals)
        tgt_normals = normal_redirect(tgt_pcd, tgt_normals, view_point=self.view_point)
        src_feats = np.ones(shape=(src_pcd.shape[0], 1))
        tgt_feats = np.ones(shape=(tgt_pcd.shape[0], 1))

        return src_pcd_deformed.astype(np.float32), tgt_pcd.astype(np.float32), \
               src_normals.astype(np.float32), tgt_normals.astype(np.float32),\
               src_feats.astype(np.float32), tgt_feats.astype(np.float32),\
               rot.astype(np.float32), trans.astype(np.float32), \
               src_pcd.astype(np.float32), metric_index

    def __len__(self):
        return len(self.entries)
