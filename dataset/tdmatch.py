import torch.utils.data as data
import os
import torch
import numpy as np
from scipy.spatial.transform import Rotation
from dataset.common import collect_local_neighbors, build_ppf_patch, farthest_point_subsampling,\
    point2node_sampling, calc_patch_overlap_ratio, get_square_distance_matrix, calc_ppf_cpu, sample_gt_node_correspondence, calc_gt_patch_correspondence, normal_redirect
from lib.utils import to_o3d_pcd
import open3d as o3d


class TDMatchDataset(data.Dataset):
    '''
    Load subsampled coordinates, relative rotation and translation
    Output (torch.Tensor):
    src_pcd: (N, 3) source point cloud
    tgt_pcd: (M, 3) target point cloud
    src_node_xyz: (n, 3) nodes sparsely sampled from source point cloud
    tgt_node_xyz: (m, 3) nodes sparsely sampled from target point cloud
    rot: (3, 3)
    trans: (3, 1)
    correspondences: (?, 3)
    '''

    def __init__(self, infos, config, data_augmentation=True):
        super(TDMatchDataset, self).__init__()
        # information of data
        self.infos = infos
        # root dir
        self.base_dir = config.root
        # whether to do data augmentation
        self.data_augmentation = data_augmentation
        #self.data_augmentation = True
        # configurations
        self.config = config
        # factor used to control the maximum rotation during data augmentation
        self.rot_factor = 1.
        # maximum noise used in data augmentation
        self.augment_noise = config.augment_noise
        # the maximum number allowed in each single frame of point cloud
        self.points_lim = 30000
        # can be in ['train', 'val', 'test']
        self.mode = config.mode
        # original benchmark or rotated benchmark
        self.rotated = config.rotated
        # view point
        self.view_point = np.array([0., 0., 0.])


    def __getitem__(self, index):

        # get gt transformation
        rot = self.infos['rot'][index]
        trans = self.infos['trans'][index]
        # get original input point clouds
        src_path = os.path.join(self.base_dir, self.infos['src'][index])
        tgt_path = os.path.join(self.base_dir, self.infos['tgt'][index])
        # remove a dirty data
        if src_path.split('/')[-2] == '7-scenes-fire' and src_path.split('/')[-1] == 'cloud_bin_19.pth':
            index = (index + 1) % self.__len__()
            rot = self.infos['rot'][index]
            trans = self.infos['trans'][index]
            # get original input point clouds
            src_path = os.path.join(self.base_dir, self.infos['src'][index])
            tgt_path = os.path.join(self.base_dir, self.infos['tgt'][index])

        src_pcd = torch.load(src_path)
        tgt_pcd = torch.load(tgt_path)

        ##################################################################################################
        # if we get too many points, we do random down-sampling
        if src_pcd.shape[0] > self.points_lim:
            idx = np.random.permutation(src_pcd.shape[0])[:self.points_lim]
            src_pcd = src_pcd[idx]

        if tgt_pcd.shape[0] > self.points_lim:
            idx = np.random.permutation(tgt_pcd.shape[0])[:self.points_lim]
            tgt_pcd = tgt_pcd[idx]

        ##################################################################################################
        # whether to augment data for training / to rotate data for testing
        if self.data_augmentation:
            # rotate the point cloud
            euler_ab = np.random.rand(3) * np.pi * 2. / self.rot_factor  # anglez, angley, anglex
            rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
            if (np.random.rand(1)[0] > 0.5):
                src_pcd = np.matmul(rot_ab, src_pcd.T).T

                rot = np.matmul(rot, rot_ab.T)
            else:
                tgt_pcd = np.matmul(rot_ab, tgt_pcd.T).T

                rot = np.matmul(rot_ab, rot)
                trans = np.matmul(rot_ab, trans)
            # add noise
            src_pcd += (np.random.rand(src_pcd.shape[0], 3) - 0.5) * self.augment_noise
            tgt_pcd += (np.random.rand(tgt_pcd.shape[0], 3) - 0.5) * self.augment_noise
        # wheter test on rotated benchmark
        elif self.rotated:
            # rotate the point cloud
            np.random.seed(index)
            euler_ab = np.random.rand(3) * np.pi * 2. / self.rot_factor  # anglez, angley, anglex
            rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
            if (np.random.rand(1)[0] > 0.5):
                src_pcd = np.matmul(rot_ab, src_pcd.T).T

                rot = np.matmul(rot, rot_ab.T)
            else:
                tgt_pcd = np.matmul(rot_ab, tgt_pcd.T).T

                rot = np.matmul(rot_ab, rot)
                trans = np.matmul(rot_ab, trans)
        else:
            pass

        if (trans.ndim == 1):
            trans = trans[:, None]
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

        return src_pcd.astype(np.float32), tgt_pcd.astype(np.float32), \
               src_normals.astype(np.float32), tgt_normals.astype(np.float32),\
               src_feats.astype(np.float32), tgt_feats.astype(np.float32),\
               rot.astype(np.float32), trans.astype(np.float32),\
               src_pcd.astype(np.float32), None

    def __len__(self):
        return len(self.infos['rot'])
