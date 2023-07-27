import numpy as np
import os, copy
import open3d as o3d
import torch
from visualizer.feature_space import visualize_feature_space
from lib.utils import get_correspondences, matching_descriptors, to_o3d_pcd, to_tsfm, square_distance


class Visualizer(object):
    def __init__(self, src_pcd, tgt_pcd, src_nodes, tgt_nodes, src_patches, tgt_patches, src_node_desc, tgt_node_desc, src_point_desc, tgt_point_desc, rot, trans, to_mesh=False, offset=3.):
        self.src_pcd = (np.matmul(rot, src_pcd.T) + trans).T
        self.tgt_pcd = tgt_pcd
        self.tgt_pcd[:, 1] += offset

        self.src_nodes = (np.matmul(rot, src_nodes.T) + trans).T
        self.tgt_nodes = tgt_nodes
        self.tgt_nodes[:, 1] += offset

        self.src_patches = (np.matmul(rot, src_patches.reshape(-1, 3).T) + trans).T.reshape(tgt_patches.shape[0], tgt_patches.shape[1], 3)
        self.tgt_patches = tgt_patches
        self.tgt_patches[:, :, 1] += offset

        self.src_node_desc = src_node_desc
        self.tgt_node_desc = tgt_node_desc
        self.src_point_desc = src_point_desc
        self.tgt_point_desc = tgt_point_desc

        self.rot = rot
        self.trans = trans
        self.offset = offset
        self.vis_src_pcd = self.generate_open3d_vis_pcd_from_points(self.src_pcd, color=self.get_blue())
        self.vis_tgt_pcd = self.generate_open3d_vis_pcd_from_points(self.tgt_pcd, color=self.get_yellow())
        self.point_radius = 0.025
        self.node_radius = 0.05
        self.patch_overlap_radius = 0.0375
        self.to_mesh = to_mesh


        #self.correspondences = get_correspondences(to_o3d_pcd(src_nodes), to_o3d_pcd(tgt_nodes), to_tsfm(rot, trans), search_voxel_size=self.patch_overlap_radius)

        #if self.src_desc is not None and self.tgt_desc is not None:
        #    self.estimated_correspondences = matching_descriptors(src_desc, tgt_desc, mutual=False)
        #self.transformed_src_node = (np.matmul(self.rot, self.src_nodes.T) + self.trans).T
        #self.matching_mask = np.sqrt(square_distance(torch.from_numpy(self.transformed_src_node[np.newaxis, :]), torch.from_numpy(self.tgt_nodes[np.newaxis, :]))[0].numpy()) < self.patch_overlap_radius
    @staticmethod
    def get_random_color():
        '''
        Get a random color
        :return:
        '''
        color = list(np.random.choice(range(256), size=3))
        color = np.array(color).astype(np.float32) / 256.
        return color

    @staticmethod
    def get_blue():
        '''
        Get color blue
        :return:
        '''
        return np.array([0, 0.651, 0.929])

    @staticmethod
    def get_yellow():
        '''
        Get color yellow
        :return:
        '''
        return np.array([1, 0.706, 0])

    @staticmethod
    def get_red():
        '''
        Get color red
        :return:
        '''
        return np.array([1., 0., 0.])

    @staticmethod
    def get_green():
        '''
        Get color green
        :return:
        '''
        return np.array([0., 1., 0.])

    @staticmethod
    def mesh_sphere(pcd, voxel_size=0.0025, sphere_size=0.6):
        '''
        Convert point cloud to sphere mesh
        :param pcd:
        :param voxel_size:
        :param sphere_size:
        :return:
        '''
        # Create a mesh sphere
        spheres = o3d.geometry.TriangleMesh()
        s = o3d.geometry.TriangleMesh.create_sphere(radius=voxel_size * sphere_size)
        s.compute_vertex_normals()

        for i, p in enumerate(pcd.points):
            si = copy.deepcopy(s)
            trans = np.identity(4)
            trans[:3, 3] = p
            si.transform(trans)
            si.paint_uniform_color(pcd.colors[i])
            spheres += si
        return spheres

    @staticmethod
    def generate_open3d_vis_pcd_from_points(pcd, color=None):
        '''
        Convert points in numpy.ndarray to open3d format which can be visualized
        :return:
        '''
        vis_pcd = o3d.geometry.PointCloud()
        vis_pcd.points = o3d.utility.Vector3dVector(pcd)
        if len(color.shape) == 1:
            color = color[np.newaxis, :].repeat(pcd.shape[0], axis=0)
        elif color.shape[0] == pcd.shape[0]:
            pass
        else:
            raise NotImplementedError
        vis_pcd.colors = o3d.utility.Vector3dVector(color)
        return vis_pcd

    def show_alignment(self):
        '''
        Show the alignment of two frames
        :return:
        '''
        #src_pcd_mesh = self.mesh_sphere(self.vis_src_pcd, self.point_radius)
        #tgt_pcd_mesh = self.mesh_sphere(self.vis_tgt_pcd, self.point_radius)
        src_pcd_mesh = self.vis_src_pcd
        tgt_pcd_mesh = self.vis_tgt_pcd
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[-2, -2, -2])
        visualize_list = [src_pcd_mesh, tgt_pcd_mesh, axis]
        o3d.visualization.draw_geometries(visualize_list)

    def show_correspondences(self, correspondences=None, colors=None):
        '''
        Draw correspondences between two point clouds
        :param correspondences:
        :param color:
        :return:
        '''
        src_pcd_mesh = self.mesh_sphere(self.vis_src_pcd, self.point_radius)
        tgt_pcd_mesh = self.mesh_sphere(self.vis_tgt_pcd, self.point_radius)
        visualize_list = [src_pcd_mesh, tgt_pcd_mesh]

        src_nodes = self.src_nodes
        tgt_nodes = self.tgt_nodes

        vis_src_nodes = self.generate_open3d_vis_pcd_from_points(src_nodes, color=self.get_red())
        vis_tgt_nodes = self.generate_open3d_vis_pcd_from_points(tgt_nodes, color=self.get_red())
        src_node_mesh = self.mesh_sphere(vis_src_nodes, self.node_radius)
        visualize_list.append(src_node_mesh)
        tgt_node_mesh = self.mesh_sphere(vis_tgt_nodes, self.node_radius)
        visualize_list.append(tgt_node_mesh)

        if correspondences is None:
            lines = self.correspondences
            colors = np.repeat(self.get_green()[np.newaxis, :], lines.shape[0], axis=0)
        else:
            lines = correspondences

        points = np.concatenate((self.src_nodes, self.tgt_nodes), axis=0)
        lines[:, 1] += self.src_nodes.shape[0]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        visualize_list.append(line_set)
        o3d.visualization.draw_geometries(visualize_list)


    def show_learned_tsnf_feature_space(self):
        #print(self.src_pcd.shape, ' ', self.src_point_desc.shape, ' ', self.tgt_pcd.shape, ' ', self.tgt_point_desc.shape)
        visualize_feature_space(self.src_pcd, self.src_point_desc, self.tgt_pcd, self.tgt_point_desc)


    def show_pcd_with_nodes_and_one_patch(self, with_node=False, with_patch=None):
        if self.to_mesh:
            src_pcd_mesh = self.mesh_sphere(self.vis_src_pcd, self.point_radius)
            tgt_pcd_mesh = self.mesh_sphere(self.vis_tgt_pcd, self.point_radius)
        else:
            src_pcd_mesh = self.vis_src_pcd
            tgt_pcd_mesh = self.vis_tgt_pcd

        visualize_list = [src_pcd_mesh, tgt_pcd_mesh]
        if with_node:
            src_nodes = self.src_nodes
            tgt_nodes = self.tgt_nodes

            vis_src_nodes = self.generate_open3d_vis_pcd_from_points(src_nodes, color=self.get_red())
            vis_tgt_nodes = self.generate_open3d_vis_pcd_from_points(tgt_nodes, color=self.get_red())
            if self.to_mesh:
                src_node_mesh = self.mesh_sphere(vis_src_nodes, self.node_radius)
                tgt_node_mesh = self.mesh_sphere(vis_tgt_nodes, self.node_radius)
            else:

                src_node_mesh = self.mesh_sphere(vis_src_nodes, self.node_radius)
                tgt_node_mesh = self.mesh_sphere(vis_tgt_nodes, self.node_radius)
            visualize_list.append(src_node_mesh)
            visualize_list.append(tgt_node_mesh)

        if with_patch is not None:
            patch_id = with_patch
            src_patch = self.src_patches[patch_id]
            random_color = self.get_random_color()
            vis_src_patch = self.generate_open3d_vis_pcd_from_points(src_patch, random_color)
            tgt_patch = self.tgt_patches[patch_id]

            vis_tgt_patch = self.generate_open3d_vis_pcd_from_points(tgt_patch, random_color)
            if self.to_mesh:
                src_patch_mesh = self.mesh_sphere(vis_src_patch, self.point_radius)
                tgt_patch_mesh = self.mesh_sphere(vis_tgt_patch, self.point_radius)
            else:
                src_patch_mesh = self.mesh_sphere(vis_src_patch, self.point_radius)
                tgt_patch_mesh = self.mesh_sphere(vis_tgt_patch, self.point_radius)

            visualize_list.append(src_patch_mesh)
            visualize_list.append(tgt_patch_mesh)

        o3d.visualization.draw_geometries(visualize_list)

    def save_gt_correspondences(self, save_dir=None, idx=None):
        '''
        Draw correspondences between two point clouds
        :param correspondences:
        :param color:
        :return:
        '''
        src_nodes = self.src_nodes
        tgt_nodes = self.tgt_nodes

        src_node_color = np.zeros_like(src_nodes)
        src_node_color[:, 0] = 1
        tgt_node_color = np.zeros_like(tgt_nodes)
        tgt_node_color[:, 0] = 1
        src_node_color[self.correspondences[:, 0]] = np.array([[0, 1, 0]])
        tgt_node_color[self.correspondences[:, 1]] = np.array([[0, 1, 0]])

        vis_src_nodes = self.generate_open3d_vis_pcd_from_points(src_nodes, color=src_node_color)
        vis_tgt_nodes = self.generate_open3d_vis_pcd_from_points(tgt_nodes, color=tgt_node_color)

        src_node_mesh = self.mesh_sphere(vis_src_nodes, self.node_radius)
        o3d.io.write_triangle_mesh(os.path.join(save_dir, '{}_src_nodes.ply'.format(idx)), src_node_mesh)
        tgt_node_mesh = self.mesh_sphere(vis_tgt_nodes, self.node_radius)
        o3d.io.write_triangle_mesh(os.path.join(save_dir, '{}_tgt_nodes.ply'.format(idx)), tgt_node_mesh)


        lines = self.correspondences

        src_pcd = src_nodes[lines[:, 0]]
        tgt_pcd = tgt_nodes[lines[:, 1]]
        with open(os.path.join(save_dir, '{}_node_gt_correspondences.obj'.format(idx)), 'w') as f:
            for i in range(lines.shape[0]):
                f.write('v {} {} {} {} {} {}\n'.format(src_pcd[i, 0], src_pcd[i, 1], src_pcd[i, 2], 0, 255, 0))
                f.write('v {} {} {} {} {} {}\n'.format(tgt_pcd[i, 0], tgt_pcd[i, 1], tgt_pcd[i, 2], 0, 255, 0))

            for i in range(lines.shape[0]):
                f.write('l {} {}\n'.format(i + 1, i + 2))

    def save_est_correspondences(self, save_dir=None, idx=None):
        '''
        Draw correspondences between two point clouds
        :param correspondences:
        :param color:
        :return:
        '''
        src_nodes = self.src_nodes
        tgt_nodes = self.tgt_nodes
        src_node_color = np.zeros_like(src_nodes)
        src_node_color[:, 0] = 1
        tgt_node_color = np.zeros_like(tgt_nodes)
        tgt_node_color[:, 0] = 1
        row_major_corr = matching_descriptors(self.src_desc, self.tgt_desc, mutual=False, major='row')
        src_mask = self.matching_mask[row_major_corr[:, 0], row_major_corr[:, 1]]
        src_node_color[src_mask] = np.array([[0, 1, 0]])
        col_major_corr = matching_descriptors(self.src_desc, self.tgt_desc, mutual=False, major='col')
        tgt_mask = self.matching_mask[col_major_corr[:, 0], col_major_corr[:, 1]]
        tgt_node_color[tgt_mask] = np.array([[0, 1, 0]])


        vis_src_nodes = self.generate_open3d_vis_pcd_from_points(src_nodes, color=src_node_color)
        vis_tgt_nodes = self.generate_open3d_vis_pcd_from_points(tgt_nodes, color=tgt_node_color)
        src_node_mesh = self.mesh_sphere(vis_src_nodes, self.node_radius)
        o3d.io.write_triangle_mesh(os.path.join(save_dir, '{}_src_est_nodes.ply'.format(idx)), src_node_mesh)
        tgt_node_mesh = self.mesh_sphere(vis_tgt_nodes, self.node_radius)
        o3d.io.write_triangle_mesh(os.path.join(save_dir, '{}_tgt_est_nodes.ply'.format(idx)), tgt_node_mesh)

        lines = self.estimated_correspondences
        mask = self.matching_mask[lines[:, 0], lines[:, 1]] == 0

        src_pcd = src_nodes[lines[:, 0]]
        tgt_pcd = tgt_nodes[lines[:, 1]]

        with open(os.path.join(save_dir, '{}_node_est_correspondences.obj'.format(idx)), 'w') as f:
            for i in range(lines.shape[0]):
                if mask[i]:
                    f.write('v {} {} {} {} {} {}\n'.format(src_pcd[i, 0], src_pcd[i, 1], src_pcd[i, 2], 255, 0, 0))
                    f.write('v {} {} {} {} {} {}\n'.format(tgt_pcd[i, 0], tgt_pcd[i, 1], tgt_pcd[i, 2], 255, 0, 0))
                else:
                    f.write('v {} {} {} {} {} {}\n'.format(src_pcd[i, 0], src_pcd[i, 1], src_pcd[i, 2], 0, 255, 0))
                    f.write('v {} {} {} {} {} {}\n'.format(tgt_pcd[i, 0], tgt_pcd[i, 1], tgt_pcd[i, 2], 0, 255, 0))

            for i in range(lines.shape[0]):
                f.write('l {} {}\n'.format(i + 1, i + 2))

    def save_pcd_and_patch(self, save_dir=None, idx=None):
        src_pcd = self.generate_open3d_vis_pcd_from_points(self.src_pcd, color=self.get_blue())
        tgt_pcd = self.generate_open3d_vis_pcd_from_points(self.tgt_pcd, color=self.get_yellow())

        o3d.io.write_point_cloud(os.path.join(save_dir, '{}_src_points.ply'.format(idx)), src_pcd)
        o3d.io.write_point_cloud(os.path.join(save_dir, '{}_tgt_points.ply'.format(idx)), tgt_pcd)
        src_patches = np.empty(shape=(0, 3))
        src_colors = np.empty(shape=(0, 3))
        tgt_patches = np.empty(shape=(0, 3))
        tgt_colors = np.empty(shape=(0, 3))
        for patch_id in range(self.src_patches.shape[0]):
            src_patch = self.src_patches[patch_id]
            random_color = self.get_random_color()
            src_patches = np.concatenate((src_patches, src_patch), axis=0)
            src_colors = np.concatenate((src_colors, random_color[np.newaxis, :].repeat(src_patch.shape[0], axis=0)))
            tgt_patch = self.tgt_patches[patch_id]
            tgt_patches = np.concatenate((tgt_patches, tgt_patch), axis=0)
            tgt_colors = np.concatenate((tgt_colors, random_color[np.newaxis, :].repeat(tgt_patch.shape[0], axis=0)))
        vis_src_patches = self.generate_open3d_vis_pcd_from_points(src_patches, color=src_colors)
        vis_tgt_patches = self.generate_open3d_vis_pcd_from_points(tgt_patches, color=tgt_colors)
        o3d.io.write_point_cloud(os.path.join(save_dir, '{}_src_patches.ply'.format(idx)), vis_src_patches)
        o3d.io.write_point_cloud(os.path.join(save_dir, '{}_tgt_patches.ply'.format(idx)), vis_tgt_patches)


def create_visualizer(inputs, src_node_desc=None, tgt_node_desc=None, src_point_desc=None, tgt_point_desc=None, pcd_i=0, to_mesh=True, max_points=10000, offset=0.):
    '''
    Create visualizer from model input dict
    :param inputs: input dict, consists of torch.tensor on gpu
    :param pcd_i: index of point cloud to be visualized
    :return: created point cloud visualizer
    '''
    # get src point cloud
    if pcd_i == 0:
        src_start = 0
    else:
        src_start = inputs['src_lengths'][pcd_i - 1]
    src_end = inputs['src_lengths'][pcd_i]
    src_pcd = inputs['src_points'][src_start:src_end, :].cpu().numpy()

    # get tgt point cloud

    if pcd_i == 0:
        tgt_start = 0
    else:
        tgt_start = inputs['tgt_lengths'][pcd_i - 1]
    tgt_end = inputs['tgt_lengths'][pcd_i]
    tgt_pcd = inputs['tgt_points'][tgt_start:tgt_end, :].cpu().numpy()

    src_point_desc = src_point_desc[src_start:src_end, :].detach().cpu().numpy()
    tgt_point_desc = tgt_point_desc[tgt_start:tgt_end, :].detach().cpu().numpy()

    if src_pcd.shape[0] > max_points:
        src_ind_sel = np.random.choice(src_pcd.shape[0], max_points, replace=False)
        src_pcd = src_pcd[src_ind_sel, :]
        src_point_desc = src_point_desc[src_ind_sel, :]

    if tgt_pcd.shape[0] > max_points:
        tgt_ind_sel = np.random.choice(tgt_pcd.shape[0], max_points, replace=False)
        tgt_pcd = tgt_pcd[tgt_ind_sel, :]
        tgt_point_desc = tgt_point_desc[tgt_ind_sel, :]

    src_nodes = inputs['src_nodes'][pcd_i].cpu().numpy()
    tgt_nodes = inputs['tgt_nodes'][pcd_i].cpu().numpy()

    src_patch_xyz = inputs['src_patch_xyz'][pcd_i].cpu().numpy()
    tgt_patch_xyz = inputs['tgt_patch_xyz'][pcd_i].cpu().numpy()



    rot = inputs['rot'][pcd_i].cpu().numpy()
    trans = inputs['trans'][pcd_i].cpu().numpy()
    visualizer = Visualizer(src_pcd, tgt_pcd, src_nodes, tgt_nodes, src_patch_xyz, tgt_patch_xyz,
                            src_node_desc[pcd_i].detach().cpu().numpy(), tgt_node_desc[pcd_i].detach().cpu().numpy(),
                            src_point_desc, tgt_point_desc, rot, trans, to_mesh=to_mesh, offset=offset)
    return visualizer


def save_points(pcd, path=None):
    pcd = to_o3d_pcd(pcd)
    o3d.io.write_point_cloud(path, pcd)
