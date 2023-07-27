# Reference: https://github.com/POSTECH-CVLab/point-transformer

import torch
import numpy as np
import torch.nn as nn
from cpp_wrappers.pointops.functions import pointops
from lib.utils import calc_ppf_gpu, group_all, point_to_node_partition, get_node_correspondences, index_select, get_node_occlusion_score, to_o3d_pcd
import torch.nn.functional as F
from model.transformer.ppftransformer import LocalPPFTransformer, PPFTransformer
from model.transformer.geotransformer import GeometricTransformer
import open3d as o3d

class RIPointTransformerLayer(nn.Module):
    '''
    Rotation-invariant point transformer layer
    '''
    def __init__(self, in_planes, out_planes, num_heads=4, nsample=16, factor=1):
        super().__init__()
        self.nsample = nsample
        self.in_planes = in_planes
        self.output_planes = out_planes
        self.num_heas = num_heads
        self.nsample = nsample
        self.factor = factor
        self.transformer = LocalPPFTransformer(input_dim=in_planes, hidden_dim=min(out_planes, 256*factor), output_dim=out_planes, num_heads=num_heads)
        

    def forward(self, pxon, mask=None) -> torch.Tensor:
        p, x, o, n, idx, ppf_r = pxon  # (n, 3), (n, c), (b), (n, 4)
        if idx is None:
            group_idx = pointops.queryandgroup(self.nsample, p, p, p, idx, o, o, return_idx=True).long() #(n, nsample)
        else:
            group_idx = idx

        node_idx = torch.from_numpy(np.arange(p.shape[0])).to(p).long()


        p_r = p[group_idx, :]
        n_r = n[group_idx, :]

        if ppf_r is None:
            ppf_r = calc_ppf_gpu(p, n, p_r, n_r)  # (n, nsample, 4)
        x = self.transformer(x, node_idx, group_idx, ppf_r)
        return [x, group_idx, ppf_r]


class TransitionDown(nn.Module):
    '''
    Down-sampling
    '''
    def __init__(self, in_planes, out_planes, num_heads=4, stride=1, nsample=16, factor=1):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        self.transformer = LocalPPFTransformer(input_dim=in_planes, hidden_dim=min(out_planes, 256*factor), output_dim=out_planes, num_heads=num_heads)

    def forward(self, pxon):
        p, x, o, n, _, _, _ = pxon  # (n, 3), (n, c), (b), (n, 3)
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i - 1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)
            idx = pointops.furthestsampling(p, o, n_o).long()  # (m)


            n_p = p[idx, :]  # (m, 3)
            n_n = n[idx, :]  # (m, 3)
        else:
            n_o = o
            n_p = p
            n_n = n
            idx = torch.from_numpy(np.arange(p.shape[0])).to(n_o).long()

        group_idx = pointops.queryandgroup(self.nsample, p, n_p, p, None, o, n_o, return_idx=True).long()  # (m, nsample, 3 + 4 + c)
        c_p, c_n = p[group_idx, :], n[group_idx, :]
        ppf = calc_ppf_gpu(n_p, n_n, c_p, c_n) # (m, nsample, 4]

        x = self.transformer(x, idx, group_idx, ppf)
        return [n_p, x, n_o, n_n, None, None, idx]


class TransitionUp(nn.Module):
    '''
    Up-sampling
    '''
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(nn.Linear(2 * in_planes, in_planes), nn.LayerNorm(in_planes),
                                         nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes), nn.LayerNorm(out_planes),
                                         nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes), nn.LayerNorm(out_planes),
                                         nn.ReLU(inplace=True))

    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            _, x, o = pxo1  # (n, 3), (n, c), (b)
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i - 1], o[i], o[i] - o[i - 1]
                x_b = x[s_i:e_i, :]
                x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x)
        else:
            p1, x1, o1 = pxo1
            p2, x2, o2 = pxo2
            x = self.linear1(x1) + pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)
        return x


class RIPointTransformerBlock(nn.Module):
    '''
    Rotation-invariant point transformer block
    '''
    expansion = 1

    def __init__(self, in_planes, planes, num_heads=4, nsample=16, factor=1):
        super(RIPointTransformerBlock, self).__init__()
        self.transformer = RIPointTransformerLayer(in_planes, planes, num_heads, nsample, factor)
        self.bn2 = nn.LayerNorm(planes)

    def forward(self, pxon, mask=None):
        #print(len(pxon))
        p, x, o, n, idx, ppf_r, down_idx = pxon  # (n, 3), (n, c), (b), (n, 4)
        identity = x

        x, idx, ppf_r = self.transformer([p, x, o, n, idx, ppf_r], mask)
        #print(idx.dtype)
        x = self.bn2(x)
        x += identity
        x = F.relu(x)

        return [p, x, o, n, idx, ppf_r, down_idx]


class RIPointTransformer(nn.Module):
    def __init__(self, blocks=[2, 3, 3, 3], block=RIPointTransformerBlock, c=1, transformer_architecture=None, with_cross_pos_embed=None, factor=1, occ_thres=0.):
        super().__init__()
        self.c = c
        self.num_heads = 4
        self.in_planes, planes = c, [64*factor, 128*factor, 256*factor, 256*factor]
        stride, nsample = [1, 4, 4, 4], [8, 16, 16, 16]
        self.enc1 = self._make_enc(block, planes[0], blocks[0], self.num_heads, stride=stride[0], nsample=nsample[0], factor=factor)  # N/1

        self.enc2 = self._make_enc(block, planes[1], blocks[1], self.num_heads, stride=stride[1], nsample=nsample[1], factor=factor)  # N/4
        self.enc3 = self._make_enc(block, planes[2], blocks[2], self.num_heads, stride=stride[2], nsample=nsample[2], factor=factor)  # N/16
        self.enc4 = self._make_enc(block, planes[3], blocks[3], self.num_heads, stride=stride[3], nsample=nsample[3], factor=factor)  # N/64

        self.dec4 = self._make_dec(block, planes[3], 2, self.num_heads, nsample=nsample[3], factor=factor, is_head=True)  # fusion p5 and p4
        self.dec3 = self._make_dec(block, planes[2], 2, self.num_heads, nsample=nsample[2], factor=factor)  # fusion p4 and p3
        self.dec2 = self._make_dec(block, planes[1], 2, self.num_heads, nsample=nsample[1], factor=factor)  # fusion p3 and p2
        self.dec1 = self._make_dec(block, planes[0], 2, self.num_heads, nsample=nsample[0], factor=factor)  # fusion p2 and p1
        self.nsample = nsample
        self.transformer_architecture = transformer_architecture

        self.global_transformer = GeometricTransformer(256*factor, 256*factor, 256*factor, 4, self.transformer_architecture, sigma_d=0.2, sigma_a=15, angle_k=3, )
        self.occ_proj = nn.Linear(256*factor, 1)

        self.with_cross_pos_embed = with_cross_pos_embed #whether to use the learned rotation-invariant cross-frame positional representation


    def _make_enc(self, block, planes, blocks, num_heads=4, stride=1, nsample=16, factor=1):
        layers = []
        layers.append(TransitionDown(self.in_planes, planes, num_heads, stride, nsample, factor))
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, num_heads, nsample=nsample, factor=factor))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, factor=1, is_head=False):
        layers = []
        layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample, factor=factor))
        return nn.Sequential(*layers)

    def forward(self, s_pxon, t_pxon, src_deformed_pcd):

        s_p0, s_x0, s_o0, s_n0 = s_pxon  # (n, 3), (n, c), (b), (n, 4)
        t_p0, t_x0, t_o0, t_n0 = t_pxon  # (n, 3), (n, c), (b), (n, 4)
        ############################
        # encoder
        ############################
        # source pcd
        s_p1, s_x1, s_o1, s_n1, s_idx1, s_ppf1, _ = self.enc1([s_p0, s_x0, s_o0, s_n0, None, None, None])
        s_p2, s_x2, s_o2, s_n2, s_idx2, s_ppf2, s_d_idx2 = self.enc2([s_p1, s_x1, s_o1, s_n1, None, None, None])
        s_p3, s_x3, s_o3, s_n3, s_idx3, s_ppf3, s_d_idx3 = self.enc3([s_p2, s_x2, s_o2, s_n2, None, None, None])
        s_p4, s_x4, s_o4, s_n4, s_idx4, s_ppf4, s_d_idx4 = self.enc4([s_p3, s_x3, s_o3, s_n3, None, None, None])

        # target pcd
        t_p1, t_x1, t_o1, t_n1, t_idx1, t_ppf1, _ = self.enc1([t_p0, t_x0, t_o0, t_n0, None, None, None])
        t_p2, t_x2, t_o2, t_n2, t_idx2, t_ppf2, _ = self.enc2([t_p1, t_x1, t_o1, t_n1, None, None, None])
        t_p3, t_x3, t_o3, t_n3, t_idx3, t_ppf3, _ = self.enc3([t_p2, t_x2, t_o2, t_n2, None, None, None])
        t_p4, t_x4, t_o4, t_n4, t_idx4, t_ppf4, _ = self.enc4([t_p3, t_x3, t_o3, t_n3, None, None, None])
        ###########################
        # global PPF & transformer
        ###########################
        s_nr, s_pr = group_all(s_n4), group_all(s_p4)
        s_ppf = calc_ppf_gpu(s_p4, s_n4, s_pr, s_nr)  # (n, nsample, 4)

        t_nr, t_pr = group_all(t_n4), group_all(t_p4)
        t_ppf = calc_ppf_gpu(t_p4, t_n4, t_pr, t_nr)   # (n, nsample, 4)

        s_g_x4, t_g_x4 = self.global_transformer(s_p4.unsqueeze(0), t_p4.unsqueeze(0), s_x4.unsqueeze(0), t_x4.unsqueeze(0))
        s_g_x4 = s_g_x4[0]
        t_g_x4 = t_g_x4[0]

        ##########################
        # decoder
        ##########################

        # source pcd
        s_x4 = self.dec4[1:]([s_p4, self.dec4[0]([s_p4, s_x4, s_o4]), s_o4, s_n4, s_idx4, s_ppf4, None])[1]
        s_x3 = self.dec3[1:]([s_p3, self.dec3[0]([s_p3, s_x3, s_o3], [s_p4, s_x4, s_o4]), s_o3, s_n3, s_idx3, s_ppf3, None])[1]
        s_x2 = self.dec2[1:]([s_p2, self.dec2[0]([s_p2, s_x2, s_o2], [s_p3, s_x3, s_o3]), s_o2, s_n2, s_idx2, s_ppf2, None])[1]
        s_x1 = self.dec1[1:]([s_p1, self.dec1[0]([s_p1, s_x1, s_o1], [s_p2, s_x2, s_o2]), s_o1, s_n1, s_idx1, s_ppf1, None])[1]
        # target pcd
        t_x4 = self.dec4[1:]([t_p4, self.dec4[0]([t_p4, t_x4, t_o4]), t_o4, t_n4, t_idx4, t_ppf4, None])[1]
        t_x3 = self.dec3[1:]([t_p3, self.dec3[0]([t_p3, t_x3, t_o3], [t_p4, t_x4, t_o4]), t_o3, t_n3, t_idx3, t_ppf3, None])[1]
        t_x2 = self.dec2[1:]([t_p2, self.dec2[0]([t_p2, t_x2, t_o2], [t_p3, t_x3, t_o3]), t_o2, t_n2, t_idx2, t_ppf2, None])[1]
        t_x1 = self.dec1[1:]([t_p1, self.dec1[0]([t_p1, t_x1, t_o1], [t_p2, t_x2, t_o2]), t_o1, t_n1, t_idx1, t_ppf1, None])[1]

        s_d_idx3 = s_d_idx2[s_d_idx3.long()]
        s_d_idx4 = s_d_idx3[s_d_idx4.long()].long()
        s_p4 = src_deformed_pcd[s_d_idx4]

        return s_p4, s_g_x4, src_deformed_pcd, s_x1, t_p4, t_g_x4, t_p1, t_x1
