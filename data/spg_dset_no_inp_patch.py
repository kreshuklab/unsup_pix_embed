import torch
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import h5py
import os
from glob import glob
from patch_manager import StridedRollingPatches2D, StridedPatches2D, NoPatches2D
from utils import squeeze_repr
import torch.utils.data as torch_data
import numpy as np
from transforms import RndAugmentationTfs, add_sp_gauss_noise
import torchvision
import nifty.graph.rag as nrag


class SpgDset(torch_data.Dataset):
    def __init__(self, root_dir, patch_manager="", patch_stride=None, patch_shape=None, reorder_sp=False,
                 spatial_augmentation=False,
                 intensity_augmentation=False,
                 noise_augmentation=False):
        """ dataset for loading images (raw, gt, superpixel segs) and according rags"""
        self.norm_tf = torchvision.transforms.Normalize(0, 1, inplace=False)
        self.graph_dir = os.path.join(root_dir, 'graph_data')
        self.pix_dir = os.path.join(root_dir, 'pix_data')
        self.graph_file_names = sorted(glob(os.path.join(self.graph_dir, "*.h5")))
        self.pix_file_names = sorted(glob(os.path.join(self.pix_dir, "*.h5")))
        self.reorder_sp = reorder_sp
        self.augm_tf = RndAugmentationTfs(patch_shape, n_chnl_for_intensity=1)
        self.spatial_augmentation = spatial_augmentation
        self.intensity_augmentation = intensity_augmentation
        self.noise_augmentation = noise_augmentation
        pix_file = h5py.File(self.pix_file_names[0], 'r')
        self.image_shape = pix_file["gt"][:].shape
        if patch_manager == "rotated":
            self.pm = StridedRollingPatches2D(patch_stride, patch_shape, self.image_shape)
        elif patch_manager == "no_cross":
            self.pm = StridedPatches2D(patch_stride, patch_shape, self.image_shape)
        else:
            self.pm = NoPatches2D()
        self.length = len(self.graph_file_names) * np.prod(self.pm.n_patch_per_dim)
        print('found ', self.length, " data files")

    def __len__(self):
        return self.length

    def viewItem(self, idx):
        pix_file = h5py.File(self.pix_file_names[idx], 'r')
        graph_file = h5py.File(self.graph_file_names[idx], 'r')

        raw = pix_file["raw"][:]
        gt = pix_file["gt"][:]
        sp_seg = graph_file["node_labeling"][:]

        fig, (a1, a2, a3) = plt.subplots(1, 3, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})
        a1.imshow(raw, cmap='gray')
        a1.set_title('raw')
        a2.imshow(cm.prism(gt/gt.max()))
        a2.set_title('gt')
        a3.imshow(cm.prism(sp_seg/sp_seg.max()))
        a3.set_title('sp')
        plt.tight_layout()
        plt.show()



    def __getitem__(self, idx):
        img_idx = idx // np.prod(self.pm.n_patch_per_dim)
        patch_idx = idx % np.prod(self.pm.n_patch_per_dim)
        pix_file = h5py.File(self.pix_file_names[img_idx], 'r')
        graph_file = h5py.File(self.graph_file_names[img_idx], 'r')

        raw = pix_file["raw"][:]
        if raw.ndim == 2:
            raw = torch.from_numpy(raw.astype(np.float)).float().unsqueeze(0)
        else:
            raw = torch.from_numpy(raw.astype(np.float)).permute(2, 0, 1).float()
        raw -= raw.min()
        raw /= raw.max()
        gt = torch.from_numpy(pix_file["gt"][:].astype(np.long)).unsqueeze(0).float()
        sp_seg = torch.from_numpy(graph_file["node_labeling"][:].astype(np.long)).unsqueeze(0).float()

        augm_or_not = torch.randint(0, 3, (3,))
        all = torch.cat([raw, gt, sp_seg], 0)
        spat_tf, int_tf = self.augm_tf.sample(1, 1)
        patch = self.pm.get_patch(all, patch_idx)
        if augm_or_not[0] == 0 and self.spatial_augmentation:
            patch = spat_tf(patch)

        if not self.reorder_sp:
            return patch[:-2], patch[-2].unsqueeze(0), patch[-1].unsqueeze(0), torch.tensor([img_idx, patch_idx])

        sp_seg = patch[-1].unsqueeze(0)
        gt = patch[-2].unsqueeze(0)
        new_sp_seg = torch.zeros_like(sp_seg)
        new_gt = torch.zeros_like(gt)
        for i, sp in enumerate(torch.unique(sp_seg)):
            new_sp_seg[sp_seg == sp] = i
        for i, obj in enumerate(torch.unique(gt)):
            new_gt[gt == obj] = i

        augm_raw = self.norm_tf(patch[:-2])
        if augm_or_not[1] == 0 and self.intensity_augmentation:
            augm_raw = int_tf(augm_raw)
        if augm_or_not[2] == 0 and self.noise_augmentation:
            augm_raw = add_sp_gauss_noise(augm_raw, 0.2, 0.1, 0.3)

        return augm_raw, new_gt, new_sp_seg, torch.tensor([idx, img_idx, patch_idx])


    def get_graphs(self, indices, patches, device="cpu"):
        edges, edge_feat, diff_to_gt, gt_edge_weights, affs = [], [], [], [], []
        for idx, patch in zip(indices, patches):
            img_idx, patch_idx = idx[0], idx[1]
            nodes = torch.unique(patch).unsqueeze(-1).unsqueeze(-1)
            graph_file = h5py.File(self.graph_file_names[img_idx], 'r')

            # get subgraph defined by patch overlap
            es = torch.from_numpy(graph_file["edges"][:]).to(device).sort(0)[0]
            iters_1 = (es.unsqueeze(0) == nodes).float().sum(0).sum(0) >= 2
            es = es[:, iters_1]
            nodes, es, patch = squeeze_repr(nodes.squeeze(-1).squeeze(-1), es, patch.squeeze(0))

            rag = nrag.gridRag(patch.squeeze(0).long().cpu().numpy(), int(patch.max()) + 1,
                               numberOfThreads=1)
            _edges = torch.from_numpy(rag.uvIds().astype(np.int)).to(device).T.sort(0)[0]
            iters_2 = ((es.unsqueeze(1) == _edges.unsqueeze(-1)).float().sum(0) == 2.0).sum(0) == 1
            es = es[:, iters_2]

            edges.append(es)
            edge_feat.append(torch.from_numpy(graph_file["edge_feat"][:]).to(device)[iters_1][iters_2])
            diff_to_gt.append(torch.tensor(graph_file["diff_to_gt"][()], device=device))
            gt_edge_weights.append(torch.from_numpy(graph_file["gt_edge_weights"][:]).to(device)[iters_1][iters_2])
            affs.append(self.pm.get_patch(torch.from_numpy(graph_file["affinities"][:]).to(device), patch_idx))

        return edges, edge_feat, diff_to_gt, gt_edge_weights, affs

if __name__ == "__main__":
    set = SpgDset()
    ret = set.get(3)
    a=1