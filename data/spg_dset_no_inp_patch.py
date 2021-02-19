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
import nifty.graph.rag as nrag


class SpgDset(torch_data.Dataset):
    def __init__(self, root_dir,
                 spatial_augmentation=False,
                 intensity_augmentation=False,
                 noise_augmentation=False,
                 patched_output=True):
        """ dataset for loading images (raw, gt, superpixel segs) and according rags"""
        self.graph_dir = os.path.join(root_dir, 'graph_data')
        self.pix_dir = os.path.join(root_dir, 'pix_data')
        self.patched_output = patched_output
        self.graph_file_names = sorted(glob(os.path.join(self.graph_dir, "*.h5")))
        self.pix_file_names = sorted(glob(os.path.join(self.pix_dir, "*.h5")))
        self.spatial_augmentation = spatial_augmentation
        self.intensity_augmentation = intensity_augmentation
        self.noise_augmentation = noise_augmentation
        pix_file = h5py.File(self.pix_file_names[0], 'r')
        self.image_shape = pix_file["gt"][:].shape
        self.augm_tf = RndAugmentationTfs(self.image_shape, n_chnl_for_intensity=1)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img_idx = idx // np.prod(self.n_patch_per_dim) if self.patched_output else idx

        pix_file = h5py.File(self.pix_file_names[img_idx], 'r')
        graph_file = h5py.File(self.graph_file_names[img_idx], 'r')

        raw = pix_file["raw"][:]
        if raw.ndim == 2:
            raw = torch.from_numpy(raw.astype(np.float)).float().unsqueeze(0)
        else:
            raw = torch.from_numpy(raw.astype(np.float)).permute(2, 0, 1).float()
        raw -= raw.min()
        raw /= raw.max()
        gt = torch.from_numpy(pix_file["gt"][:].astype(np.long)).float()
        sp_seg = torch.from_numpy(graph_file["node_labeling"][:].astype(np.long)).float()
        affinities = torch.from_numpy(graph_file["affinities"][:]).float()

        # spat_tf, int_tf = self.augm_tf.sample(1, 1)
        # raw = int_tf(raw)

        # relabel to consecutive ints starting at 0
        mask = sp_seg[None] == torch.unique(sp_seg)[:, None, None]
        sp_seg = (mask * (torch.arange(len(torch.unique(sp_seg)), device=sp_seg.device)[:, None, None] + 1)).sum(0) - 1

        mask = gt[None] == torch.unique(gt)[:, None, None]
        gt = (mask * (torch.arange(len(torch.unique(gt)), device=gt.device)[:, None, None] + 1)).sum(0) - 1

        return raw, gt[None], sp_seg[None], affinities, torch.tensor([idx, img_idx])


if __name__ == "__main__":
    set = SpgDset()
    ret = set.get(3)
    a=1