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
import warnings
import torchvision
import nifty.graph.rag as nrag
from elf.segmentation.features import compute_rag

class LeptinDset(torch_data.Dataset):
    def __init__(self, raw_dir, affinity_dir, patch_manager="", patch_stride=None, patch_shape=None, reorder_sp=False):
        """ dataset for loading images (raw, gt, superpixel segs) and according rags"""
        self.transform = torchvision.transforms.Normalize(0, 1, inplace=False)
        self.affinity_dir = affinity_dir
        self.raw_dir = raw_dir
        self.affinity_file_names = []
        self.raw_file_names = sorted(glob(os.path.join(self.raw_dir, "*.h5")))

        self.sep_chnl = 2

        for i, fname in enumerate(self.raw_file_names):
            head, tail = os.path.split(fname)
            self.affinity_file_names.append(os.path.join(self.affinity_dir, tail[:-3] + '_predictions' + '.h5'))

        raw_file = h5py.File(self.raw_file_names[0], 'r')
        shape = raw_file["raw"][:].shape
        if patch_manager == "rotated":
            self.pm = StridedRollingPatches2D(patch_stride, patch_shape, shape)
        elif patch_manager == "no_cross":
            self.pm = StridedPatches2D(patch_stride, patch_shape, shape)
        else:
            self.pm = NoPatches2D()
        self.length = len(self.raw_file_names) * np.prod(self.pm.n_patch_per_dim)
        print('found ', self.length, " data files")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img_idx = idx // np.prod(self.pm.n_patch_per_dim)
        patch_idx = idx % np.prod(self.pm.n_patch_per_dim)
        raw_file = h5py.File(self.raw_file_names[img_idx], 'r')
        affinity_file = h5py.File(self.affinity_file_names[img_idx], 'r')

        raw = torch.from_numpy(raw_file["raw"][:].astype(np.float))
        wtsd = torch.from_numpy(raw_file["wtsd"][:].astype(np.float))
        raw -= raw.min()
        raw /= raw.max()
        affinities = torch.sigmoid(torch.from_numpy(affinity_file["predictions"][:]).squeeze(1))

        mask = wtsd[None] == torch.unique(wtsd)[:, None, None]
        wtsd = (mask * (torch.arange(len(torch.unique(wtsd)), device=wtsd.device)[:, None, None] + 1)).sum(0) - 1

        return raw[None].float(), wtsd.long(), affinities, torch.tensor([img_idx, patch_idx])


class LeptinValDset(torch_data.Dataset):
    def __init__(self, file):
        self.file = file
        self.length = 10

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        raw_file = h5py.File(self.file, 'r')

        raw = torch.from_numpy(raw_file["raw"][:].astype(np.float))
        gt = torch.from_numpy(raw_file["gt"][:].astype(np.float))
        raw -= raw.min()
        raw /= raw.max()

        mask = gt[None] == torch.unique(gt)[:, None, None]
        gt = (mask * (torch.arange(len(torch.unique(gt)), device=gt.device)[:, None, None] + 1)).sum(0) - 1
        raw_file.close()
        return raw[None].float(), gt.long()


if __name__=="__main__":
    slc_dict = {0: "4", 1: "43", 2: "97", 3: "125", 4: "165", 5: "201", 6: "230", 7: "239", 8: "291", 9: "308"}
    gt_file = h5py.File("/g/kreshuk/kaziakhm/plant_seg/processed_gt.h5", 'r')
    gts = torch.from_numpy(gt_file["label"][:].astype(np.long))
    raws = torch.from_numpy(gt_file["raw"][:].astype(np.long))
    write_dir = "/g/kreshuk/hilt/projects/data/leptin_fused_tp1_ch_0/true_val/raw_gt"
    for i in range(10):
        gt = gts[i]
        slc = slc_dict[i]

        # mask = gt[None] == torch.unique(gt)[:, None, None]
        # gt = (mask * (torch.arange(len(torch.unique(gt)), device=gt.device)[:, None, None] + 1)).sum(0) - 1

        pix_file = h5py.File(os.path.join(write_dir, "slice_" + slc + ".h5"), 'a')
        pix_file.create_dataset(name="label", data=gt)
        pix_file.create_dataset(name="raw", data=raws[i])
        pix_file.close()
