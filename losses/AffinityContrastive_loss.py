import torch.nn as nn
import torch
from utils import get_valid_edges, get_naive_affinities
from skimage.filters import gaussian
import numpy as np
import matplotlib.pyplot as plt


class AffinityContrastive(nn.Module):

    def __init__(self, delta_var, delta_dist, alpha=1.0, beta=1.0):
        super(AffinityContrastive, self).__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.alpha = alpha
        self.beta = beta
        self.sep_chnl = 3
        self.sigma = 1.2
        self.overseg_factor = 1.
        # self.offs = [[0, -1], [-1, 0], [-1, -1], [0, -2], [-2, 0], [-2, -2], [0, -3], [-3, 0], [-3, -3]]
        self.offs = [[1, 0], [0, 1], [2, 0], [0, 2], [3, 0], [0, 3], [4, 0], [0, 4], [8, 0], [0, 8], [16, 0], [0, 16]]
        self.offs_indices = [0, 1, 2, 3]
        self.distance = lambda x, y, dim, kd=True: 1.0 - (x * y).sum(dim=dim, keepdim=kd)

    def forward(self, embeds, affs, *args):
        # affs = get_naive_affinities(torch.from_numpy(gaussian(raw.permute(1, 2, 0).cpu(), self.sigma)).to(raw.device).permute(2, 0, 1), self.offs)
        affs *= -1
        affs += +1
        loss = torch.tensor([0.0], device=embeds.device)
        masks = torch.from_numpy(get_valid_edges([len(self.offs)] + list(embeds.shape[-2:]), self.offs)).to(embeds.device)
        masks = masks[self.offs_indices]
        offs = [self.offs[i] for i in self.offs_indices]
        for _embeds, _affs in zip(embeds, affs):
            for i, (off, aff, mask) in enumerate(zip(offs, _affs, masks)):
                rolled = torch.roll(_embeds, tuple(-np.array(off)), (-2, -1))
                dist = self.distance(_embeds, rolled, dim=0, kd=False) * mask
                dist = dist * (0.5 - aff)
                loss = loss + dist.mean()

        return loss.mean()
