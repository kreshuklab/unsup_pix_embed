import torch.nn as nn
import torch
from utils import get_valid_edges, get_naive_affinities
from skimage.filters import gaussian
import numpy as np
import matplotlib.pyplot as plt


class AugmentedAffinityContrastive(nn.Module):

    def __init__(self, delta_var, delta_dist, alpha=1.0, beta=1.0):
        super(AugmentedAffinityContrastive, self).__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.alpha = alpha
        self.beta = beta
        self.sep_chnl = 3
        self.sigma = 1.2
        self.overseg_factor = 1.
        self.offs = [[0, -1], [-1, 0], [-1, -1], [0, -2], [-2, 0], [-2, -2], [0, -3], [-3, 0], [-3, -3]]
        self.distance = lambda x, y, dim, kd=True: 1.0 - (x * y).sum(dim=dim, keepdim=kd)

    def forward(self, embeds, tf_embeds, raw, mask, *args):
        raw = raw[0]
        mask = mask[0]
        affs = get_naive_affinities(torch.from_numpy(gaussian(raw.permute(1, 2, 0).cpu(), self.sigma)).to(raw.device).permute(2, 0, 1), self.offs)

        # scale affinities in order to get an oversegmentation
        affs[:self.sep_chnl] /= self.overseg_factor
        affs[self.sep_chnl:] *= self.overseg_factor
        affs = torch.clamp(affs, 0, 1)

        loss = torch.tensor([0.0], device=embeds.device)
        for i, (off, aff, mask) in enumerate(zip(self.offs, affs, mask)):
            rolled = torch.roll(tf_embeds, tuple(-np.array(off)), (-2, -1))
            dist = self.distance(embeds, rolled, dim=1, kd=False) * mask[None]

            aff = aff - aff.min()
            aff = aff / aff.max()

            dist = dist * (0.5 - aff[None])

            loss = loss + dist.mean()

        return loss.mean()
