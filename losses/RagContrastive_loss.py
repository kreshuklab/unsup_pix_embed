import torch.nn as nn
import torch
import nifty.graph.rag as nrag
import numpy as np


class RagContrastive(nn.Module):

    def __init__(self, delta_var, delta_dist, alpha=1.0, beta=1.0):
        super(RagContrastive, self).__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.alpha = alpha
        self.beta = beta
        self.sep_chnl = 2
        self.distance = lambda x, y, dim, kd=True: 1.0 - (x * y).sum(dim=dim, keepdim=kd)

    def forward(self, embeddings, sp_seg, affs, offs):
        # expecting first two args as N, C, D, H, W and third as 2, L. First should be l2-normalized w.r.t. C
        assert embeddings.ndim == sp_seg.ndim

        # get edge weights
        affs = affs[0]  # bs must be 1
        affs[self.sep_chnl:] *= -1
        affs[self.sep_chnl:] += +1
        rag = nrag.gridRag(sp_seg.squeeze(0).squeeze(0).long().cpu().numpy(), int(sp_seg.max()) + 1, numberOfThreads=1)
        edges = torch.from_numpy(rag.uvIds().astype(np.int)).to(affs.device).T
        hmap = (affs[0] + affs[1]) / 2
        hmap = (hmap - hmap.min()).float()
        hmap = hmap / (hmap.max() + 1e-6)
        weights = torch.from_numpy(nrag.accumulateEdgeStandartFeatures(rag, hmap.cpu().numpy(), 0, 1, numberOfThreads=1)).to(affs.device)

        C = int(sp_seg.max()) + 1
        mask = torch.zeros((C, ) + sp_seg.shape, dtype=torch.int8, device=sp_seg.device)
        mask.scatter_(0, sp_seg[None].long(), 1)
        masked_embeddings = mask * embeddings[None]
        n_pix_per_sp = mask.flatten(1).sum(1, keepdim=True)
        sp_means = masked_embeddings.transpose(1, 2).flatten(2).sum(2) / n_pix_per_sp

        intra_sp_dist = self.distance(sp_means[..., None, None, None], masked_embeddings.transpose(1, 2), dim=1)
        intra_sp_dist = torch.clamp(intra_sp_dist - self.delta_var, min=0) / n_pix_per_sp[..., None, None, None]
        intra_sp_dist = intra_sp_dist[mask.bool()].sum() / C
        edge_feats = sp_means[edges]
        inter_sp_dist = self.distance(edge_feats[0], edge_feats[1], dim=1, kd=False)
        inter_sp_dist = inter_sp_dist * weights[:, 0]
        inter_sp_dist = torch.clamp(self.delta_dist - inter_sp_dist, min=0)
        inter_sp_dist = inter_sp_dist.sum() / edges.shape[1]

        loss = self.alpha * inter_sp_dist + self.beta * intra_sp_dist
        return loss.mean()


class RagContrastiveWeights(nn.Module):

    def __init__(self, delta_var, delta_dist, alpha=1.0, beta=1.0, gamma=0.001):
        super(RagContrastiveWeights, self).__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.distance = lambda x, y, dim, kd=True: 1.0 - (x * y).sum(dim=dim, keepdim=kd)
        # self.distance = lambda x, y, dim, kd=True: torch.norm(x - y, dim=dim, keepdim=kd)

    def _compute_regularizer_term(self, cluster_means, C, ndim):
        # squeeze space dims
        for _ in range(ndim):
            cluster_means = cluster_means.squeeze(-1)
        norms = torch.norm(cluster_means, p=self.norm, dim=2)
        assert norms.size()[1] == C
        # return the average norm per batch
        return torch.sum(norms, dim=1).div(C)

    def forward(self, embeddings, sp_seg, edges, weights, chunks=4):
        # expecting first two args as N, C, D, H, W and third as 2, L. First should be l2-normalized w.r.t. C
        loss = torch.tensor([0.0], device=embeddings.device)

        for i, (s_embeddings, s_sp_seg, s_edges) in enumerate(zip(embeddings, sp_seg, edges)):
            C = int(s_sp_seg.max()) + 1
            sp_ids = torch.unique(sp_seg)

            slc_sz = C // chunks
            slices = [slice(slc_sz * step, slc_sz * (step + 1), 1) for step in range(chunks)]
            if C != chunks * slc_sz:
                slices.append(slice(slc_sz * chunks, C, 1))
            sp_means, intra_sp_dist = tuple(zip(*[self.get_mean_sp_embedding_and_dist(s_sp_seg, s_embeddings, sp_ids[slc], C) for slc in slices]))

            sp_means = torch.cat(sp_means, 1)
            intra_sp_dist = torch.cat(intra_sp_dist).sum()

            edge_feats = sp_means.T[s_edges]
            inter_sp_dist = self.distance(edge_feats[0], edge_feats[1], dim=1, kd=False)
            if weights is not None:
                inter_sp_dist = inter_sp_dist * weights[i]
            inter_sp_dist = torch.clamp(self.delta_dist - inter_sp_dist, min=0)
            inter_sp_dist = inter_sp_dist.sum() / s_edges.shape[1]

            # reg_term = torch.norm(sp_means, dim=0).mean()

            loss = loss + self.alpha * inter_sp_dist + self.beta * intra_sp_dist# + self.gamma * reg_term
        return loss.mean()

    def get_mean_sp_embedding_and_dist(self, s_sp_seg, s_embeddings, sp_ids, C):
        mask = sp_ids[:, None, None, None] == s_sp_seg[None]
        n_pix_per_sp = mask.flatten(1).sum(1)
        masked_embeddings = mask * s_embeddings[None]
        sp_means = masked_embeddings.transpose(0, 1).flatten(2).sum(2) / n_pix_per_sp[None]

        sp_means = sp_means / torch.clamp(torch.norm(sp_means, dim=0, keepdim=True), min=1e-10)

        intra_sp_dist = self.distance(sp_means[..., None, None], masked_embeddings.transpose(0, 1), dim=0, kd=False)
        intra_sp_dist = torch.clamp(intra_sp_dist - self.delta_var, min=0) / n_pix_per_sp[..., None, None]
        intra_sp_dist = intra_sp_dist[mask.bool().squeeze(1)].sum() / C

        return sp_means, intra_sp_dist[None]