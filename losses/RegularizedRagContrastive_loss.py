import torch.nn as nn
import torch
import nifty.graph.rag as nrag
import numpy as np


class RegRagContrastiveWeights(nn.Module):

    def __init__(self, delta_var, delta_dist, slices, alpha=1.0, beta=1.0, gamma=0.001):
        super(RegRagContrastiveWeights, self).__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.dim_slices = slices
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

    def forward(self, embeddings, sp_seg, rot_sp, edges, weights, val_sp, rotation_angle, chunks=1):
        # expecting first two args as N, C, D, H, W and third as 2, L. First should be l2-normalized w.r.t. C
        loss = torch.tensor([0.0], device=embeddings.device)
        emb, rot_emb = embeddings.chunk(2, dim=0)

        for i, (s_emb, s_remb, s_sp, s_rsp, s_edges, sp_ids) in enumerate(zip(emb, rot_emb, sp_seg, rot_sp, edges, val_sp)):
            sp_means = []
            intra_sp_dist = torch.tensor([0.0], device=s_sp.device)
            s1_loss = torch.tensor([0.0], device=s_sp.device)
            s2_loss = torch.tensor([0.0], device=s_sp.device)

            C = len(sp_ids)
            chunks = max(chunks, 1)
            slc_sz = C // chunks
            slices = [slice(slc_sz * step, slc_sz * (step + 1), 1) for step in range(chunks)]
            if C != chunks * slc_sz:
                slices.append(slice(slc_sz * chunks, C, 1))

            for emb, sp in zip([s_emb, s_remb], [s_sp, s_rsp]):
                _sp_means, _intra_sp_dist = tuple(zip(*[self.get_mean_sp_embedding_and_dist(sp, emb, sp_ids[slc], C) for slc in slices]))
                _sp_means = torch.cat(_sp_means, 1)
                sp_means.append(_sp_means)
                intra_sp_dist = intra_sp_dist + torch.cat(_intra_sp_dist).sum()

            # space 1, is the space of texture and shape
            for _sp_means in sp_means:
                edge_feats = _sp_means[self.dim_slices[0]].T[s_edges]
                inter_sp_dist = self.distance(edge_feats[0], edge_feats[1], dim=1, kd=False)
                if weights is not None:
                    inter_sp_dist = inter_sp_dist * weights[i]
                inter_sp_dist = torch.clamp(self.delta_dist - inter_sp_dist, min=0)
                inter_sp_dist = inter_sp_dist.sum() / s_edges.shape[1]
                s1_loss = s1_loss + inter_sp_dist

            # space 1 should be rotation invariance
            rot_dist = self.distance(sp_means[0][:, self.dim_slices[0]], sp_means[1][:, self.dim_slices[0]], dim=1).mean()
            rot_dist = torch.clamp(rot_dist - self.delta_var, min=0)
            s1_loss = s1_loss + rot_dist

            # space 2 should measure the rotation
            rot_dist = self.distance(sp_means[0][:, self.dim_slices[1]], sp_means[1][:, self.dim_slices[1]], dim=1).mean()
            rot_dist = torch.clamp(self.delta_dist - rot_dist, min=0) * rotation_angle
            s2_loss = s2_loss + rot_dist
            loss = loss + self.alpha * (s1_loss + s2_loss) + self.beta * intra_sp_dist
        return loss.mean()

    def get_mean_sp_embedding_and_dist(self, s_sp_seg, s_embeddings, sp_ids, C):
        mask = sp_ids[:, None, None] == s_sp_seg[None]
        n_pix_per_sp = mask.flatten(1).sum(1)
        masked_embeddings = mask[:, None] * s_embeddings[None]
        sp_means = masked_embeddings.transpose(0, 1).flatten(2).sum(2) / n_pix_per_sp[None]

        sp_means = self.norm_each_space(sp_means, 0)

        intra_sp_dist = self.distance(sp_means[self.dim_slices[0], :, None, None], masked_embeddings.transpose(0, 1)[self.dim_slices[0]], dim=0, kd=False) + \
                        self.distance(sp_means[self.dim_slices[1], :, None, None], masked_embeddings.transpose(0, 1)[self.dim_slices[1]], dim=0, kd=False)
        intra_sp_dist = intra_sp_dist / 2
        intra_sp_dist = torch.clamp(intra_sp_dist - self.delta_var, min=0) / n_pix_per_sp[..., None, None]
        intra_sp_dist = intra_sp_dist[mask.bool()].sum() / C

        return sp_means, intra_sp_dist[None]

    def norm_each_space(self, embeddings, dim):
        _embeddings = []
        slcs = (slice(None),)*dim if dim > 0 else ()
        for slc in self.dim_slices:
            _slc = slcs + (slc,)
            _embeddings.append(embeddings[_slc] / torch.norm(embeddings[_slc], dim=dim, keepdim=True) + 1e-10)
        return torch.cat(_embeddings, dim)