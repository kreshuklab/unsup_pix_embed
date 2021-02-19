import torch.nn as nn
import torch
import nifty
import nifty.graph.agglo as nagglo
import numpy as np


class EntrInfoNCE(nn.Module):

    def __init__(self, alpha, beta, lbd, tau, gamma, num_neg, subs_size, p=2):
        super(EntrInfoNCE, self).__init__()
        self.smoothing_alpha = 5.0
        self.eps = 1e-10
        self.alpha = alpha
        self.beta = beta
        self.lbd = lbd
        self.tau = tau
        self.gamma = gamma
        self.num_neg = num_neg
        self.subs_size = subs_size
        self.prox = 40
        # self.p = p
        # self.distance = lambda x, y, dim: torch.norm(x-y, dim=dim, p=2)
        self.distance = lambda x, y, dim: 1.0 - (x * y).sum(dim=dim)
        self.similarity = lambda x, y, dim: 1.0 + (x * y).sum(dim=dim)

    def get_rand_proximity_ind(self, assign, shape):
        inds = []
        _assign = np.unravel_index(assign.detach().cpu().numpy(), shape)
        for i, ax in enumerate(shape):
            ind = np.random.randint(self.prox, ax - self.prox, size=(len(_assign[i]), self.num_neg))
            inds.append((_assign[i][:, None] + ind) % ax)
        return torch.from_numpy(np.ravel_multi_index(inds, shape)).to(assign.device)

    def graph_cluster(self, features, k, shape):
        # _centroid, label = kmeans2(mom_embed.cpu(), k, minit='random', iter=20)
        # build image grid graph
        graph = nifty.graph.undirectedGridGraph(shape)

        edge_weights = np.ones(graph.numberOfEdges, dtype=np.int)  # (beta=1) -> edge weights will not be used
        edge_sizes = np.ones(graph.numberOfEdges, dtype=np.int)  # won't be used as well
        node_sizes = np.ones(graph.numberOfNodes, dtype=np.int)  # nodes have equal initial masses
        # define cluster policy that does not use edge stats and that does not have higher merge affinities for small clusters
        # that policy is only available on this nifty fork https://github.com/paulhfu/nifty
        policy = nagglo.cosineDistNodeAndEdgeWeightedClusterPolicy(
            graph=graph,
            edgeIndicators=edge_weights,
            edgeSizes=edge_sizes,
            nodeFeatures=features.detach().cpu(),
            nodeSizes=node_sizes,
            numberOfNodesStop=k,
            beta=1,
            sizeRegularizer=0
        )
        clustering = nagglo.agglomerativeClustering(policy)
        clustering.run()

        label = clustering.result()
        label = torch.from_numpy(label.astype(np.int)).to(features.device).long()
        unique_lbl = torch.unique(label)  # using kmeans does not guarantee k clusters (has no effect when using agglomerative clustering)
        # relabel to consecutive labeling
        for new, lbl in enumerate(unique_lbl):
            label[label==lbl] = new
        return label

    def forward(self, embeddings, mom_embeddings, k, mask, whiten=False, warmup=False, psi=0.5):
        # inputs should be normed
        assert embeddings.ndim == 3
        shape = embeddings.shape[1:]
        embed, mom_embed, mask = embeddings.flatten(1).T, mom_embeddings.flatten(1).T, mask.flatten()
        # get some negatives
        sel_neg = self.get_rand_proximity_ind(torch.arange(embed.shape[0]), shape)
        sel_neg = mom_embed[sel_neg]
        # calculate the distancematrix to the positive and all the negatives of the momentum
        distances = self.similarity(embed[:, None], torch.cat([mom_embed[:, None], sel_neg], 1), -1)
        # distance exp scaled by entropy weight
        exp_dist = (distances/self.tau).exp()
        # get the loss, mask out void embeddings of the positives (negatives are allowed to be in the void)
        info_nce_loss = -(exp_dist[:, 0]/exp_dist.sum(-1, keepdim=True)).log() * mask
        if torch.isnan(info_nce_loss).any() or torch.isinf(info_nce_loss).any():
            halt = 1

        prot_info_nce_loss = torch.tensor([0.0], device=embed.device)
        if not warmup:
            # _centroid = torch.from_numpy(_centroid).to(embed.device)
            # get labels from graph agglo clustering
            label = self.graph_cluster(mom_embed, k, shape)
            unique_lbl, z = torch.unique(label, sorted=True, return_counts=True)
            # scatter the momentum embeddings to the labels
            _centroid = torch.zeros(((unique_lbl.shape[0],) + mom_embed.shape), device=embed.device).float()
            _centroid.scatter_(0, label.expand(mom_embed.shape[-1], label.shape[0]).T[None], mom_embed[None])
            # mean nonzero scattered embeddings along batch dim
            _centroid = _centroid.sum(-2)
            _centroid /= z[:, None]
            # normalize cluster centers
            _centroid = _centroid / torch.norm(_centroid, dim=-1, keepdim=True)
            centroid = _centroid[unique_lbl]  # remove empty clusters
            z = z[unique_lbl]

            if centroid.shape[0] != 0:
                assign = self.distance(centroid[None], embed[:, None], -1).min(1)[1]
                if self.num_neg < centroid.shape[0] - 1: # select some random negative clusters
                    neg = (torch.rand((embed.shape[0], self.num_neg), device=embed.device) * (centroid.shape[0] - 2)).round().long() + 1
                else: # or all negative clusters
                    neg = (torch.arange(centroid.shape[0] - 2, device=embed.device) + 1).expand((embed.shape[0], centroid.shape[0] - 2))

                sel_neg = (assign[:, None] + neg) % centroid.shape[0]
                # get negatives and positives
                neg_centroids = centroid[sel_neg]
                pos_centroids = centroid[assign]
                # calculate entropy weight. entropy should be high if the cluster has a low density
                mom_dists = self.distance(mom_embed, _centroid[label], -1)  # dist of each momentum embedding to its assigned cluster center
                # scatter that distances to its clusters
                scattered = torch.zeros(((_centroid.shape[0], ) + mom_dists.shape), device=embed.device).float()
                scattered.scatter_(0, label[None], mom_dists[None])
                scattered = scattered[unique_lbl]
                # calculate the weight. sigmoid for stability reasons. input to the exp should not be too large
                phi = torch.sigmoid(scattered.sum(-1) / (z * self.lbd)) - self.lbd

                prot_dists = self.similarity(embed[:, None], torch.cat([pos_centroids[:, None], neg_centroids], 1), -1)
                logit = (prot_dists / phi[assign][:, None]).exp()
                prot_info_nce_loss = -(logit[:, 0]/logit.sum(-1, keepdim=True)).log() * mask
                if torch.isnan(prot_info_nce_loss).any() or torch.isinf(prot_info_nce_loss).any():
                    halt=1

                # scattered = torch.zeros(((assign.max() + 1, ) + embed.shape), device=embed.device).float()
                # scattered.scatter_(0, assign.expand(tuple(reversed(embed.shape))).T[None], embed[None])
                # cluster_size = torch.bincount(assign)
                # non_empty_clusters = cluster_size > 0
                # cluster_centers = scattered[non_empty_clusters].sum(1) / cluster_size[non_empty_clusters, None]
                #
                # dist_matrix = cluster_centers.expand((cluster_centers.shape[0], ) + cluster_centers.shape)
                # dist_matrix = self.distance(dist_matrix, dist_matrix.transpose(0, 1), dim=-1)
                # dist_matrix = dist_matrix[torch.triu(torch.ones_like(dist_matrix), diagonal=1).bool()]
                # entropy_term = -(torch.softmax(-dist_matrix.detach(), 0) * dist_matrix).sum()

        loss = self.alpha * info_nce_loss.mean()  + self.beta * prot_info_nce_loss.mean()
        return loss
