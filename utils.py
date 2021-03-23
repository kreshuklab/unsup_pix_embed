import numpy as np
import torch
import elf
from torch import multiprocessing as mp
import math
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.cluster.vq import kmeans2, whiten, kmeans
import elf.segmentation.features as feats

def get_contour_from_2d_binary(mask: torch.Tensor):
    """
    :param mask: n_dim should be three (N|H|W). can be bool or long but should be binary if long.
    :return: tensor of the same shape and type bool containing all inner contours of objects in mask
    """
    max_p = torch.nn.MaxPool2d(3, stride=1, padding=1)
    return ((max_p(mask) != mask) | (-max_p(-mask) != mask)).long()

def get_valid_edges(shape, offsets):
    # compute valid edges
    ndim = len(offsets[0])
    image_shape = shape[1:]
    valid_edges = np.ones(shape, dtype=bool)
    for i, offset in enumerate(offsets):
        for j, o in enumerate(offset):
            inv_slice = slice(0, -o) if o < 0 else slice(image_shape[j] - o, image_shape[j])
            invalid_slice = (i, ) + tuple(slice(None) if j != d else inv_slice
                                          for d in range(ndim))
            valid_edges[invalid_slice] = 0
    return valid_edges

# Adjusts learning rate
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def pca_svd(X, k, center=True):
    # code from https://gist.github.com/project-delphi/e1112dbc0940d729a90f59846d25342b
    n = X.size()[0]
    ones = torch.ones(n).view([n, 1])
    h = ((1/n) * torch.mm(ones, ones.t())) if center else torch.zeros(n*n).view([n, n])
    H = torch.eye(n) - h
    X_center = torch.mm(H.double(), X.double())
    u, s, v = torch.svd(X_center)
    components = v[:k].t()
    explained_variance = torch.mul(s[:k], s[:k])/(n-1)  # remove normalization?
    return components, explained_variance

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def pca_project(embeddings, n_comps=3):
    assert embeddings.ndim == 3
    # reshape (C, H, W) -> (C, H * W) and transpose
    flattened_embeddings = embeddings.reshape(embeddings.shape[0], -1).T
    # init PCA with 3 principal components: one for each RGB channel
    pca = PCA(n_components=n_comps)
    # fit the model with embeddings and apply the dimensionality reduction
    flattened_embeddings = pca.fit_transform(flattened_embeddings)
    # reshape back to original
    shape = list(embeddings.shape)
    shape[0] = n_comps
    img = flattened_embeddings.T.reshape(shape)
    # normalize to [0, 255]
    img = 255 * (img - np.min(img)) / np.ptp(img)
    return np.moveaxis(img.astype('uint8'), 0, -1)

def pca_project_1d(embeddings, n_comps=3):
    assert embeddings.ndim == 2
    # reshape (C, H, W) -> (C, H * W) and transpose
    pca = PCA(n_components=n_comps)
    # fit the model with embeddings and apply the dimensionality reduction
    flattened_embeddings = pca.fit_transform(embeddings)
    # reshape back to original
    return flattened_embeddings.transpose()

def get_angles(x):
    """
        for a set of vectors this returns the angle [-pi, pi]
        of the vector with each vector in the unit othonormal basis.
        x should be a set of normalized vectors (NCHW)
    """
    ob = torch.eye(x.shape[1], device=x.device)
    return torch.acos(torch.matmul(ob[None, None, None], x.permute(0, 2, 3, 1)[..., None])).squeeze(-1).permute(0, 3, 1, 2)

def squeeze_repr(nodes, edges, seg):
    """
    This functions renames the nodes to [0,..,len(nodes)-1] in a superpixel rag consisting of nodes edges and a segmentation
    :param nodes: pt tensor
    :param edges: pt tensor
    :param seg: pt tensor
    :return: none
    """

    _nodes = torch.arange(0, len(nodes), device=nodes.device)
    indices = torch.where(edges.unsqueeze(0) == nodes.unsqueeze(-1).unsqueeze(-1))
    edges[indices[1], indices[2]] = _nodes[indices[0]]
    indices = torch.where(seg.unsqueeze(0) == nodes.unsqueeze(-1).unsqueeze(-1))
    seg[indices[1], indices[2]] = _nodes[indices[0]].float().type(seg.dtype)

    return nodes, edges, seg

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_naive_affinities(raw, offsets):
    """get naive pixel affinities based on differences in pixel intensities."""
    affinities = []
    for i, off in enumerate(offsets):
        rolled = torch.roll(raw, tuple(-np.array(off)), (-2, -1))
        dist = torch.norm(raw - rolled, dim=0)
        affinities.append(dist / dist.max())
    return torch.stack(affinities)

def get_edge_features_1d(sp_seg, offsets, affinities):
    offsets_3d = []
    for off in offsets:
        offsets_3d.append([0] + off)

    rag = feats.compute_rag(np.expand_dims(sp_seg, axis=0))
    edge_feat = feats.compute_affinity_features(rag, np.expand_dims(affinities, axis=1), offsets_3d)[:, :]
    return edge_feat, rag.uvIds()