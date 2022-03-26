import os

import h5py
import vigra

import numpy as np
import elf.segmentation.features as feats
from skimage import draw
from skimage.filters import gaussian
from affogato.segmentation import compute_mws_segmentation
from elf.segmentation.features import project_node_labels_to_pixels
from elf.segmentation.multicut import multicut_decomposition
import matplotlib.pyplot as plt
from matplotlib import cm

##############################################################################################################
# This script generates a segmentation task of segmenting artificially generated circles and rectangles.
# The environment necessary to execute this script can be setup with: conda install -c conda-forge python-elf
##############################################################################################################

def get_multicut_sln(rag, edge_weights):
    p_min = 1e-5
    p_max = 1.
    probs = edge_weights.copy()
    costs = (p_max - p_min) * probs + p_min
    costs = (np.log((1. - costs) / costs.clip(1e-10, 1)))
    node_labels = multicut_decomposition(rag, costs, internal_solver='kernighan-lin', n_threads=4)  # greedy-additive
    node_labels = vigra.analysis.relabelConsecutive(node_labels, keep_zeros=False, start_label=0)[0]
    return project_node_labels_to_pixels(rag, node_labels).squeeze()

def get_edge_features_1d(sp_seg, offsets, affinities):
    sp_seg -= sp_seg.min()
    offsets_3d = []
    for off in offsets:
        offsets_3d.append([0] + off)

    rag = feats.compute_rag(np.expand_dims(sp_seg, axis=0))
    edge_feat = feats.compute_affinity_features(rag, np.expand_dims(affinities, axis=1), offsets_3d)[:, :]
    return edge_feat, rag

def calculate_gt_edge_costs(neighbors, new_seg, gt_seg, thresh):
    gt_edges = np.zeros(len(neighbors))
    new_seg += 1
    neighbors += 1
    gt_seg += 1

    for idx, neighbor in enumerate(neighbors):
        mask_n1, mask_n2 = new_seg == neighbor[0], new_seg == neighbor[1]
        mask = mask_n1 + mask_n2
        mskd_gt_seg = mask * gt_seg
        mskd_new_seg = mask * new_seg
        n_obj_gt = np.unique(mskd_gt_seg)
        n_obj_new = np.unique(mskd_new_seg)
        n_obj_gt = n_obj_gt[1:] if n_obj_gt[0] == 0 else n_obj_gt
        if len(n_obj_gt) == 1:
            gt_edges[idx] = 0
        else:
            n_obj_new = n_obj_new[1:] if n_obj_new[0] == 0 else n_obj_new
            assert len(n_obj_new) == 2
            overlaps =  np.zeros((len(n_obj_gt), 2))
            for j, obj in enumerate(n_obj_gt):
                mask_gt = mskd_gt_seg == obj
                overlaps[j, 0] = np.sum(mask_gt * mask_n1) / np.sum(mask_n1)
                overlaps[j, 1] = np.sum(mask_gt * mask_n2) / np.sum(mask_n2)
            if np.sum(overlaps.max(axis=1) > thresh) >= 2:
                gt_edges[idx] = 1
            else:
                gt_edges[idx] = 0
    new_seg -= 1
    neighbors -= 1
    gt_seg -= 1
    return gt_edges

def get_naive_affinities(raw, offsets):
    """get naive pixel affinities based on differences in pixel intensities."""
    affinities = []
    for i, off in enumerate(offsets):
        rolled = np.roll(raw, tuple(-np.array(off)), (0, 1))
        dist = np.linalg.norm(raw - rolled, axis=-1)
        affinities.append(dist / dist.max())
    return np.stack(affinities)

def get_pix_data(shape=(256, 256)):
    """ This generates raw-gt-superpixels and correspondinng rags of rectangles and circles"""

    rsign = lambda: (-1)**np.random.randint(0, 2)
    edge_offsets = [[0, -1], [-1, 0], [-3, 0], [0, -3], [-6, 0], [0, -6]]  # offsets defining the edges for pixel affinities
    overseg_factor = 1.7
    sep_chnl = 2  # channel separating attractive from repulsive edges
    n_circles = 5  # number of ellipses in image
    n_polys = 10  # number of rand polys in image
    n_rect = 5  # number rectangles in image
    circle_color = np.array([1, 0, 0], dtype=float)
    rect_color = np.array([0, 0, 1], dtype=float)
    col_diff = 0.4  # by this margin object color can vary ranomly
    min_r, max_r = 10, 20  # min and max radii of ellipses/circles
    min_dist = max_r

    img = np.random.randn(*(shape + (3,))) / 5  # init image with some noise
    gt = np.zeros(shape)

    #  get some random frequencies
    ri1, ri2, ri3, ri4, ri5, ri6 = rsign() * ((np.random.rand() * 2) + .5), \
                                   rsign() * ((np.random.rand() * 2) + .5), \
                                   (np.random.rand() * 4) + 3, \
                                   (np.random.rand() * 4) + 3, \
                                   rsign() * ((np.random.rand() * 2) + .5), \
                                   rsign() * ((np.random.rand() * 2) + .5)
    x = np.zeros(shape)
    x[:, :] = np.arange(img.shape[0])[np.newaxis, :]
    y = x.transpose()
    # add background frequency interferences
    img += (np.sin(np.sqrt((x * ri1) ** 2 + ((shape[1] - y) * ri2) ** 2) * ri3 * np.pi / shape[0]))[
        ..., np.newaxis]
    img += (np.sin(np.sqrt((x * ri5) ** 2 + ((shape[1] - y) * ri6) ** 2) * ri4 * np.pi / shape[1]))[
        ..., np.newaxis]
    # smooth a bit
    img = gaussian(np.clip(img, 0.1, 1), sigma=.8)
    # add some circles
    circles = []
    cmps = []
    while len(circles) < n_circles:
        mp = np.random.randint(min_r, shape[0] - min_r, 2)
        too_close = False
        for cmp in cmps:
            if np.linalg.norm(cmp - mp) < min_dist:
                too_close = True
        if too_close:
            continue
        r = np.random.randint(min_r, max_r, 2)
        circles.append(draw.disk((mp[0], mp[1]), r[0], shape=shape))
        cmps.append(mp)

    # add some random polygons
    polys = []
    while len(polys) < n_polys:
        mp = np.random.randint(min_r, shape[0] - min_r, 2)
        too_close = False
        for cmp in cmps:
            if np.linalg.norm(cmp - mp) < min_dist // 2:
                too_close = True
        if too_close:
            continue
        circle = draw.circle_perimeter(mp[0], mp[1], max_r)
        poly_vert = np.random.choice(len(circle[0]), np.random.randint(3, 6), replace=False)
        polys.append(draw.polygon(circle[0][poly_vert], circle[1][poly_vert], shape=shape))
        cmps.append(mp)

    # add some random rectangles
    rects = []
    while len(rects) < n_rect:
        mp = np.random.randint(min_r, shape[0] - min_r, 2)
        _len = np.random.randint(min_r // 2, max_r, (2,))
        too_close = False
        for cmp in cmps:
            if np.linalg.norm(cmp - mp) < min_dist:
                too_close = True
        if too_close:
            continue
        start = (mp[0] - _len[0], mp[1] - _len[1])
        rects.append(draw.rectangle(start, extent=(_len[0] * 2, _len[1] * 2), shape=shape))
        cmps.append(mp)


    # draw polys and give them some noise
    for poly in polys:
        color = np.random.rand(3)
        while np.linalg.norm(color - circle_color) < col_diff or np.linalg.norm(
                color - rect_color) < col_diff:
            color = np.random.rand(3)
        img[poly[0], poly[1], :] = color
        img[poly[0], poly[1], :] += np.random.randn(len(poly[1]), 3) / 5  # add noise to the polygons

    # draw circles with some textural frequency
    cols = np.random.choice(np.arange(4, 11, 1).astype(float) / 10, n_circles, replace=False)  # get colors
    for i, circle in enumerate(circles):
        gt[circle[0], circle[1]] = 1 + (i/10)
        ri1, ri2, ri3, ri4, ri5, ri6 = rsign() * ((np.random.rand() * 4) + 7), \
                                       rsign() * ((np.random.rand() * 4) + 7), \
                                       (np.random.rand() + 1) * 8, \
                                       (np.random.rand() + 1) * 8, \
                                       rsign() * ((np.random.rand() * 4) + 7), \
                                       rsign() * ((np.random.rand() * 4) + 7)

        img[circle[0], circle[1], :] = np.array([cols[i], 0.0, 0.0])  # set even color intensity
        # set interference of two freqs in circle color channel
        img[circle[0], circle[1], :] += np.array([1.0, 1.0, 0.0]) * ((np.sin(np.sqrt((x[circle[0], circle[1]] * ri5) ** 2 + ((shape[1] - y[circle[0], circle[1]]) * ri2) ** 2) * ri3 * np.pi / shape[0]))[..., np.newaxis] * 0.15) + 0.2
        img[circle[0], circle[1], :] += np.array([1.0, 1.0, 0.0]) * ((np.sin(np.sqrt(
            (x[circle[0], circle[1]] * ri6) ** 2 + (
                        (shape[1] - y[circle[0], circle[1]]) * ri1) ** 2) * ri4 * np.pi / shape[1]))[
                                                                           ..., np.newaxis] * 0.15) + 0.2

    # draw rectangles with some frequency
    cols = np.random.choice(np.arange(4, 11, 1).astype(float) / 10, n_rect, replace=False)
    for i, rect in enumerate(rects):
        gt[rect[0], rect[1]] = 2+(i/10)
        ri1, ri2, ri3, ri4, ri5, ri6 = rsign() * ((np.random.rand() * 4) + 7), \
                                       rsign() * ((np.random.rand() * 4) + 7), \
                                       (np.random.rand() + 1) * 8, \
                                       (np.random.rand() + 1) * 8, \
                                       rsign() * ((np.random.rand() * 4) + 7), \
                                       rsign() * ((np.random.rand() * 4) + 7)
        img[rect[0], rect[1], :] = np.array([0.0, 0.0, cols[i]])
        img[rect[0], rect[1], :] += np.array([1.0, 1.0, 0.0]) * ((np.sin(
            np.sqrt((x[rect[0], rect[1]] * ri5) ** 2 + ((shape[1] - y[rect[0], rect[1]]) * ri2) ** 2) * ri3 * np.pi /
            shape[0]))[..., np.newaxis] * 0.15) + 0.2
        img[rect[0], rect[1], :] += np.array([1.0, 1.0, 0.0]) * ((np.sin(
            np.sqrt((x[rect[0], rect[1]] * ri1) ** 2 + ((shape[1] - y[rect[0], rect[1]]) * ri6) ** 2) * ri4 * np.pi /
            shape[1]))[..., np.newaxis] * 0.15) + 0.2

    img = np.clip(img, 0, 1)  # clip to valid range
    # get affinities and calc superpixels with mutex watershed
    affinities = get_naive_affinities(gaussian(img, sigma=.2), edge_offsets)
    affinities[:sep_chnl] *= -1
    affinities[:sep_chnl] += +1
    # scale affinities in order to get an oversegmentation
    affinities[:sep_chnl] /= overseg_factor
    affinities[sep_chnl:] *= overseg_factor
    affinities = np.clip(affinities, 0, 1)
    node_labeling = compute_mws_segmentation(affinities, edge_offsets, sep_chnl)
    node_labeling = node_labeling - 1
    nodes = np.unique(node_labeling)
    try:
        assert all(nodes == np.array(range(len(nodes)), dtype=float))
    except:
        Warning("node ids are off")

    # get edges from node labeling and edge features from affinity stats
    edge_feat, rag = get_edge_features_1d(node_labeling, edge_offsets, affinities)
    edges = rag.uvIds()
    # get gt edge weights based on edges and gt image
    gt_edge_weights = calculate_gt_edge_costs(edges, node_labeling.squeeze(), gt.squeeze(), 0.5)
    mc_gt = get_multicut_sln(rag, gt_edge_weights)
    edges = edges.astype(np.compat.long)

    affinities = affinities.astype(np.float32)
    edge_feat = edge_feat.astype(np.float32)
    nodes = nodes.astype(np.float32)
    node_labeling = node_labeling.astype(np.float32)
    gt_edge_weights = gt_edge_weights.astype(np.float32)
    diff_to_gt = np.abs((edge_feat[:, 0] - gt_edge_weights)).sum()

    edges = np.sort(edges, axis=-1)
    edges = edges.T

    return img, gt, mc_gt, edges, edge_feat, diff_to_gt, gt_edge_weights, node_labeling, nodes, affinities


if __name__ == "__main__":
    file = h5py.File(f"data.h5", "w")
    for i in range(20):
        img, gt, mc_gt, edges, edge_feat, diff_to_gt, gt_edge_weights, node_labeling, nodes, affinities\
            = get_pix_data(shape=(512, 512))

        file.create_group(name=f"sample{i}")
        file[f"sample{i}"].create_dataset(name="raw", data=img)  # raw image data
        file[f"sample{i}"].create_dataset(name="gt", data=gt)  # ground truth segmantation
        file[f"sample{i}"].create_dataset(name="mc_gt", data=mc_gt)  # multicut based on ground truth edges from superpixel segmentation
        file[f"sample{i}"].create_dataset(name="edges", data=edges)  # edge ids for region adjacency graph of superpixel segmentation
        file[f"sample{i}"].create_dataset(name="edge_feat", data=edge_feat)  # edge features based on affinities responsible for the edge
        file[f"sample{i}"].create_dataset(name="diff_to_gt", data=diff_to_gt)  # difference between naive sln based on edge features and ground truth
        file[f"sample{i}"].create_dataset(name="gt_edge_weights", data=gt_edge_weights)  # ground truth edge weights of rag
        file[f"sample{i}"].create_dataset(name="node_labeling", data=node_labeling)  # superpixel segmentation
        file[f"sample{i}"].create_dataset(name="nodes", data=nodes)  # node ids in superpixel segmentation
        file[f"sample{i}"].create_dataset(name="affinities", data=affinities)  # naively calculated affinities based on pairwise pixel intensities

    file.close()
