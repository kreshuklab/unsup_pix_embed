import matplotlib
# matplotlib.use('Agg')
import hydra
import os
import PIL
from data.spg_dset import SpgDset
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from torchvision import transforms
from torch.utils.data import DataLoader
from unet3d.model import UNet2D
from utils import get_contour_from_2d_binary, pca_project, get_angles, set_seed_everywhere, get_edge_features_1d
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from losses.RagContrastive_loss import RagContrastive, RagContrastiveWeights
from losses.contrastive_loss import ContrastiveLoss
from losses.contrastive_loss_with_edgeweights import ContrastiveWeights
from losses.RegularizedRagContrastive_loss import RegRagContrastiveWeights
from tensorboardX import SummaryWriter
from patch_manager import StridedPatches2D, NoPatches2D
from elf.segmentation.features import compute_rag
from pt_gaussfilter import GaussianSmoothing
import numpy as np
from yaml_conv_parser import YamlConf
import matplotlib.cm as cm


class Trainer():
    def __init__(self, cfg):
        self.cfg = cfg
        seeds = torch.randint(0, 2 ** 32, torch.Size([4]))
        set_seed_everywhere(seeds[0])
        self.save_dir = os.path.join(self.cfg.gen.base_dir, 'results/unsup_cl_rag', self.cfg.gen.target_dir, str(seeds[0].item()))
        self.log_dir = os.path.join(self.save_dir, 'logs')
        print("embeddings are on sphere")
        print(f"save dir: {self.save_dir}")
        print(f"log dir: {self.log_dir}")

    def train(self):
        writer = SummaryWriter(logdir=self.log_dir)
        device = "cuda:0"
        wu_cfg = self.cfg.fe.trainer
        model = UNet2D(**self.cfg.fe.backbone)
        model.cuda(device)
        # train_set = SpgDset(self.cfg.gen.data_dir_raw_train, patch_manager="no_cross", patch_stride=(10,10), patch_shape=(300,300), reorder_sp=True)
        # val_set = SpgDset(self.cfg.gen.data_dir_raw_val, patch_manager="no_cross", patch_stride=(10,10), patch_shape=(300,300), reorder_sp=True)
        train_set = SpgDset(self.cfg.gen.data_dir_raw_train, reorder_sp=True)
        val_set = SpgDset(self.cfg.gen.data_dir_raw_val, reorder_sp=True)
        # pm = StridedPatches2D(wu_cfg.patch_stride, wu_cfg.patch_shape, train_set.image_shape)
        pm = NoPatches2D()
        train_set.length = len(train_set.graph_file_names) * np.prod(pm.n_patch_per_dim)
        train_set.n_patch_per_dim = pm.n_patch_per_dim
        val_set.length = len(val_set.graph_file_names)
        gauss_kernel = GaussianSmoothing(1, 5, 3, device=device)
        # dset = LeptinDset(self.cfg.gen.data_dir_raw, self.cfg.gen.data_dir_affs, wu_cfg.patch_manager, wu_cfg.patch_stride, wu_cfg.patch_shape, wu_cfg.reorder_sp)
        train_loader = DataLoader(train_set, batch_size=wu_cfg.batch_size, shuffle=True, pin_memory=True,
                             num_workers=0)
        val_loader = DataLoader(val_set, batch_size=wu_cfg.batch_size, shuffle=True, pin_memory=True,
                             num_workers=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.fe.lr)
        sheduler = ReduceLROnPlateau(optimizer,
                                     patience=80,
                                     threshold=1e-4,
                                     min_lr=1e-8,
                                     factor=0.1)
        slcs = [slice(None, self.cfg.fe.embeddings_separator), slice(self.cfg.fe.embeddings_separator, None)]
        criterion = RegRagContrastiveWeights(delta_var=0.1, delta_dist=0.3, slices=slcs)
        acc_loss = 0
        valit = 0
        iteration = 0
        best_loss = np.inf

        while iteration <= wu_cfg.n_iterations:
            for it, (raw, gt, sp_seg, affinities, offs, indices) in enumerate(train_loader):
                raw, gt, sp_seg, affinities = raw.to(device), gt.to(device), sp_seg.to(device), affinities.to(device)
                sp_seg = sp_seg + 1
                edge_img = F.pad(get_contour_from_2d_binary(sp_seg), (2, 2, 2, 2), mode='constant')
                edge_img = gauss_kernel(edge_img.float())
                all = torch.cat([raw, gt, sp_seg, edge_img], dim=1)

                angle = float(torch.randint(-180, 180, (1,)).item())
                rot_all = tvF.rotate(all, angle, PIL.Image.NEAREST)
                rot_raw = rot_all[:, :1]
                rot_gt = rot_all[:, 1:2]
                rot_sp = rot_all[:, 2:3]
                rot_edge_img = rot_all[:, 3:]
                angle = abs(angle / 180)
                valid_sp = []
                for i in range(len(rot_sp)):
                    _valid_sp = torch.unique(rot_sp[i], sorted=True)
                    _valid_sp = _valid_sp[1:] if _valid_sp[0] == 0 else _valid_sp
                    if len(_valid_sp) > self.cfg.gen.sp_samples_per_step:
                        inds = torch.multinomial(torch.ones_like(_valid_sp), self.cfg.gen.sp_samples_per_step, replacement=False)
                        _valid_sp = _valid_sp[inds]
                    valid_sp.append(_valid_sp)

                _rot_sp, _sp_seg = [], []
                for val_sp, rsp, sp in zip(valid_sp, rot_sp, sp_seg):
                    mask = rsp == val_sp[:, None, None]
                    _rot_sp.append((mask * (torch.arange(len(val_sp), device=rsp.device)[:, None, None] + 1)).sum(0))
                    mask = sp == val_sp[:, None, None]
                    _sp_seg.append((mask * (torch.arange(len(val_sp), device=sp.device)[:, None, None] + 1)).sum(0))

                rot_sp = torch.stack(_rot_sp)
                sp_seg = torch.stack(_sp_seg)
                valid_sp = [torch.unique(_rot_sp, sorted=True) for _rot_sp in rot_sp]
                valid_sp = [_valid_sp[1:] if _valid_sp[0] == 0 else _valid_sp for _valid_sp in valid_sp]

                inp = torch.cat([torch.cat([raw, edge_img], 1), torch.cat([rot_raw, rot_edge_img], 1)], 0)
                offs = offs.numpy().tolist()
                edge_feat, edges = tuple(zip(*[get_edge_features_1d(seg.squeeze().cpu().numpy(), os, affs.squeeze().cpu().numpy()) for seg, os, affs in zip(sp_seg, offs, affinities)]))
                edges = [torch.from_numpy(e.astype(np.long)).to(device).T for e in edges]
                edge_weights = [torch.from_numpy(ew.astype(np.float32)).to(device)[:, 0][None] for ew in edge_feat]
                valid_edges_masks = [(_edges[None] == _valid_sp[:, None, None]).sum(0).sum(0) == 2 for _valid_sp, _edges in zip(valid_sp, edges)]
                edges = [_edges[:, valid_edges_mask] - 1 for _edges, valid_edges_mask in zip(edges, valid_edges_masks)]
                edge_weights = [_edge_weights[:, valid_edges_mask] for _edge_weights, valid_edges_mask in zip(edge_weights, valid_edges_masks)]

                # put embeddings on unit sphere so we can use cosine distance
                loss_embeds = model(inp[:, :, None]).squeeze(2)
                loss_embeds = criterion.norm_each_space(loss_embeds, 1)

                loss = criterion(loss_embeds, sp_seg.long(), rot_sp.long(), edges, edge_weights, valid_sp, angle, chunks=int(sp_seg.max().item()//self.cfg.gen.train_chunk_size))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f"step {iteration}: {loss.item()}")
                writer.add_scalar("fe_train/lr", optimizer.param_groups[0]['lr'], iteration)
                writer.add_scalar("fe_train/loss", loss.item(), iteration)
                if (iteration) % 100 == 0:
                    with torch.set_grad_enabled(False):
                        model.eval()
                        print("####start validation####")
                        for it, (raw, gt, sp_seg, affinities, offs, indices) in enumerate(val_loader):
                            raw, gt, sp_seg, affinities = raw.to(device), gt.to(device), sp_seg.to(
                                device), affinities.to(device)
                            sp_seg = sp_seg + 1
                            edge_img = F.pad(get_contour_from_2d_binary(sp_seg), (2, 2, 2, 2), mode='constant')
                            edge_img = gauss_kernel(edge_img.float())
                            all = torch.cat([raw, gt, sp_seg, edge_img], dim=1)

                            angle = float(torch.randint(-180, 180, (1,)).item())
                            rot_all = tvF.rotate(all, angle, PIL.Image.NEAREST)
                            rot_raw = rot_all[:, :1]
                            rot_gt = rot_all[:, 1:2]
                            rot_sp = rot_all[:, 2:3]
                            rot_edge_img = rot_all[:, 3:]
                            angle = abs(angle / 180)
                            valid_sp = [torch.unique(_rot_sp, sorted=True) for _rot_sp in rot_sp]
                            valid_sp = [_valid_sp[1:] if _valid_sp[0] == 0 else _valid_sp for _valid_sp in valid_sp]

                            _rot_sp, _sp_seg = [], []
                            for val_sp, rsp, sp in zip(valid_sp, rot_sp, sp_seg):
                                mask = rsp == val_sp[:, None, None]
                                _rot_sp.append(
                                    (mask * (torch.arange(len(val_sp), device=rsp.device)[:, None, None] + 1)).sum(0))
                                mask = sp == val_sp[:, None, None]
                                _sp_seg.append(
                                    (mask * (torch.arange(len(val_sp), device=sp.device)[:, None, None] + 1)).sum(0))

                            rot_sp = torch.stack(_rot_sp)
                            sp_seg = torch.stack(_sp_seg)
                            valid_sp = [torch.unique(_rot_sp, sorted=True) for _rot_sp in rot_sp]
                            valid_sp = [_valid_sp[1:] if _valid_sp[0] == 0 else _valid_sp for _valid_sp in valid_sp]

                            inp = torch.cat([torch.cat([raw, edge_img], 1), torch.cat([rot_raw, rot_edge_img], 1)], 0)
                            offs = offs.numpy().tolist()
                            edge_feat, edges = tuple(zip(
                                *[get_edge_features_1d(seg.squeeze().cpu().numpy(), os, affs.squeeze().cpu().numpy())
                                  for seg, os, affs in zip(sp_seg, offs, affinities)]))
                            edges = [torch.from_numpy(e.astype(np.long)).to(device).T for e in edges]
                            edge_weights = [torch.from_numpy(ew.astype(np.float32)).to(device)[:, 0][None] for ew in
                                            edge_feat]
                            valid_edges_masks = [(_edges[None] == _valid_sp[:, None, None]).sum(0).sum(0) == 2 for
                                                 _valid_sp, _edges in zip(valid_sp, edges)]
                            edges = [_edges[:, valid_edges_mask] - 1 for _edges, valid_edges_mask in
                                     zip(edges, valid_edges_masks)]
                            edge_weights = [_edge_weights[:, valid_edges_mask] for _edge_weights, valid_edges_mask in
                                            zip(edge_weights, valid_edges_masks)]

                            # put embeddings on unit sphere so we can use cosine distance
                            embeds = model(inp[:, :, None]).squeeze(2)
                            embeds = criterion.norm_each_space(embeds, 1)

                            ls = criterion(embeds, sp_seg.long(), rot_sp.long(), edges, edge_weights, valid_sp,
                                           angle, chunks=int(sp_seg.max().item() // self.cfg.gen.train_chunk_size))

                            acc_loss += ls
                            writer.add_scalar("fe_val/loss", ls, valit)
                            print(f"step {it}: {ls.item()}")
                            valit += 1

                    acc_loss = acc_loss / len(val_loader)
                    if acc_loss < best_loss:
                        print(self.save_dir)
                        torch.save(model.state_dict(), os.path.join(self.save_dir, "best_val_model.pth"))
                        best_loss = acc_loss
                    sheduler.step(acc_loss)
                    acc_loss = 0
                    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})
                    a1.imshow(raw[0].cpu().permute(1, 2, 0).squeeze())
                    a1.set_title('raw')
                    a2.imshow(cm.prism(sp_seg[0].cpu().squeeze() / sp_seg[0].cpu().squeeze().max()))
                    a2.set_title('sp')
                    a3.imshow(pca_project(embeds[0, slcs[0]].detach().cpu()))
                    a3.set_title('embed', y=-0.01)
                    a4.imshow(pca_project(embeds[0, slcs[1]].detach().cpu()))
                    a4.set_title('embed rot', y=-0.01)
                    plt.show()
                    writer.add_figure("examples", fig, iteration//100)
                    # model.train()
                    print("####end validation####")
                iteration += 1
                if iteration > wu_cfg.n_iterations:
                    print(self.save_dir)
                    torch.save(model.state_dict(), os.path.join(self.save_dir, "last_model.pth"))
                    break
        return


def main():
    tr = Trainer(YamlConf("conf").cfg)
    tr.train()

if __name__ == '__main__':
    main()
