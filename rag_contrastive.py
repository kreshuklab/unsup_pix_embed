import matplotlib
# matplotlib.use('Agg')
import hydra
import os
from data.spg_dset import SpgDset
import torch
from torch.utils.data import DataLoader
from unet3d.model import UNet2D
from utils import soft_update_params, pca_project, get_angles, set_seed_everywhere, get_edge_features_1d, get_contour_from_2d_binary
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from losses.RagContrastive_loss import RagContrastive, RagContrastiveWeights
from losses.contrastive_loss import ContrastiveLoss
from losses.contrastive_loss_with_edgeweights import ContrastiveWeights
from tensorboardX import SummaryWriter
from patch_manager import StridedPatches2D, NoPatches2D
from elf.segmentation.features import compute_rag
from data.leptin_dset import LeptinDset
import numpy as np
from yaml_conv_parser import YamlConf
import matplotlib.cm as cm
from pt_gaussfilter import GaussianSmoothing
import torch.nn.functional as F


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
        train_set = SpgDset(self.cfg.gen.data_dir_raw_train, reorder_sp=False)
        val_set = SpgDset(self.cfg.gen.data_dir_raw_val, reorder_sp=False)
        # pm = StridedPatches2D(wu_cfg.patch_stride, wu_cfg.patch_shape, train_set.image_shape)
        pm = NoPatches2D()
        train_set.length = len(train_set.graph_file_names) * np.prod(pm.n_patch_per_dim)
        train_set.n_patch_per_dim = pm.n_patch_per_dim
        val_set.length = len(val_set.graph_file_names)
        # dset = LeptinDset(self.cfg.gen.data_dir_raw, self.cfg.gen.data_dir_affs, wu_cfg.patch_manager, wu_cfg.patch_stride, wu_cfg.patch_shape, wu_cfg.reorder_sp)
        train_loader = DataLoader(train_set, batch_size=wu_cfg.batch_size, shuffle=True, pin_memory=True,
                             num_workers=0)
        val_loader = DataLoader(val_set, batch_size=wu_cfg.batch_size, shuffle=True, pin_memory=True,
                             num_workers=0)
        gauss_kernel = GaussianSmoothing(1, 5, 3, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.fe.lr)
        sheduler = ReduceLROnPlateau(optimizer,
                                     patience=20,
                                     threshold=1e-4,
                                     min_lr=1e-5,
                                     factor=0.1)
        criterion = RagContrastiveWeights(delta_var=0.1, delta_dist=0.4)
        acc_loss = 0
        valit = 0
        iteration = 0
        best_loss = np.inf

        while iteration <= wu_cfg.n_iterations:
            for it, (raw, gt, sp_seg, affinities, offs, indices) in enumerate(train_loader):
                raw, gt, sp_seg, affinities = raw.to(device), gt.to(device), sp_seg.to(device), affinities.to(device)

                # edge_img = F.pad(get_contour_from_2d_binary(sp_seg), (2, 2, 2, 2), mode='constant')
                # edge_img = gauss_kernel(edge_img.float())
                # input = torch.cat([raw, edge_img], dim=1)

                offs = offs.numpy().tolist()
                loss_embeds = model(raw[:, :, None]).squeeze(2)

                edge_feat, edges = tuple(zip(*[get_edge_features_1d(seg.squeeze().cpu().numpy(), os, affs.squeeze().cpu().numpy()) for seg, os, affs in zip(sp_seg, offs, affinities)]))
                edges = [torch.from_numpy(e.astype(np.long)).to(device).T for e in edges]
                edge_weights = [torch.from_numpy(ew.astype(np.float32)).to(device)[:, 0][None] for ew in edge_feat]

                # put embeddings on unit sphere so we can use cosine distance
                loss_embeds = loss_embeds / (torch.norm(loss_embeds, dim=1, keepdim=True) + 1e-9)

                loss = criterion(loss_embeds, sp_seg.long(), edges, edge_weights,
                                 chunks=int(sp_seg.max().item()//self.cfg.gen.train_chunk_size),
                                 sigm_factor=self.cfg.gen.sigm_factor, pull_factor=self.cfg.gen.pull_factor)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(loss.item())
                writer.add_scalar("fe_train/lr", optimizer.param_groups[0]['lr'], iteration)
                writer.add_scalar("fe_train/loss", loss.item(), iteration)
                if (iteration) % 100 == 0:
                    with torch.set_grad_enabled(False):
                        for it, (raw, gt, sp_seg, affinities, offs, indices) in enumerate(val_loader):
                            raw, gt, sp_seg, affinities = raw.to(device), gt.to(device), sp_seg.to(device), affinities.to(device)

                            offs = offs.numpy().tolist()
                            embeddings = model(raw[:, :, None]).squeeze(2)

                            # relabel to consecutive ints starting at 0
                            edge_feat, edges = tuple(zip(
                                *[get_edge_features_1d(seg.squeeze().cpu().numpy(), os, affs.squeeze().cpu().numpy())
                                  for seg, os, affs in zip(sp_seg, offs, affinities)]))
                            edges = [torch.from_numpy(e.astype(np.long)).to(device).T for e in edges]
                            edge_weights = [torch.from_numpy(ew.astype(np.float32)).to(device)[:, 0][None] for ew in edge_feat]

                            # put embeddings on unit sphere so we can use cosine distance
                            embeddings = embeddings / (torch.norm(embeddings, dim=1, keepdim=True) + 1e-9)

                            ls = criterion(embeddings, sp_seg.long(), edges, edge_weights,
                                           chunks=int(sp_seg.max().item()//self.cfg.gen.train_chunk_size),
                                           sigm_factor=self.cfg.gen.sigm_factor, pull_factor=self.cfg.gen.pull_factor)
                            # ls = 0
                            acc_loss += ls
                            writer.add_scalar("fe_val/loss", ls, valit)
                            valit += 1
                    acc_loss = acc_loss / len(val_loader)
                    if acc_loss < best_loss:
                        print(self.save_dir)
                        torch.save(model.state_dict(), os.path.join(self.save_dir, "best_val_model.pth"))
                        best_loss = acc_loss
                    sheduler.step(acc_loss)
                    acc_loss = 0
                    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})
                    a1.imshow(raw[0].cpu().permute(1, 2, 0)[..., 0].squeeze())
                    a1.set_title('raw')
                    a2.imshow(cm.prism(sp_seg[0, 0].cpu().squeeze() / sp_seg[0, 0].cpu().squeeze().max()))
                    a2.set_title('sp')
                    a3.imshow(pca_project(get_angles(embeddings)[0].detach().cpu()))
                    a3.set_title('angle_embed')
                    a4.imshow(pca_project(embeddings[0].detach().cpu()))
                    a4.set_title('embed')
                    # plt.show()
                    writer.add_figure("examples", fig, iteration//100)
                iteration += 1
                print(iteration)
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
