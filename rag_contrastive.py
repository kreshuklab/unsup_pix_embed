import matplotlib
# matplotlib.use('Agg')
import hydra
import os
from data.spg_dset_no_inp_patch import SpgDset
import torch
from torch.utils.data import DataLoader
from unet3d.model import UNet2D
from utils import soft_update_params, pca_project, get_angles, set_seed_everywhere, get_edge_features_1d
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from losses.RagContrastive_loss import RagContrastive, RagContrastiveWeights
from losses.contrastive_loss import ContrastiveLoss
from tensorboardX import SummaryWriter
from patch_manager import StridedPatches2D, NoPatches2D
from elf.segmentation.features import compute_rag
from data.leptin_dset import LeptinDset
import numpy as np
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
        train_set = SpgDset(self.cfg.gen.data_dir_raw_train,
                            intensity_augmentation=False,
                            noise_augmentation=False)
        val_set = SpgDset(self.cfg.gen.data_dir_raw_val, patched_output=False)
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
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.fe.lr)
        sheduler = ReduceLROnPlateau(optimizer,
                                     patience=7,
                                     threshold=1e-4,
                                     min_lr=1e-8,
                                     factor=0.1)
        # criterion = RagContrastiveWeights(delta_var=0.1, delta_dist=0.4)
        criterion = ContrastiveLoss(delta_var=0.5, delta_dist=1.5)
        acc_loss = 0
        valit = 0
        iteration = 0
        best_loss = np.inf

        while iteration <= wu_cfg.n_iterations:
            for it, (raw, gt, sp_seg, affinities, indices) in enumerate(train_loader):
                patch_idx = indices.squeeze(0)[0] % np.prod(pm.n_patch_per_dim)
                raw, gt, sp_seg, affinities = raw.to(device), gt.to(device), sp_seg.to(device).squeeze(1), affinities.to(device)
                loss_embeds = model(raw.unsqueeze(2)).squeeze(2)


                # all_p = pm.get_patch(torch.cat([raw.squeeze(0), gt.squeeze(0), sp_seg.squeeze(0), embeddings.squeeze(0), affinities.squeeze(0)], 0).to(device), patch_idx)
                # raw, gt, sp_seg, loss_embeds, affinities = all_p[0][None].detach(), all_p[1][None].detach(), all_p[2][None].detach(), all_p[3:3+self.cfg.fe.backbone.out_channels][None], all_p[3+self.cfg.fe.backbone.out_channels:][None].detach()
                # relabel to consecutive ints starting at 0
                # unique = torch.unique(sp_seg)
                # mask = sp_seg == unique[:, None, None]
                # sp_seg = (mask * (torch.arange(len(unique), device=sp_seg.device)[:, None, None] + 1)).sum(0) - 1
                # unique = torch.unique(gt)
                # mask = gt == unique[:, None, None]
                # gt = (mask * (torch.arange(len(unique), device=gt.device)[:, None, None] + 1)).sum(0) - 1

                # edge_feat, edges = get_edge_features_1d(sp_seg.squeeze().cpu().numpy(), [[0, -1], [-1, 0], [-5, 0], [0, -5]], affinities.squeeze().cpu().numpy())
                # edges = torch.from_numpy(edges.astype(np.long)).to(device).T
                # edge_weights = torch.from_numpy(edge_feat.astype(np.float)).to(device)[:, 0][None]
                edges, edge_weights = None, None

                # put embeddings on unit sphere so we can use cosine distance
                # loss_embeds = loss_embeds / torch.norm(loss_embeds, dim=1, keepdim=True)

                loss = criterion(loss_embeds, sp_seg[:, None].long(), edges, edge_weights, chunks=int(sp_seg.max().item()//self.cfg.gen.train_chunk_size))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(loss.item())
                writer.add_scalar("fe_train/lr", optimizer.param_groups[0]['lr'], iteration)
                writer.add_scalar("fe_train/loss", loss.item(), iteration)
                if (iteration) % 100 == 0:
                    with torch.set_grad_enabled(False):
                        for it, (raw, gt, sp_seg, affinities, indices) in enumerate(val_loader):
                            inp, sp_seg, affinities = raw.to(device), sp_seg.to(device).squeeze(1), affinities.to(device)

                            # unique = torch.unique(sp_seg)
                            # mask = sp_seg.squeeze(0) == unique[:, None, None]
                            # sp_seg = (mask * (torch.arange(len(unique), device=sp_seg.device)[:, None, None] + 1)).sum(0) - 1
                            # unique = torch.unique(gt)
                            # mask = gt.squeeze(0) == unique[:, None, None]
                            # gt = (mask * (torch.arange(len(unique), device=gt.device)[:, None, None] + 1)).sum(0) - 1

                            # edge_feat, edges = get_edge_features_1d(sp_seg.squeeze().cpu().numpy(), [[0, -1], [-1, 0], [-5, 0], [0, -5]],
                            #                                         affinities.squeeze().cpu().numpy())
                            # edges = torch.from_numpy(edges.astype(np.long)).to(device).T
                            # edge_weights = torch.from_numpy(edge_feat.astype(np.float)).to(device)[:, 0][None]
                            edges, edge_weights = None, None

                            embeddings = model(inp.unsqueeze(2)).squeeze(2)
                            # embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)
                            ls = criterion(embeddings, sp_seg[:, None].long(), edges, edge_weights, chunks=int(sp_seg.max().item()//self.cfg.gen.val_chunk_size)).item()
                            # ls = 0
                            acc_loss += ls
                            writer.add_scalar("fe_val/loss", ls, valit)
                            valit += 1
                    acc_loss = acc_loss / len(val_loader)
                    if acc_loss < best_loss:
                        torch.save(model.state_dict(), os.path.join(self.save_dir, "best_val_model.pth"))
                        best_loss = acc_loss
                    sheduler.step(acc_loss)
                    acc_loss = 0
                    fig, (a1, a2, a3) = plt.subplots(1, 3, sharex='col', sharey='row',
                                                 gridspec_kw={'hspace': 0, 'wspace': 0})
                    a1.imshow(inp[0].cpu().permute(1, 2, 0).squeeze())
                    a1.set_title('raw')
                    a2.imshow(cm.prism(sp_seg[0].cpu().squeeze() / sp_seg[0].cpu().squeeze().max()))
                    a2.set_title('sp')
                    # a3.imshow(pca_project(get_angles(embeddings)[0].detach().cpu()))
                    a3.imshow(pca_project(embeddings[0].detach().cpu()))
                    a3.set_title('embed')
                    plt.show()
                    # writer.add_figure("examples", fig, iteration//100)
                iteration += 1
                if iteration > wu_cfg.n_iterations:
                    torch.save(model.state_dict(), os.path.join(self.save_dir, "last_model.pth"))
                    break
        return


@hydra.main(config_path="/g/kreshuk/hilt/projects/unsup_pix_embed/conf")
def main(cfg):
    tr = Trainer(cfg)
    tr.train()

if __name__ == '__main__':
    main()
