import matplotlib
# matplotlib.use('Agg')
import hydra
import os
from data.leptin_dset import LeptinValDset
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
import elf.segmentation.features as feats
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
        train_set = SpgDset("/g/kreshuk/hilt/projects/data/leptin_fused_tp1_ch_0/true_val", reorder_sp=True)
        val_set = SpgDset("/g/kreshuk/hilt/projects/data/leptin_fused_tp1_ch_0/train", reorder_sp=True)
        # pm = StridedPatches2D(wu_cfg.patch_stride, wu_cfg.patch_shape, train_set.image_shape)
        train_loader = DataLoader(train_set, batch_size=wu_cfg.batch_size, shuffle=True, pin_memory=True,
                             num_workers=0)
        val_loader = DataLoader(val_set, batch_size=wu_cfg.batch_size, shuffle=True, pin_memory=True,
                             num_workers=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.fe.lr)
        sheduler = ReduceLROnPlateau(optimizer,
                                     patience=40,
                                     threshold=1e-4,
                                     min_lr=1e-5,
                                     factor=0.1)
        criterion = RagContrastiveWeights(delta_var=0.1, delta_dist=0.3)
        acc_loss = 0
        valit = 0
        iteration = 0
        best_loss = np.inf

        while iteration <= wu_cfg.n_iterations:
            for it, (raw, gt, sp_seg, affinities, offs, indices) in enumerate(train_loader):
                raw, gt = raw.to(device), gt.to(device)

                loss_embeds = model(raw[:, :, None]).squeeze(2)
                loss_embeds = loss_embeds / (torch.norm(loss_embeds, dim=1, keepdim=True) + 1e-9)

                edges = [feats.compute_rag(seg.cpu().numpy()).uvIds() for seg in gt]
                edges = [torch.from_numpy(e.astype(np.long)).to(device).T for e in edges]

                loss = criterion(loss_embeds, gt.long(), edges, None, 30)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(loss.item())
                # writer.add_scalar("fe_train/lr", optimizer.param_groups[0]['lr'], iteration)
                # writer.add_scalar("fe_train/loss", loss.item(), iteration)
                # if (iteration) % 100 == 0:
                #
                #     fig, (a1, a2, a3) = plt.subplots(3, 1, sharex='col', sharey='row',
                #                                  gridspec_kw={'hspace': 0, 'wspace': 0})
                #     a1.imshow(raw[0, 0].cpu().squeeze())
                #     a1.set_title('train raw')
                #     a2.imshow(pca_project(loss_embeds[0].detach().cpu()))
                #     a2.set_title('train embed')
                #     a3.imshow(gt[0, 0].cpu().squeeze())
                #     a3.set_title('train gt')
                #     plt.show()
                #
                #     with torch.set_grad_enabled(False):
                #         for it, (raw, gt, sp_seg, affinities, offs, indices) in enumerate(val_loader):
                #             raw = raw.to(device)
                #             embeds = model(raw[:, :, None]).squeeze(2)
                #             embeds = embeds / (torch.norm(embeds, dim=1, keepdim=True) + 1e-9)
                #
                #             print(loss.item())
                #             writer.add_scalar("fe_train/lr", optimizer.param_groups[0]['lr'], iteration)
                #             writer.add_scalar("fe_train/loss", loss.item(), iteration)
                #             fig, (a1, a2) = plt.subplots(2, 1, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})
                #             a1.imshow(raw[0, 0].cpu().squeeze())
                #             a1.set_title('raw')
                #             a2.imshow(pca_project(embeds[0].detach().cpu()))
                #             a2.set_title('embed')
                #             plt.show()
                #             if it > 2:
                #                 break
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
