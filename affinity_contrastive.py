import matplotlib
# matplotlib.use('Agg')
import os
import numpy as np
from data.spg_dset import SpgDset
from data.leptin_dset import LeptinDset
import torch
from torch.utils.data import DataLoader
from unet3d.model import UNet2D
from utils import pca_project, get_angles, set_seed_everywhere, get_contour_from_2d_binary
import matplotlib.pyplot as plt
from transforms import RndAugmentationTfs, add_sp_gauss_noise
from losses.AffinityContrastive_loss import AffinityContrastive
from tensorboardX import SummaryWriter
from yaml_conv_parser import YamlConf
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pt_gaussfilter import GaussianSmoothing
import torch.nn.functional as F


class Trainer():
    def __init__(self, cfg):
        self.cfg = cfg
        seeds = torch.randint(0, 2 ** 32, torch.Size([4]))
        set_seed_everywhere(seeds[0])
        self.save_dir = os.path.join(self.cfg.gen.base_dir, 'results/unsup_cl_affinity', self.cfg.gen.target_dir, str(seeds[0].item()))
        self.log_dir = os.path.join(self.save_dir, 'logs')

    def train(self):
        writer = SummaryWriter(logdir=self.log_dir)
        device = "cuda:0"
        wu_cfg = self.cfg.fe.trainer
        model = UNet2D(**self.cfg.fe.backbone)
        model.cuda(device)
        train_set = SpgDset(self.cfg.gen.data_dir_raw_train, reorder_sp=False)
        val_set = SpgDset(self.cfg.gen.data_dir_raw_val, reorder_sp=False)
        train_loader = DataLoader(train_set, batch_size=wu_cfg.batch_size, shuffle=True, pin_memory=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=wu_cfg.batch_size, shuffle=True, pin_memory=True, num_workers=0)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.fe.lr)
        criterion = AffinityContrastive(delta_var=0.1, delta_dist=0.3)
        sheduler = ReduceLROnPlateau(optimizer,
                                     patience=5,
                                     threshold=1e-4,
                                     min_lr=1e-5,
                                     factor=0.1)
        valit = 0
        iteration = 0
        best_loss = np.inf

        while iteration <= wu_cfg.n_iterations:
            for it, (raw, gt, sp_seg, affinities, offs, indices) in enumerate(train_loader):
                raw, gt, sp_seg, affinities, offs = raw.to(device), gt.to(device), sp_seg.to(device), affinities.to(device), offs[0].to(device)

                input = torch.cat([raw, affinities], dim=1)

                embeddings = model(input.unsqueeze(2)).squeeze(2)

                embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)

                loss = criterion(embeddings, affinities, offs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                lr = optimizer.param_groups[0]['lr']
                print(f"step {it}; lr({lr}); loss({loss.item()})")
                writer.add_scalar("fe_warm_start/loss", loss.item(), iteration)
                writer.add_scalar("fe_warm_start/lr", lr, iteration)
                if (iteration) % 100 == 0:
                    acc_loss = 0
                    with torch.set_grad_enabled(False):
                        for val_it, (raw, gt, sp_seg, affinities, offs, indices) in enumerate(val_loader):
                            raw, gt, sp_seg, affinities, offs = raw.to(device), gt.to(device), sp_seg.to(
                                device), affinities.to(device), offs[0].to(device)

                            input = torch.cat([raw, affinities], dim=1)

                            embeddings = model(input.unsqueeze(2)).squeeze(2)

                            embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)

                            loss = criterion(embeddings, affinities, offs)
                            acc_loss += loss
                            writer.add_scalar("fe_val/loss", loss, valit)
                            valit += 1
                    acc_loss = acc_loss / len(val_loader)
                    if acc_loss < best_loss:
                        torch.save(model.state_dict(), os.path.join(self.save_dir, "best_val_model.pth"))
                        best_loss = acc_loss
                    sheduler.step(acc_loss)
                    fig, (a1, a2) = plt.subplots(1, 2, sharex='col', sharey='row',
                                                 gridspec_kw={'hspace': 0, 'wspace': 0})
                    a1.imshow(raw[0].cpu().permute(1, 2, 0).squeeze())
                    a1.set_title('raw')
                    a2.imshow(pca_project(embeddings[0].detach().cpu()))
                    a2.set_title('embed')
                    plt.show()
                    # writer.add_figure("examples", fig, iteration // 50)
                iteration += 1
                if iteration > wu_cfg.n_iterations:
                    break
        return


def main():
    tr = Trainer(YamlConf("conf").cfg)
    tr.train()

if __name__ == '__main__':
    main()
