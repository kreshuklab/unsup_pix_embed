import matplotlib
matplotlib.use('Agg')
import hydra
import math
import os
from data.spg_dset import SpgDset
import torch
from torch.utils.data import DataLoader
from unet3d.model import UNet2D
from utils import soft_update_params, pca_project, get_angles, set_seed_everywhere
import matplotlib.pyplot as plt
from transforms import RndAugmentationTfs, add_sp_gauss_noise
from torch.optim.lr_scheduler import ReduceLROnPlateau
from losses.ProtInfoNce_loss import EntrInfoNCE
from tensorboardX import SummaryWriter
import matplotlib.cm as cm


class Trainer():
    def __init__(self, cfg):
        self.cfg = cfg
        seeds = torch.randint(0, 2 ** 32, torch.Size([4]))
        set_seed_everywhere(seeds[0])
        self.save_dir = os.path.join(self.cfg.gen.base_dir, 'results/unsup_cl', self.cfg.gen.target_dir, str(seeds[0].item()))
        self.log_dir = os.path.join(self.save_dir, 'logs')

    def train(self):
        writer = SummaryWriter(logdir=self.log_dir)
        writer.add_text("conf", self.cfg.pretty())
        device = "cuda:0"
        wu_cfg = self.cfg.fe.trainer
        model = UNet2D(self.cfg.fe.n_raw_channels, self.cfg.fe.n_embedding_features, final_sigmoid=False, num_levels=5)
        momentum_model = UNet2D(self.cfg.fe.n_raw_channels, self.cfg.fe.n_embedding_features, final_sigmoid=False, num_levels=5)
        if wu_cfg.identical_initialization:
            soft_update_params(model, momentum_model, 1)
        momentum_model.cuda(device)
        for param in momentum_model.parameters():
            param.requires_grad = False
        model.cuda(device)
        dset = SpgDset(self.cfg.gen.data_dir, wu_cfg.patch_manager, wu_cfg.patch_stride, wu_cfg.patch_shape, wu_cfg.reorder_sp)
        dloader = DataLoader(dset, batch_size=wu_cfg.batch_size, shuffle=True, pin_memory=True,
                             num_workers=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.fe.lr)
        sheduler = ReduceLROnPlateau(optimizer,
                                     patience=100,
                                     threshold=1e-3,
                                     min_lr=1e-6,
                                     factor=0.1)
        criterion = EntrInfoNCE(alpha=self.cfg.fe.alpha, beta=self.cfg.fe.beta, lbd=self.cfg.fe.lbd,
                                tau=self.cfg.fe.tau, gamma=self.cfg.fe.gamma, num_neg=self.cfg.fe.num_neg,
                                subs_size=self.cfg.fe.subs_size)
        tfs = RndAugmentationTfs(wu_cfg.patch_shape)
        acc_loss = 0
        iteration = 0
        k_step = math.ceil((wu_cfg.n_iterations - wu_cfg.n_k_stop_it) / (wu_cfg.k_start - wu_cfg.k_stop))
        k = wu_cfg.k_start
        psi_step = (wu_cfg.psi_start - wu_cfg.psi_stop) / (wu_cfg.n_iterations - wu_cfg.n_k_stop_it)
        psi = wu_cfg.psi_start

        while iteration <= wu_cfg.n_iterations:
            for it, (raw, gt, sp_seg, indices) in enumerate(dloader):
                inp, gt, sp_seg = raw.to(device), gt.to(device), sp_seg.to(device)
                mask = torch.ones((inp.shape[0], 1,) + inp.shape[2:], device=device).float()
                # get transforms
                spat_tf, int_tf = tfs.sample(1, 1)
                _, _int_tf = tfs.sample(1, 1)
                # add noise to intensity tf of input for momentum network
                mom_inp = add_sp_gauss_noise(_int_tf(inp), 0.2, 0.1, 0.3)
                # get momentum prediction
                embeddings_mom = momentum_model(mom_inp.unsqueeze(2)).squeeze(2)
                # do the same spatial tf for input, mask and momentum prediction
                paired = spat_tf(torch.cat((mask, inp, embeddings_mom), -3))
                embeddings_mom, mask = paired[..., inp.shape[1]+1:, :, :], paired[..., 0, :, :][:, None]
                # do intensity transform for spatial transformed input
                aug_inp = int_tf(paired[..., 1:inp.shape[1]+1, :, :])
                # and add some noise
                aug_inp = add_sp_gauss_noise(aug_inp, 0.2, 0.1, 0.3)
                # get prediction of the augmented input
                embeddings = model(aug_inp.unsqueeze(2)).squeeze(2)

                # put embeddings on unit sphere so we can use cosine distance
                embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)
                embeddings_mom = embeddings_mom + (mask == 0)  # set the void of the image to the 1-vector
                embeddings_mom = embeddings_mom / torch.norm(embeddings_mom, dim=1, keepdim=True)

                loss = criterion(embeddings.squeeze(0),
                                 embeddings_mom.squeeze(0),
                                 k,
                                 mask.squeeze(0),
                                 whiten=wu_cfg.whitened_embeddings,
                                 warmup=iteration < wu_cfg.n_warmup_it,
                                 psi=psi)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                acc_loss += loss.item()

                print(loss.item())
                writer.add_scalar("fe_warm_start/loss", loss.item(), iteration)
                writer.add_scalar("fe_warm_start/lr", optimizer.param_groups[0]['lr'], iteration)
                if (iteration) % 50 == 0:
                    sheduler.step(acc_loss / 10)
                    acc_loss = 0
                    fig, (a1, a2, a3, a4) = plt.subplots(1, 4, sharex='col', sharey='row',
                                                         gridspec_kw={'hspace': 0, 'wspace': 0})
                    a1.imshow(inp[0].cpu().permute(1, 2, 0).squeeze())
                    a1.set_title('raw')
                    a2.imshow(aug_inp[0].cpu().permute(1, 2, 0))
                    a2.set_title('augment')
                    a3.imshow(pca_project(get_angles(embeddings).squeeze(0).detach().cpu()))
                    a3.set_title('embed')
                    a4.imshow(pca_project(get_angles(embeddings_mom).squeeze(0).detach().cpu()))
                    a4.set_title('mom_embed')
                    writer.add_figure("examples", fig, iteration//100)
                iteration += 1
                psi = max(psi-psi_step, wu_cfg.psi_stop)
                if iteration % k_step == 0:
                    k = max(k-1, wu_cfg.k_stop)

                if iteration > wu_cfg.n_iterations:
                    break
                if iteration % wu_cfg.momentum == 0:
                    soft_update_params(model, momentum_model, wu_cfg.momentum_tau)
        return


@hydra.main(config_path="/g/kreshuk/hilt/projects/unsup_pix_embed/conf")
def main(cfg):
    tr = Trainer(cfg)
    tr.train()

if __name__ == '__main__':
    main()
