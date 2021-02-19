import matplotlib
# matplotlib.use('Agg')
import hydra
import os
from data.spg_dset import SpgDset
import torch
from torch.utils.data import DataLoader
from unet3d.model import UNet2D
from utils import pca_project, get_angles, set_seed_everywhere, get_valid_edges
import matplotlib.pyplot as plt
from transforms import RndAugmentationTfs, add_sp_gauss_noise
from losses.AugmentedAffinityContrastive_loss import AugmentedAffinityContrastive
from tensorboardX import SummaryWriter


class Trainer():
    def __init__(self, cfg):
        self.cfg = cfg
        seeds = torch.randint(0, 2 ** 32, torch.Size([4]))
        set_seed_everywhere(seeds[0])
        self.save_dir = os.path.join(self.cfg.gen.base_dir, 'results/unsup_cl_affinity', self.cfg.gen.target_dir, str(seeds[0].item()))
        self.log_dir = os.path.join(self.save_dir, 'logs')

    def train(self):
        writer = SummaryWriter(logdir=self.log_dir)
        writer.add_text("conf", self.cfg.pretty())
        device = "cuda:0"
        wu_cfg = self.cfg.fe.trainer
        model = UNet2D(self.cfg.fe.n_raw_channels, self.cfg.fe.n_embedding_features, final_sigmoid=False, num_levels=5)
        model.cuda(device)
        dset = SpgDset(self.cfg.gen.data_dir, wu_cfg.patch_manager, wu_cfg.patch_stride, wu_cfg.patch_shape, wu_cfg.reorder_sp)
        dloader = DataLoader(dset, batch_size=wu_cfg.batch_size, shuffle=True, pin_memory=True,
                             num_workers=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.fe.lr)
        tfs = RndAugmentationTfs(wu_cfg.patch_shape)
        criterion = AugmentedAffinityContrastive(delta_var=0.1, delta_dist=0.3)
        acc_loss = 0
        iteration = 0

        while iteration <= wu_cfg.n_iterations:
            for it, (raw, gt, sp_seg, indices) in enumerate(dloader):
                raw, gt, sp_seg = raw.to(device), gt.to(device), sp_seg.to(device)
                # this is still not the correct mask calculation as the affinity offsets go in no tf offset direction
                mask = torch.from_numpy(get_valid_edges([len(criterion.offs)] + list(raw.shape[-2:]), criterion.offs)).to(device)[None]
                # _, _, _, _, affs = dset.get_graphs(indices, sp_seg, device)
                spat_tf, int_tf = tfs.sample(1, 1)
                _, _int_tf = tfs.sample(1, 1)
                inp = add_sp_gauss_noise(_int_tf(raw), 0.2, 0.1, 0.3)
                embeddings = model(inp.unsqueeze(2)).squeeze(2)

                paired = spat_tf(torch.cat((mask, raw, embeddings), -3))
                embeddings_0, mask = paired[..., inp.shape[1]+len(criterion.offs):, :, :], paired[..., :len(criterion.offs), :, :].detach()
                # do intensity transform for spatial transformed input
                aug_inp = int_tf(paired[..., len(criterion.offs):inp.shape[1]+len(criterion.offs), :, :]).detach()
                # get prediction of the augmented input
                embeddings_1 = model(add_sp_gauss_noise(aug_inp, 0.2, 0.1, 0.3).unsqueeze(2)).squeeze(2)

                # put embeddings on unit sphere so we can use cosine distance
                embeddings_0 = embeddings_0 / (torch.norm(embeddings_0, dim=1, keepdim=True) + 1e-6)
                embeddings_1 = embeddings_1 / (torch.norm(embeddings_1, dim=1, keepdim=True) + 1e-6)

                loss = criterion(embeddings_0, embeddings_1, aug_inp, mask)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                acc_loss += loss.item()

                print(loss.item())
                writer.add_scalar("fe_warm_start/loss", loss.item(), iteration)
                writer.add_scalar("fe_warm_start/lr", optimizer.param_groups[0]['lr'], iteration)
                if (iteration) % 50 == 0:
                    acc_loss = 0
                    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2)
                    a1.imshow(aug_inp[0].cpu().permute(1, 2, 0).squeeze())
                    a1.set_title('tf_raw')
                    a3.imshow(pca_project(get_angles(embeddings_0).squeeze(0).detach().cpu()))
                    a3.set_title('tf_embed')
                    a4.imshow(pca_project(get_angles(embeddings_1).squeeze(0).detach().cpu()))
                    a4.set_title('embed')
                    a2.imshow(raw[0].cpu().permute(1, 2, 0).squeeze())
                    a2.set_title('raw')
                    plt.show()
                    # writer.add_figure("examples", fig, iteration//100)
                iteration += 1
                if iteration > wu_cfg.n_iterations:
                    break
        return


@hydra.main(config_path="/g/kreshuk/hilt/projects/unsup_pix_embed/conf")
def main(cfg):
    tr = Trainer(cfg)
    tr.train()

if __name__ == '__main__':
    main()
