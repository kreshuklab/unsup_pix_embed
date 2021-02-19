import matplotlib
matplotlib.use('Agg')
import hydra
import os
from data.spg_dset import SpgDset
import torch
from torch.utils.data import DataLoader
from unet3d.model import UNet2D
from utils import soft_update_params, pca_project, get_angles, set_seed_everywhere
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from losses.RagProtInfoNce_loss import RagInfoNCE
from tensorboardX import SummaryWriter


class Trainer():
    def __init__(self, cfg):
        self.cfg = cfg
        seeds = torch.randint(0, 2 ** 32, torch.Size([4]))
        set_seed_everywhere(seeds[0])
        self.save_dir = os.path.join(self.cfg.gen.base_dir, 'results/unsup_cl_rag', self.cfg.gen.target_dir, str(seeds[0].item()))
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
        sheduler = ReduceLROnPlateau(optimizer,
                                     patience=100,
                                     threshold=1e-3,
                                     min_lr=1e-6,
                                     factor=0.1)
        criterion = RagInfoNCE(tau=self.cfg.fe.tau)
        acc_loss = 0
        iteration = 0

        while iteration <= wu_cfg.n_iterations:
            for it, (raw, gt, sp_seg, indices) in enumerate(dloader):
                inp, gt, sp_seg = raw.to(device), gt.to(device), sp_seg.to(device)
                edges = dloader.dataset.get_graphs(indices, sp_seg, device)[0]

                off = 0
                for i in range(len(edges)):
                    sp_seg[i] += off
                    edges[i] += off
                    off = sp_seg[i].max() + 1
                edges = torch.cat(edges, 1)
                embeddings = model(inp.unsqueeze(2)).squeeze(2)

                # put embeddings on unit sphere so we can use cosine distance
                embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)

                loss = criterion(embeddings, sp_seg, edges)

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
                    fig, (a1, a2) = plt.subplots(1, 2, sharex='col', sharey='row',
                                                         gridspec_kw={'hspace': 0, 'wspace': 0})
                    a1.imshow(inp[0].cpu().permute(1, 2, 0).squeeze())
                    a1.set_title('raw')
                    a2.imshow(pca_project(get_angles(embeddings).squeeze(0).detach().cpu()))
                    a2.set_title('embed')
                    writer.add_figure("examples", fig, iteration//100)
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
