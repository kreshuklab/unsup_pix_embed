import torch
from torchvision import transforms

class RndAugmentationTfs():
    """ returns two random sets of image augmentation transforms.
    The first contains n_spat spatial image tfs and the second one n_int intensity based tfs.
    """
    def __init__(self, img_size):
        self.spat_tfs = [
            transforms.RandomAffine(180),
            transforms.RandomResizedCrop(tuple(img_size)),
            transforms.RandomHorizontalFlip(p=1.),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=2, fill=0),
            transforms.RandomRotation(180),
            transforms.RandomVerticalFlip(p=1.)
        ]

        self.int_tfs = [
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=.5),
            transforms.RandomGrayscale(p=0.5),
            transforms.GaussianBlur(kernel_size=5)
        ]
    def sample(self, n_spat, n_int):
        spat_inds = torch.multinomial(torch.ones(len(self.spat_tfs)), n_spat)
        int_inds = torch.multinomial(torch.ones(len(self.int_tfs)), n_int)

        spat = transforms.Compose([self.spat_tfs[idx] for idx in spat_inds])
        int = transforms.Compose([self.int_tfs[idx] for idx in int_inds])

        return spat, int

def whiten(x, subset_size=512):
    """ applies zca whitening to subsets of data matrix x"""
    eps = 1e-10
    ret = []
    nsp = len(x)
    ind = list(range(0, (nsp // subset_size) * subset_size + 1, subset_size))
    ind = [0] if ind == [] else ind
    ind = ind + [ind[-1] + nsp % subset_size] if nsp % subset_size > 0 else ind
    if len(ind) > 2 and ind[-1] - ind[-2] < subset_size // 2:
        ind[-2] -= subset_size // 2
    for i in range(len(ind) - 1):
        _x = x[ind[i]:ind[i+1]]
        zcx = _x - _x.mean()
        cov = torch.mm(zcx, zcx.T)
        u, s, v = torch.svd(cov, some=False)
        w = torch.mm(u, torch.mm(torch.diag(1.0 / torch.sqrt(s + eps)), u.T))
        ret.append(torch.mm(w, _x))
    return torch.cat(ret)


def add_sp_gauss_noise(input, tau, sp_ratio, prob=0.5):
    """ adds std gaussian and s&p noise to the input of random intensity (0-tau)
        input is assumed to be normalized to [0,1].
        The additional noise is added with a proba of prob. Otw no noise is added.
    """
    if torch.multinomial(torch.tensor([prob, 1-prob]), 1) == 1:
        return input
    shape = input.shape
    flattened = input.reshape(-1)
    flattened = flattened + torch.randn_like(flattened) * torch.rand(1).item() * tau  # add gaussian noise
    flattened = flattened - flattened.min()  # renormalize
    flattened = flattened / flattened.max()
    sp = torch.multinomial(torch.ones_like(flattened), int(flattened.shape[0] * sp_ratio))  # get sp indices
    salt, pepper = sp[:sp.shape[0]//2], sp[sp.shape[0]//2:]
    flattened[salt] = 1
    flattened[pepper] = 0

    return flattened.reshape(shape)