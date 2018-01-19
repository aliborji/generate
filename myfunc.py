import torch
from torch.nn import functional as F
from torchvision.utils import make_grid
import pdb


def loss_function(recon_x, x, mu, logvar, bsize, img_size):
    BCE = F.binary_cross_entropy(recon_x.view(bsize, -1), x.view(bsize, -1))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= bsize * img_size**2

    return BCE + KLD


def make_image_grid(img, mean, std):
    img = make_grid(img)
    for i in range(3):
        img[i] *= std[i]
        img[i] += mean[i]
    return img


def make_label_grid(label):
    label = make_grid(label.expand(-1, 3, -1, -1))
    return label