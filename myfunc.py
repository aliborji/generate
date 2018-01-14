import torch
from torch.nn import functional as F

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