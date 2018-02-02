import torch
import torch.nn as nn
import torch.optim as optim
from model import Decoder, Encoder
import pdb
from torch.autograd.variable import Variable
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import os
import numpy as np
from myfunc import loss_function
from tensorboard import SummaryWriter
from datetime import datetime
from torchvision.utils import make_grid
import gc
import matplotlib.pyplot as plt

check_root = '/home/zeng/data/generate/vae_64'

if not os.path.exists(check_root):
    os.mkdir(check_root)

img_size = 64
bsize = 512
nz = 100
ngf = 64
ndf = 64
nc = 3
l = 5

decoder = Decoder(nz, ngf, nc, l)
decoder.cuda()
decoder.load_state_dict(torch.load('/home/zeng/data/models/vae_64/decoder-epoch-24-step-997.pth'))

noise = torch.FloatTensor(bsize, nz, 1, 1)

noise = noise.cuda()

# train
for epoch in range(20):
    noise.normal_(0, 1)
    output = decoder((Variable(noise)))
    output = output.data.cpu().numpy()
    output = np.transpose(output, (0, 2, 3, 1))
    for ib in range(bsize):
        plt.imsave('%s/%d_%d.png'%(check_root, epoch, ib), output[ib])