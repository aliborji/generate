import torch
import torch.nn as nn
from torch import cuda
import torch.optim as optim
from draw_attn import draw, loss_function
import pdb
from torch.autograd.variable import Variable
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import os
from tensorboard import SummaryWriter
from datetime import datetime
from torchvision.utils import make_grid
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt


data_root = '/home/zeng/data/datasets/nature_obj'
check_root = '/home/'

if not os.path.exists(check_root):
    os.mkdir(check_root)

batch_size = 1024
img_size = 64
img_size = 64

dataset = dset.ImageFolder(root=data_root,
                               transform=transforms.Compose([
                                   transforms.Resize(img_size),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor()
                               ]))
# dataset = dset.MNIST('../data', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor()
#                    ]))
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=4)
sb = enumerate(loader, 0)
# train
for epoch in range(10):
    print epoch
    i, (data, _) = sb.next()
    data = data.numpy()
    data = np.transpose(data, (0, 2, 3, 1))
    for ib in range(batch_size):
        plt.imsave('%s/%d_%d.png'%(check_root, epoch, ib), data[ib])