import torch
import torch.nn as nn
from torch import cuda
import torch.optim as optim
from draw import draw, loss_function
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

check_root = './generate/draw_noattn'

os.system('rm -rf ./runs/*')
writer = SummaryWriter('./runs/'+datetime.now().strftime('%B%d  %H:%M:%S'))

if not os.path.exists(check_root):
    os.mkdir(check_root)

batch_size = 512 # picture
#batch_size = 300 # cartoon
seq_len = 20
img_size = 64

enc_hidden_size = 800
dec_hidden_size = 1600
nz = 100

model = draw(seq_len)
model.cuda()
# model.load_state_dict(torch.load('/home/zeng/data/models/draw_64/draw-epoch-19-step-1701.pth'))
model.load_state_dict(torch.load('/home/zeng/data/models/draw-epoch-4-step-498.pth'))

# train
for epoch in range(20):
    c = Variable(torch.zeros(batch_size, 3, img_size, img_size)).cuda()
    h_dec = Variable(torch.zeros(batch_size, dec_hidden_size)).cuda()
    noise = torch.zeros(batch_size, nz).cuda()
    c_dec = Variable(torch.zeros(batch_size, dec_hidden_size)).cuda()
    for seq in range(seq_len):
        noise.normal_(0, 1)
        c, h_dec, c_dec = model.decoder_network(Variable(noise), c_dec, h_dec, c)
    output = F.sigmoid(c)
    output = output.data.cpu().numpy()
    output = np.transpose(output, (0, 2, 3, 1))
    for ib in range(batch_size):
        plt.imsave('%s/%d_%d.png'%(check_root, epoch, ib), output[ib])