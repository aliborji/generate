from model import Net_G, Net_D
from torch.autograd.variable import Variable
import torch.utils.data
import os
import numpy as np
import matplotlib.pyplot as plt

check_root = './cartoon/wgan'

if not os.path.exists(check_root):
    os.mkdir(check_root)

img_size = 128
bsize = 512
nz = 100
ngf = 64
ndf = 64
nc = 3
l = 5

net_g = Net_G(nz, ngf, nc, l)
net_g.cuda()
# net_g.load_state_dict(torch.load('/home/zeng/data/models/wgan_64/NetG-epoch-48-step-499.pth'))
net_g.load_state_dict(torch.load('/home/zeng/data/models/wgan_cartoon_128/NetG-epoch-24-step-1705.pth'))
# net_g.load_state_dict(torch.load('/home/zeng/data/models/dcgan_64/NetG-epoch-24-step-997.pth'))

noise = torch.FloatTensor(bsize, nz, 1, 1)
noise = noise.cuda()

# train
for epoch in range(25):
    print epoch
    noise.normal_(0, 1)
    input_fake = net_g(Variable(noise))
    input_fake = input_fake.data.cpu().numpy()
    input_fake = (input_fake+1)/2
    input_fake = np.transpose(input_fake, (0, 2, 3, 1))
    for ib in range(bsize):
        plt.imsave('%s/%d_%d.png'%(check_root, epoch, ib), input_fake[ib])
