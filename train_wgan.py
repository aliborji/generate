import torch
import torch.nn as nn
import torch.optim as optim
from model import Net_G, Net_D
import pdb
from torch.autograd.variable import Variable
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import os
from tensorboard import SummaryWriter
from datetime import datetime
from torchvision.utils import make_grid


data_root = '/home/zeng/data/datasets/nature_obj'
check_root = '/home/zeng/data/models/wgan'

os.system('rm -rf ./runs2/*')
writer = SummaryWriter('./runs2/'+datetime.now().strftime('%B%d  %H:%M:%S'))

if not os.path.exists(check_root):
    os.mkdir(check_root)

img_size = 256
bsize = 180
nz = 100
ngf = 64
ndf = 64
nc = 3
l = 7

net_g = Net_G(nz, ngf, nc, l)
net_g.cuda()
net_d = Net_D(ndf, nc, l)
net_d.cuda()

dataset = dset.ImageFolder(root=data_root,
                               transform=transforms.Compose([
                                   transforms.Resize(img_size),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

loader = torch.utils.data.DataLoader(dataset, batch_size=bsize,
                                         shuffle=True, num_workers=4)

input = torch.FloatTensor(bsize, 3, img_size, img_size)
noise = torch.FloatTensor(bsize, nz, 1, 1)
one = torch.FloatTensor([1]).cuda()
mone = one * -1

input = input.cuda()
noise = noise.cuda()

# setup optimizer
optimizerD = optim.Adam(net_d.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(net_g.parameters(), lr=0.0002, betas=(0.5, 0.999))

# train
ig = 0
for epoch in range(25):
    dataIter = iter(loader)
    ib = 0
    while ib < len(loader):
        ############################
        # (1) Update D network
        ###########################
        # train the discriminator Diters times
        if ig < 25 or ig % 500 == 0:
            Diters = 10
        else:
            Diters = 5
        id = 0
        while id < Diters and ib < len(loader):
            # clamp parameters to a cube
            for p in net_d.parameters():
                p.data.clamp_(-0.01, 0.01)

            data, _ = dataIter.next()
            ib += 1
            # train with real
            bsize_now = data.size(0)
            data = data.cuda()
            net_d.zero_grad()
            input.resize_as_(data).copy_(data)
            err_d_real = net_d(Variable(input))
            err_d_real = err_d_real.mean()
            err_d_real.backward(one)
            # train with fake
            noise.resize_(bsize_now, nz, 1, 1).normal_(0, 1)
            input_fake = net_g(Variable(noise))
            err_d_fake = net_d(input_fake.detach())
            err_d_fake = err_d_fake.mean()
            err_d_fake.backward(mone)
            err_d = err_d_fake+err_d_real
            optimizerD.step()
            id += 1
        ############################
        # (2) Update G network
        ###########################
        net_g.zero_grad()
        err_g = net_d(input_fake)
        err_g = err_g.mean()
        err_g.backward(one)
        optimizerG.step()
        ig += 1

        ##########################
        # Visualization
        ##########################
        images = make_grid((input_fake.data[:8]+1)/2)
        writer.add_image('images', images, ib)
        writer.add_scalar('error D', err_d.data[0], ib)
        writer.add_scalar('error G', err_g.data[0], ib)

        print 'epoch %d step %d, err_d=%.4f, err_g=%.4f' %(epoch, ib, err_d.data[0], err_g.data[0])
    torch.save(net_g.state_dict(), '%s/NetG-epoch-%d-step-%d.pth'%(check_root, epoch, ib))
    torch.save(net_d.state_dict(), '%s/NetD-epoch-%d-step-%d.pth'%(check_root, epoch, ib))