import torch
import torch.nn as nn
import torch.nn.functional as F
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


data_root = '/home/crow/data/datasets/cartoon'
check_root = '/home/crow/data/models/dcgan_cartoon_128'

os.system('rm -rf ./runs2/*')
writer = SummaryWriter('./runs2/'+datetime.now().strftime('%B%d  %H:%M:%S'))

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

criterion = nn.BCELoss()

input = torch.FloatTensor(bsize, 3, img_size, img_size)
noise = torch.FloatTensor(bsize, nz, 1, 1)
fixed_noise = torch.FloatTensor(bsize, nz, 1, 1).normal_(0, 1)
label = torch.FloatTensor(bsize)
real_label = 1
fake_label = 0

criterion.cuda()
input, label = input.cuda(), label.cuda()
noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizerD = optim.Adam(net_d.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(net_g.parameters(), lr=0.0002, betas=(0.5, 0.999))

for epoch in range(25):
    for i, (data, _) in enumerate(loader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        bsize_now = data.size(0)
        data = data.cuda()
        input.resize_as_(data).copy_(data)
        output = net_d(Variable(input))
        output = F.sigmoid(output)

        label.resize_(bsize_now).fill_(1)
        err_d_real = criterion(output, Variable(label))
        net_d.zero_grad()
        err_d_real.backward()

        # train with fake
        noise.resize_(bsize_now, nz, 1, 1).normal_(0, 1)
        input_fake = net_g(Variable(noise))
        output = net_d(input_fake.detach())
        output = F.sigmoid(output)

        label.fill_(0)
        err_d_fake = criterion(output, Variable(label))
        err_d_fake.backward()
        optimizerD.step()
        err_d = err_d_fake + err_d_real

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        output = net_d(input_fake)
        output = F.sigmoid(output)
        label.fill_(1)

        err_g = criterion(output, Variable(label))
        net_g.zero_grad()
        err_g.backward()
        optimizerG.step()
        if i%100 == 0:
            ##########################
            # Visualization
            ##########################
            images = make_grid((input_fake.data[:8]+1)/2)
            writer.add_image('images', images, i)
            writer.add_scalar('error D', err_d.data[0], i)
            writer.add_scalar('error G', err_g.data[0], i)

        print 'epoch %d step %d, err_d=%.4f, err_g=%.4f' %(epoch, i, err_d.data[0], err_g.data[0])
    torch.save(net_g.state_dict(), '%s/NetG-epoch-%d-step-%d.pth'%(check_root, epoch, i))
    torch.save(net_d.state_dict(), '%s/NetD-epoch-%d-step-%d.pth'%(check_root, epoch, i))
