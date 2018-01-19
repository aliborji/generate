import torch
import torch.nn as nn
import torch.optim as optim
from draw import Decoder, Encoder, read, Write, Atten
import pdb
from torch.autograd.variable import Variable
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import os
from myfunc import loss_function
from tensorboard import SummaryWriter
from datetime import datetime
from torchvision.utils import make_grid
from torch.nn import functional as F


data_root = '/home/zeng/data/datasets/nature_obj'
check_root = '/home/zeng/data/models/draw'

os.system('rm -rf ./runs2/*')
writer = SummaryWriter('./runs2/'+datetime.now().strftime('%B%d  %H:%M:%S'))

if not os.path.exists(check_root):
    os.mkdir(check_root)

bsize = 256
nz = 100
ngf = 64
ndf = 64
nc = 3
l = 7
img_size = 2**(l+1)

decoder = Decoder(nz, ngf, nc, l)
decoder.cuda()

encoder = Encoder(nz, ndf, nc, l)
encoder.cuda()

write = Write()
write.cuda()
atten = Atten(img_size, img_size)
atten.cuda()


dataset = dset.ImageFolder(root=data_root,
                               transform=transforms.Compose([
                                   transforms.Resize(img_size),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor()
                               ]))

loader = torch.utils.data.DataLoader(dataset, batch_size=bsize,
                                         shuffle=True, num_workers=4)

input = torch.FloatTensor(bsize, 3, img_size, img_size)
noise = torch.FloatTensor(bsize, nz)

input = input.cuda()
noise = noise.cuda()

# setup optimizer
optimizer_en = optim.Adam(encoder.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_de = optim.Adam(decoder.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_wr = optim.Adam(write.parameters(), lr=0.0002, betas=(0.5, 0.999))

# train
for epoch in range(25):
    for i, (data, _) in enumerate(loader, 0):
        bsize_now = data.size(0)
        data = data.cuda()
        input.resize_as_(data).copy_(data)

        x = Variable(input)
        noise.resize_(bsize_now, nz).normal_(0, 1)

        mu = torch.FloatTensor(bsize_now, nz).fill_(0)
        logvar = torch.FloatTensor(bsize_now, nz).fill_(0)
        h_dec = torch.FloatTensor(bsize_now, 3, 32, 32).fill_(0)
        c = torch.FloatTensor(bsize_now, 3, img_size, img_size).fill_(0)
        mu = Variable(mu).cuda()
        logvar = Variable(logvar).cuda()
        h_dec = Variable(h_dec).cuda()
        c = Variable(c).cuda()
        loss = 0
        for t in range(10):
            x_hat = x - F.sigmoid(c.detach())
            (Fx, Fy), gamma = atten(h_dec)

            r = read(x, x_hat, Fx, Fy, gamma, img_size, img_size)
            mu, logvar = encoder(r, h_dec, mu, logvar)
            # re-parameterize
            std = logvar.mul(0.5).exp_()
            h_dec = decoder(h_dec, (Variable(noise).mul(std).add_(mu)).view(bsize_now, nz, 1, 1))
            (Fx, Fy), gamma = atten(h_dec)

            c = c + write(h_dec, Fx, Fy, gamma, img_size, img_size)

            loss += (- 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))

        loss /= bsize * img_size**2
        loss += F.binary_cross_entropy(F.sigmoid(c).view(bsize, -1), x.view(bsize, -1))
        # loss = loss_function(F.sigmoid(c), Variable(input), mu, logvar, bsize, img_size)

        encoder.zero_grad()
        decoder.zero_grad()
        write.zero_grad()
        loss.backward()
        optimizer_de.step()
        optimizer_en.step()
        optimizer_wr.step()
        if i % 1 == 0:
            ##########################
            # Visualization
            ##########################
            images = make_grid(F.sigmoid(c).data[:8])
            writer.add_image('output', images, i)
            images = make_grid(input[:8])
            writer.add_image('images', images, i)
            writer.add_scalar('error', loss.data[0], i)

        print 'epoch %d step %d, err_d=%.4f' %(epoch, i, loss.data[0])
    torch.save(decoder.state_dict(), '%s/decoder-epoch-%d-step-%d.pth'%(check_root, epoch, i))
    torch.save(encoder.state_dict(), '%s/encoder-epoch-%d-step-%d.pth'%(check_root, epoch, i))