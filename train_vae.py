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
from myfunc import loss_function
from tensorboard import SummaryWriter
from datetime import datetime
from torchvision.utils import make_grid
import gc


data_root = '/home/crow/data/datasets/nature_obj'
check_root = '/home/crow/data/models/vae_64'

os.system('rm -rf ./runs2/*')
writer = SummaryWriter('./runs2/'+datetime.now().strftime('%B%d  %H:%M:%S'))

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

encoder = Encoder(nz, ndf, nc, l)
encoder.cuda()


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

# train
for epoch in range(25):
    for i, (data, _) in enumerate(loader, 0):
        bsize_now = data.size(0)
        data = data.cuda()
        input.resize_as_(data).copy_(data)
        mu, logvar = encoder(Variable(input))

        # re-parameterize
        std = logvar.mul(0.5).exp_()
        noise.resize_(bsize_now, nz).normal_(0, 1)

        output = decoder((Variable(noise).mul(std).add_(mu)).view(bsize_now, nz, 1, 1))
        loss = loss_function(output, Variable(input), mu, logvar, bsize, img_size)

        encoder.zero_grad()
        decoder.zero_grad()
        loss.backward()
        optimizer_de.step()
        optimizer_en.step()
        print 'epoch %d step %d, err_d=%.4f' % (epoch, i, loss.data[0])

        # if i % 100 == 0:
        # ##########################
        # # Visualization
        # ##########################
        images = make_grid(output.data[:8])
        writer.add_image('output', images, i)
        images = make_grid(input[:8])
        writer.add_image('images', images, i)
        writer.add_scalar('error', loss.data[0], i)
        del mu, logvar, std, output, loss
        gc.collect()
    torch.save(decoder.state_dict(), '%s/decoder-epoch-%d-step-%d.pth'%(check_root, epoch, i))
    torch.save(encoder.state_dict(), '%s/encoder-epoch-%d-step-%d.pth'%(check_root, epoch, i))