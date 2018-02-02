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


data_root = '/home/zeng/data/datasets/cartoon'
check_root = '/home/zeng/data/models/draw_cartoon'

os.system('rm -rf ./runs/*')
writer = SummaryWriter('./runs/'+datetime.now().strftime('%B%d  %H:%M:%S'))

if not os.path.exists(check_root):
    os.mkdir(check_root)

batch_size = 300
seq_len = 20
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


model = draw(seq_len)
model.cuda()
# setup optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))

# train
for epoch in range(25):
    for i, (data, _) in enumerate(loader, 0):
        input = Variable(data).cuda()
        recon_batch, mu_t, logvar_t = model(input, seq_len)
        # loss = Variable(torch.FloatTensor(1).fill_(0).cuda())
        cuda.synchronize()  # elsewise synchronize error
        loss = loss_function(recon_batch, input, mu_t, logvar_t, seq_len, input.size(0), img_size)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            ##########################
            # Visualization
            ##########################
            images = make_grid(recon_batch.data[:8])
            writer.add_image('output', images, i)
            images = make_grid(data[:8])
            writer.add_image('images', images, i)
        writer.add_scalar('error', loss.data[0], i)

        print 'epoch %d step %d, err_d=%.4f' %(epoch, i, loss.data[0])
    torch.save(model.state_dict(), '%s/draw-epoch-%d-step-%d.pth'%(check_root, epoch, i))