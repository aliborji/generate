import torch
import torch.nn as nn
import torch.optim as optim
from pixelcnn import PixelCNN
import pdb
import torch.nn.functional as F
from torch.autograd.variable import Variable
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import os
from myfunc import loss_function, make_image_grid, make_label_grid
from tensorboard import SummaryWriter
from datetime import datetime
import gc
import matplotlib.pyplot as plt

bsize = 64
img_size = 32
data_root = '/home/zeng/data/datasets/nature_obj'
check_root = '/home/zeng/data/models/pixelCNN'

os.system('rm -rf ./runs/*')
writer = SummaryWriter('./runs/'+datetime.now().strftime('%B%d  %H:%M:%S'))

if not os.path.exists(check_root):
    os.mkdir(check_root)

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

loader = torch.utils.data.DataLoader(dataset, batch_size=bsize,
                                         shuffle=True, num_workers=4)

pcnn = PixelCNN()
pcnn.cuda()
criterion = nn.NLLLoss2d()

optimizer = optim.Adam(pcnn.parameters(), lr=0.0002, betas=(0.5, 0.999))
# train
for epoch in range(100):
    for i, (data, _) in enumerate(loader, 0):
        data = data.mean(1, keepdim=True)
        bsize_now, _, h, w = data.size()

        ids = (255*data).long()

        label = torch.FloatTensor(bsize_now, 256, h, w).scatter_(1, ids, torch.ones(ids.size())).cuda()

        input = Variable(data).cuda()
        output = pcnn(input)

        loss = criterion(F.log_softmax(output), Variable(ids[:, 0]).cuda())
        pcnn.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            # ##########################
            # # Visualization
            # ##########################

            _, temp = torch.max(output, 1)
            images = make_label_grid(temp.data.float().unsqueeze(1)[:8]/255)
            writer.add_image('output', images, i)
            images = make_label_grid(data[:8])
            writer.add_image('images', images, i)
            writer.add_scalar('error', loss.data[0], i)
            print 'epoch %d step %d, err_d=%.4f' %(epoch, i, loss.data[0])
        if i % 1000==0:
            output = data.cuda()
            output[:, :, 14:, :] = 0
            for j in range(14, 32):
                for k in range(32):
                    temp = pcnn(Variable(output, volatile=True))
                    _, temp = torch.max(temp, 1)
                    output[:, :, j, k] = temp.data.float().unsqueeze(1)[:,:, j, k]/255
            # ##########################
            # # Visualization
            # ##########################
            images = make_label_grid(output[:8])
            writer.add_image('validation', images, i)

                        # torch.save(decoder.state_dict(), '%s/decoder-epoch-%d-step-%d.pth'%(check_root, epoch, i))
    # torch.save(encoder.state_dict(), '%s/encoder-epoch-%d-step-%d.pth'%(check_root, epoch, i))