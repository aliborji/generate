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
data_root = '/home/zeng/data/datasets/nature_obj'
check_root = '/home/zeng/data/models/pixelCNN'

os.system('rm -rf ./runs/*')
writer = SummaryWriter('./runs/'+datetime.now().strftime('%B%d  %H:%M:%S'))

if not os.path.exists(check_root):
    os.mkdir(check_root)

# dataset = dset.ImageFolder(root=data_root,
#                                transform=transforms.Compose([
#                                    transforms.Resize(img_size),
#                                    transforms.CenterCrop(img_size),
#                                    transforms.ToTensor()
#                                ]))

dataset = dset.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ]))

loader = torch.utils.data.DataLoader(dataset, batch_size=bsize,
                                         shuffle=True, num_workers=4)

pcnn = PixelCNN()
pcnn.cuda()
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(pcnn.parameters(), lr=0.0002, betas=(0.5, 0.999))
# train
for epoch in range(10):
    for i, (data, _) in enumerate(loader, 0):
        bsize_now = data.size(0)
        data = data.cuda()
        input = Variable(data)
        output = pcnn(input)
        loss = criterion(output, Variable(data))
        pcnn.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            # ##########################
            # # Visualization
            # ##########################
            images = make_label_grid(F.sigmoid(output).data[:8])
            writer.add_image('output', images, i)
            images = make_label_grid(data[:8])
            writer.add_image('images', images, i)
            writer.add_scalar('error', loss.data[0], i)
            print 'epoch %d step %d, err_d=%.4f' %(epoch, i, loss.data[0])

# test
it = 0
for i, (data, _) in enumerate(loader, 0):
    output = data.cuda()
    output[:, :, 14:, :] = 0
    # output = torch.zeros(bsize, 1, 28, 28).cuda()
    print it
    it += 1
    for i in range(14, 28):
        for j in range(28):
            temp = pcnn(Variable(output, volatile=True))
            output[:, :, i, j] = F.sigmoid(temp).data[:, :, i, j]
    # ##########################
    # # Visualization
    # ##########################
    images = make_label_grid(output[:8])
    writer.add_image('output', images, i)

                        # torch.save(decoder.state_dict(), '%s/decoder-epoch-%d-step-%d.pth'%(check_root, epoch, i))
    # torch.save(encoder.state_dict(), '%s/encoder-epoch-%d-step-%d.pth'%(check_root, epoch, i))