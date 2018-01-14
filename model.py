import torch.nn as nn
import pdb



# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Net_G(nn.Module):
    def __init__(self, nz, ngf, nc, l):
        super(Net_G, self).__init__()
        _layers = []
        _layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(nz, ngf * 2**(l-2), 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 2**(l-2)),
                nn.ReLU(True)
            )
        )
        for i in range(1, l-1):
            _layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(ngf * 2**(l-1-i), ngf * 2**(l-2-i), 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf * 2**(l-2-i)),
                    nn.ReLU(True)
                )
            )
        _layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
            )
        )
        self.layers = nn.ModuleList(_layers)

    def forward(self, input):
        for l in self.layers:
            input = l(input)
        return input


class Net_D(nn.Module):
    def __init__(self, ndf, nc, l):
        super(Net_D, self).__init__()
        _layers = []
        _layers.append(
            nn.Sequential(
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
            )
        )
        for i in range(1, l-1):
            _layers.append(
                nn.Sequential(
                    nn.Conv2d(ndf * 2**(i-1), ndf * 2**i, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ndf * 2**i),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
        _layers.append(
            nn.Sequential(
                nn.Conv2d(ndf * 2**(l-2), 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
        )
        self.layers = nn.ModuleList(_layers)

    def forward(self, input):
        for l in self.layers:
            input = l(input)

        return input.view(-1, 1).squeeze(1)


class Encoder(nn.Module):
    def __init__(self, nz, ndf, nc, l):
        super(Encoder, self).__init__()
        _layers = []
        _layers.append(
            nn.Sequential(
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
            )
        )
        for i in range(1, l-1):
            _layers.append(
                nn.Sequential(
                    nn.Conv2d(ndf * 2**(i-1), ndf * 2**i, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ndf * 2**i),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
        self.fc21 = nn.Linear(16 * ndf * 2**(l-2), nz)
        self.fc22 = nn.Linear(16 * ndf * 2**(l-2), nz)
        self.layers = nn.ModuleList(_layers)

    def forward(self, input):
        for l in self.layers:
            input = l(input)
        bsize = input.size(0)

        return self.fc21(input.view(bsize, -1)), self.fc22(input.view(bsize, -1))


class Decoder(nn.Module):
    def __init__(self, nz, ngf, nc, l):
        super(Decoder, self).__init__()
        _layers = []
        _layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(nz, ngf * 2**(l-2), 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 2**(l-2)),
                nn.ReLU(True)
            )
        )
        for i in range(1, l-1):
            _layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(ngf * 2**(l-1-i), ngf * 2**(l-2-i), 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf * 2**(l-2-i)),
                    nn.ReLU(True)
                )
            )
        _layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Sigmoid()
            )
        )
        self.layers = nn.ModuleList(_layers)

    def forward(self, input):
        for l in self.layers:
            input = l(input)
        return input