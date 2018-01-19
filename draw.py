import torch.nn as nn
from torch.autograd.variable import Variable
import torch
import pdb

def unit_prefix(x, n=1):
    for i in range(n): x = x.unsqueeze(0)
    return x


def align(x, y, start_dim=0):
    xd, yd = x.dim(), y.dim()
    if xd > yd: y = unit_prefix(y, xd - yd)
    elif yd > xd: x = unit_prefix(x, yd - xd)

    xs, ys = list(x.size()), list(y.size())
    nd = len(ys)
    for i in range(start_dim, nd):
        td = nd-i-1
        if   ys[td]==1: ys[td] = xs[td]
        elif xs[td]==1: xs[td] = ys[td]
    return x.expand(*xs), y.expand(*ys)

K = 4
N = 2**(K+1)
class Atten(nn.Module):
    def __init__(self, A, B):
        super(Atten, self).__init__()
        self.A = A
        self.B = B
        self.main = nn.Linear(N*N*3, 5)

    # correct
    def compute_mu(self,g,rng,delta):
        rng_t,delta_t = align(rng,delta)
        tmp = (rng_t - N / 2 - 0.5) * delta_t
        tmp_t,g_t = align(tmp,g)
        mu = tmp_t + g_t
        return mu

    def filterbank_matrices(self,a,mu_x,sigma2,epsilon=1e-9):
        t_a,t_mu_x = align(a,mu_x)
        temp = t_a - t_mu_x
        temp,t_sigma = align(temp,sigma2)
        temp = temp / (t_sigma * 2)
        F = torch.exp(-torch.pow(temp,2))
        F = F / (F.sum(2,True).expand_as(F) + epsilon)
        return F

    # correct
    def filterbank(self,gx,gy,sigma2,delta):
        rng = Variable(torch.arange(0,N).view(1,-1))
        rng =  rng.cuda()
        mu_x = self.compute_mu(gx,rng,delta)
        mu_y = self.compute_mu(gy,rng,delta)

        a = Variable(torch.arange(0,self.A).view(1,1,-1))
        a = a.cuda()
        b = Variable(torch.arange(0,self.B).view(1,1,-1))
        b = b.cuda()

        mu_x = mu_x.view(-1,N,1)
        mu_y = mu_y.view(-1,N,1)
        sigma2 = sigma2.view(-1,1,1)

        Fx = self.filterbank_matrices(a,mu_x,sigma2)
        Fy = self.filterbank_matrices(b,mu_y,sigma2)

        return Fx,Fy

    def attn_window(self, params):
        gx_, gy_, log_sigma_2, log_delta, log_gamma = params.split(1, 1)  # 21

        gx = (self.A + 1) / 2 * (gx_ + 1)  # 22
        gy = (self.B + 1) / 2 * (gy_ + 1)  # 23
        delta = (max(self.A, self.B) - 1) / (N - 1) * torch.exp(log_delta)  # 24
        sigma2 = torch.exp(log_sigma_2)
        gamma = torch.exp(log_gamma)

        return self.filterbank(gx, gy, sigma2, delta), gamma

    def forward(self, h_dec):
        params = self.main(h_dec.view(-1, N * N * 3))
        (Fx, Fy), gamma = self.attn_window(params)
        return (Fx, Fy), gamma


def read(x, x_hat, Fx, Fy, gamma, A, B):
    def filter_img(img, Fx, Fy, gamma, A, B):
        Fxt = Fx.transpose(2, 1)
        r, g, b = img.split(1, 1)
        r = r.view(-1, B, A)
        glimpse_r = Fy.bmm(r.bmm(Fxt))
        glimpse_r = glimpse_r.view(-1, N * N)
        glimpse_r = glimpse_r*gamma.view(-1, 1).expand_as(glimpse_r)
        g = g.view(-1, B, A)
        glimpse_g = Fy.bmm(g.bmm(Fxt))
        glimpse_g = glimpse_g.view(-1, N * N)
        glimpse_g = glimpse_g*gamma.view(-1, 1).expand_as(glimpse_g)
        b = b.view(-1, B, A)
        glimpse_b = Fy.bmm(b.bmm(Fxt))
        glimpse_b = glimpse_b.view(-1, N * N)
        glimpse_b = glimpse_b*gamma.view(-1, 1).expand_as(glimpse_b)

        return torch.stack((glimpse_r, glimpse_g, glimpse_b), 1).view(-1, 3, N, N)

    x = filter_img(x, Fx, Fy, gamma, A, B)
    x_hat = filter_img(x_hat, Fx, Fy, gamma, A, B)
    return torch.cat((x, x_hat), 1)


class Write(nn.Module):
    def __init__(self):
        super(Write, self).__init__()
        self.main = nn.Conv2d(3, 3, 1)

    def forward(self, h_dec, Fx, Fy, gamma, A, B):
        w = self.main(h_dec)

        def filter_img(img, Fx, Fy, gamma, A, B):
            Fyt = Fy.transpose(2,1)
            r, g, b = img.split(1, 1)
            r = r.view(-1, N, N)
            glimpse_r = Fyt.bmm(r.bmm(Fx))
            glimpse_r = glimpse_r.view(-1, A * B)
            glimpse_r = glimpse_r / gamma.view(-1, 1).expand_as(glimpse_r)
            g = g.view(-1, N, N)
            glimpse_g = Fyt.bmm(g.bmm(Fx))
            glimpse_g = glimpse_g.view(-1, A * B)
            glimpse_g = glimpse_g / gamma.view(-1, 1).expand_as(glimpse_g)
            b = b.view(-1, N, N)
            glimpse_b = Fyt.bmm(b.bmm(Fx))
            glimpse_b = glimpse_b.view(-1, A * B)
            glimpse_b = glimpse_b / gamma.view(-1, 1).expand_as(glimpse_b)

            return torch.stack((glimpse_r, glimpse_g, glimpse_b), 1).view(-1, 3, A, B)
        return filter_img(w, Fx, Fy, gamma, A, B)


# class Read(nn.Module):
#     def __init__(self):
#         super(Read, self).__init__()
#
#     def forward(self, x, hat_x):
#         return torch.cat((x, hat_x), 1)


# class Write(nn.Module):
#     def __init__(self):
#         super(Write, self).__init__()
#         self.main = nn.Conv2d(3, 3, 1)
#
#     def forward(self, h_dec):
#         return self.main(h_dec)


class Encoder(nn.Module):
    def __init__(self, nz, ndf, nc, l):
        super(Encoder, self).__init__()
        _layers = []
        _layers.append(
            nn.Sequential(
                nn.Conv2d(nc*3, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
            )
        )
        for i in range(1, K-1):
            _layers.append(
                nn.Sequential(
                    nn.Conv2d(ndf * 2**(i-1), ndf * 2**i, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ndf * 2**i),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
        self.fc21 = nn.Linear(16 * ndf * 2**(K-2) + nz*2, nz)  # mu
        self.fc22 = nn.Linear(16 * ndf * 2**(K-2) + nz*2, nz)  # logvar
        self.layers = nn.ModuleList(_layers)
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #         m.weight.data.normal_(0.0, 0.02)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.normal_(1.0, 0.02)
        #         m.bias.data.fill_(0)

    def forward(self, r, h_dec, mu, logvar):
        input = torch.cat((r, h_dec), 1)
        for l in self.layers:
            input = l(input)
        bsize = input.size(0)
        input = torch.cat((input.view(bsize, -1), mu, logvar), 1)
        return self.fc21(input), self.fc22(input)


class Decoder(nn.Module):
    def __init__(self, nz, ngf, nc, l):
        super(Decoder, self).__init__()
        _layers = []
        _layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(nz, ngf * 2**(K-2), 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 2**(K-2)),
                nn.ReLU(True)
            )
        )
        for i in range(1, K-1):
            _layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(ngf * 2**(K-1-i), ngf * 2**(K-2-i), 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf * 2**(K-2-i)),
                    nn.ReLU(True)
                )
            )
        _layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)
            )
        )
        _layers.append(
            nn.Conv2d(nc*2, nc, 1)
        )
        self.layers = nn.ModuleList(_layers)
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #         m.weight.data.normal_(0.0, 0.02)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.normal_(1.0, 0.02)
        #         m.bias.data.fill_(0)

    def forward(self, h_dec, z):
        for l in self.layers[:-1]:
            z = l(z)
        z = torch.cat((z, h_dec), 1)
        return self.layers[-1](z)