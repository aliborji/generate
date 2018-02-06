import torch.nn as nn
from torch.autograd.variable import Variable
import torch
from torch.nn import functional as F
import pdb


import torch.nn as nn
from torch.autograd.variable import Variable
import torch
from torch.nn import functional as F
import pdb

# img_size = 64
# N = 32
# A = img_size
# B = img_size
# input_size = img_size ** 2
# patch_size = N ** 2
# enc_hidden_size = 200
# dec_hidden_size = 400
# nz = 200
#

def loss_function(recon_x, x, mu, logvar, T, bsize, img_size):
    # after T timesteps, we compare reconstruction with original
    BCE = F.binary_cross_entropy(recon_x.view(bsize, -1), x.view(bsize, -1))
    KLD = 0.0

    for seq in range(T):
        KLD_element = -0.5 * torch.sum(1 + logvar[seq] - mu[seq].pow(2) - logvar[seq].exp())
        # Normalise by same number of elements as in reconstruction
        KLD_element /= bsize * img_size ** 2
        KLD += KLD_element

    return BCE + KLD


class draw(nn.Module):
    img_size = 64
    input_size = img_size ** 2
    enc_hidden_size = 800
    dec_hidden_size = 1600
    nz = 100

    def __init__(self, seq_len):
        super(draw, self).__init__()
        # writer -> encoder_mu
        self.enc_mu = nn.LSTMCell(2 * self.input_size * 3 + self.dec_hidden_size, self.enc_hidden_size)
        # writer -> encoder_logsigma
        self.enc_logvar = nn.LSTMCell(2 * self.input_size * 3 + self.dec_hidden_size, self.enc_hidden_size)
        # hidden_mu->mu
        self.mu_fc = nn.Linear(self.enc_hidden_size, self.nz)
        # hidden_logvar->logvar
        self.logvar_fc = nn.Linear(self.enc_hidden_size, self.nz)

        self.dec_rnn = nn.LSTMCell(self.nz, self.dec_hidden_size)
        self.write = nn.Linear(self.dec_hidden_size, self.input_size*3)

    def read(self, x, x_hat):
        return torch.cat((x, x_hat), 1)

    def encoder_network(self, r, h_mu_prev, c_mu_prev, h_logvar_prev, c_logvar_prev, h_dec_prev):
        enc_input = torch.cat((r, h_dec_prev), 1)  # skip connection from decoder

        h_mu, c_mu = self.enc_mu(enc_input, (h_mu_prev, c_mu_prev))
        mu = F.relu(self.mu_fc(h_mu))
        h_logvar, c_logvar = self.enc_logvar(enc_input, (h_logvar_prev, c_logvar_prev))
        logvar = F.relu(self.logvar_fc(h_logvar))

        # print("encoder done")
        # print("------------")

        return mu, h_mu, c_mu, logvar, h_logvar, c_logvar

    def decoder_network(self, z, c_dec_prev, h_dec_prev, c):
        h_dec, c_dec = self.dec_rnn(z, (h_dec_prev, c_dec_prev))
        sb = self.write(h_dec)
        sb = sb.view(-1, 3, self.img_size, self.img_size)
        c = c+sb
        # print("decoder done")
        # print("------------")

        return c, h_dec, c_dec

    def forward(self, x, T):
        # advance by T timesteps
        bsize = x.size(0)
        c = Variable(torch.zeros(bsize, 3, self.img_size, self.img_size)).cuda()
        h_mu = Variable(torch.zeros(bsize, self.enc_hidden_size)).cuda()
        c_mu = Variable(torch.zeros(bsize, self.enc_hidden_size)).cuda()
        h_logvar = Variable(torch.zeros(bsize, self.enc_hidden_size)).cuda()
        c_logvar = Variable(torch.zeros(bsize, self.enc_hidden_size)).cuda()
        h_dec = Variable(torch.zeros(bsize, self.dec_hidden_size)).cuda()
        noise = torch.zeros(bsize, self.nz).cuda()
        c_dec = Variable(torch.zeros(bsize, self.dec_hidden_size)).cuda()

        mu_t = []
        logvar_t = []

        for seq in range(T):
            sb = F.sigmoid(c)

            x_hat = x - sb

            rd = self.read(x, x_hat)  # cat operation

            mu, h_mu, c_mu, logvar, h_logvar, c_logvar = \
                self.encoder_network(rd.view(bsize, -1), h_mu, c_mu, h_logvar, c_logvar, h_dec)
            ## reparamize
            std = logvar.mul(0.5).exp_()
            noise.normal_(0, 1)
            c, h_dec, c_dec = self.decoder_network(Variable(noise).mul(std).add_(mu), c_dec, h_dec, c)
            mu_t.append(mu)
            logvar_t.append(logvar)
        # print("FORWARD PASS DONE")
        # print("=================")

        return F.sigmoid(c), mu_t, logvar_t