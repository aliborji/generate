import torch.nn as nn
from torch.autograd.variable import Variable
import torch
from torch.nn import functional as F
import pdb

img_size = 28
input_size = img_size ** 2
enc_hidden_size = 100
dec_hidden_size = 100
nz = 100


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
    def __init__(self, seq_len):
        super(draw, self).__init__()
        # writer -> encoder_mu
        self.enc_mu = nn.GRUCell(2 * input_size + dec_hidden_size, enc_hidden_size)
        # writer -> encoder_logsigma
        self.enc_logvar = nn.GRUCell(2 * input_size + dec_hidden_size, enc_hidden_size)
        # hidden_mu->mu
        self.mu_fc = nn.Linear(enc_hidden_size, nz)
        # hidden_logvar->logvar
        self.logvar_fc = nn.Linear(enc_hidden_size, nz)

        self.dec_rnn = nn.GRUCell(nz, dec_hidden_size)
        self.write = nn.Linear(dec_hidden_size, input_size)

    def read(self, x, x_hat):
        return torch.cat((x, x_hat), 1)

    def encoder_RNN(self, r, h_mu_prev, h_logvar_prev, h_dec_prev, seq_id):
        enc_input = torch.cat((r, h_dec_prev), 1)  # skip connection from decoder

        h_mu = self.enc_mu(enc_input, h_mu_prev)
        mu = F.relu(self.mu_fc(h_mu))
        h_logvar = self.enc_logvar(enc_input, h_logvar_prev)
        logvar = F.relu(self.logvar_fc(h_logvar))

        # print("encoder done")
        # print("------------")

        return mu, h_mu, logvar, h_logvar

    def decoder_network(self, z, h_dec_prev, c):
        h_dec = self.dec_rnn(z, h_dec_prev)
        c = c + self.write(h_dec)
        # print("decoder done")
        # print("------------")

        return c, h_dec

    def forward(self, x_in, T):
        # advance by T timesteps
        bsize = x_in.size(0)
        x = x_in.view(bsize, -1)  # flatten
        c = Variable(torch.zeros(bsize, input_size)).cuda()
        h_mu = Variable(torch.zeros(bsize, enc_hidden_size)).cuda()
        h_logvar = Variable(torch.zeros(bsize, enc_hidden_size)).cuda()
        mu = Variable(torch.zeros(bsize, nz)).cuda()
        logvar = Variable(torch.zeros(bsize, nz)).cuda()
        h_dec = Variable(torch.zeros(bsize, dec_hidden_size)).cuda()
        noise = torch.FloatTensor(bsize, nz).cuda()

        mu_t = []
        logvar_t = []

        for seq in range(T):
            x_hat = x - F.sigmoid(c)
            r = self.read(x, x_hat)  # cat operation
            mu, h_mu, logvar, h_logvar = self.encoder_RNN(r, h_mu, h_logvar, h_dec, seq)
            ## reparamize
            std = logvar.mul(0.5).exp_()
            noise.normal_(0, 1)
            c, h_dec = self.decoder_network(Variable(noise).mul(std).add_(mu), h_dec, c)
            mu_t.append(mu)
            logvar_t.append(logvar)

        # print("FORWARD PASS DONE")
        # print("=================")

        return F.sigmoid(c), mu_t, logvar_t