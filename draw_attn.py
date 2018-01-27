import torch.nn as nn
from torch.autograd.variable import Variable
import torch
from torch.nn import functional as F
import pdb

img_size = 32
N = 16
A = img_size
B = img_size
input_size = img_size ** 2
patch_size = N ** 2
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
        self.enc_mu = nn.GRUCell(2 * patch_size *3 + dec_hidden_size, enc_hidden_size)
        # writer -> encoder_logsigma
        self.enc_logvar = nn.GRUCell(2 * patch_size *3 + dec_hidden_size, enc_hidden_size)
        # hidden_mu->mu
        self.mu_fc = nn.Linear(enc_hidden_size, nz)
        # hidden_logvar->logvar
        self.logvar_fc = nn.Linear(enc_hidden_size, nz)

        self.dec_rnn = nn.LSTMCell(nz, dec_hidden_size)
        self.hdec_to_attparam = nn.Linear(dec_hidden_size, 5)
        self.hdec_to_write = nn.Linear(dec_hidden_size, patch_size*3)

    def compute_filterbank_matrices(self, g_x, g_y, delta, var, N, A, B, batch_size):
        i = torch.arange(0, N).cuda()

        gx = g_x.expand(N, A, batch_size).permute(2, 0, 1)
        # gx  = gx.permute(2,0,1)

        i = Variable(i.expand(batch_size, A, N).permute(0, 2, 1))

        dx = delta.expand(N, A, batch_size).permute(2, 0, 1)

        mu_i = gx + i * dx - (N / 2 + 0.5) * dx

        a = torch.arange(0, A).cuda()
        a = Variable(a.expand(batch_size, N, A))

        vx = var.expand(N, A, batch_size).permute(2, 0, 1)

        F_x = torch.exp(-(a - mu_i) * (a - mu_i) / (2.0 * vx))

        n_x = torch.sum(F_x, 2).expand(A, batch_size, N)
        n_x = n_x.permute(1, 2, 0)

        F_x = F_x / n_x

        # now compute F_y

        gy = g_y.expand(N, A, batch_size).permute(2, 0, 1)
        dy = delta.expand(N, B, batch_size).permute(2, 0, 1)

        j = torch.arange(0, N).cuda()
        j = Variable(j.expand(batch_size, B, N)).permute(0, 2, 1)

        mu_j = gy + j * dy - (N / 2 + 0.5) * dy

        b = torch.arange(0, B).cuda()
        b = Variable(b.expand(batch_size, N, A))

        vy = var.expand(N, B, batch_size).permute(2, 0, 1)

        F_y = torch.exp(-(b - mu_j) * (b - mu_j) / (2.0 * vy))

        n_y = torch.sum(F_x, 2).expand(A, batch_size, N)
        n_y = n_y.permute(1, 2, 0)

        F_y = F_y / n_y

        return F_x, F_y

    def get_attn_params(self, h_dec):
        params = self.hdec_to_attparam(h_dec)

        g_x = params[:, 0]
        g_y = params[:, 1]
        logvar = params[:, 2]
        logdelta = params[:, 3]
        loggamma = params[:, 4]

        delta = torch.exp(logdelta)
        gamma = torch.exp(loggamma)
        var = torch.exp(logvar)

        g_x = (A + 1) * (g_x + 1) / 2
        g_y = (B + 1) * (g_y + 1) / 2
        delta = (max(A, B) - 1) / (N - 1) * delta

        bsize = h_dec.size(0)

        F_x, F_y = self.compute_filterbank_matrices(g_x, g_y, delta, var, N, A, B, bsize)

        return F_x, F_y, gamma

    def read(self, x, x_hat, h_dec_prev):
        F_x, F_y, gamma = self.get_attn_params(h_dec_prev)
        gamma = torch.unsqueeze(gamma, 1)
        gamma = torch.unsqueeze(gamma, 2)
        bsize = x.size(0)

        F_x_t = F_x.permute(0, 2, 1)

        tmp_x_r = F_y.bmm(x[:, 0].bmm(F_x_t))
        tmp_x_r = gamma.expand_as(tmp_x_r) * tmp_x_r
        tmp_x_g = F_y.bmm(x[:, 1].bmm(F_x_t))
        tmp_x_g = gamma.expand_as(tmp_x_g) * tmp_x_g
        tmp_x_b = F_y.bmm(x[:, 2].bmm(F_x_t))
        tmp_x_b = gamma.expand_as(tmp_x_b) * tmp_x_b

        tmp_x_hat_r = F_y.bmm(x_hat[:, 0].bmm(F_x_t))
        tmp_x_hat_r = gamma.expand_as(tmp_x_hat_r) * tmp_x_hat_r
        tmp_x_hat_g = F_y.bmm(x_hat[:, 1].bmm(F_x_t))
        tmp_x_hat_g = gamma.expand_as(tmp_x_hat_g) * tmp_x_hat_g
        tmp_x_hat_b = F_y.bmm(x_hat[:, 2].bmm(F_x_t))
        tmp_x_hat_b = gamma.expand_as(tmp_x_hat_b) * tmp_x_hat_b
        # this should have size 2*NxN == 2*patch_size
        return torch.stack((tmp_x_r, tmp_x_g, tmp_x_b, tmp_x_hat_r, tmp_x_hat_g, tmp_x_hat_b), 1)

    def write(self, h_dec):
        F_x, F_y, gamma = self.get_attn_params(h_dec)

        w = self.hdec_to_write(h_dec)
        w = w.view(-1, 3, N, N)

        F_y_t = F_y.permute(0, 2, 1)
        tmp_r = F_y_t.bmm(w[:, 0].bmm(F_x))
        tmp_g = F_y_t.bmm(w[:, 1].bmm(F_x))
        tmp_b = F_y_t.bmm(w[:, 2].bmm(F_x))
        tmp = torch.stack((tmp_r, tmp_g, tmp_b), 1)

        epsilon = 0.0001 * Variable(torch.ones(gamma.size()).cuda())
        g = (gamma + epsilon)
        g = torch.unsqueeze(g, 1)
        g = torch.unsqueeze(g, 2)
        g = torch.unsqueeze(g, 3)

        tmp = (1.0/g).expand_as(tmp)*tmp
        return tmp

    def encoder_RNN(self, rd, h_mu_prev, h_logvar_prev, h_dec_prev):
        enc_input = torch.cat((rd, h_dec_prev), 1)  # skip connection from decoder

        h_mu = self.enc_mu(enc_input, h_mu_prev)
        mu = F.relu(self.mu_fc(h_mu))
        h_logvar = self.enc_logvar(enc_input, h_logvar_prev)
        logvar = F.relu(self.logvar_fc(h_logvar))

        # print("encoder done")
        # print("------------")

        return mu, h_mu, logvar, h_logvar

    def decoder_network(self, z, c_dec_prev, h_dec_prev, c):
        h_dec, c_dec = self.dec_rnn(z, (h_dec_prev, c_dec_prev))
        sb = self.write(h_dec)
        sb = sb.view(-1, 3, img_size, img_size)
        c = c + sb
        # print("decoder done")
        # print("------------")

        return c, h_dec, c_dec

    def forward(self, x, T):
        # advance by T timesteps
        bsize = x.size(0)
        c = Variable(torch.zeros(bsize, 3, img_size, img_size)).cuda()
        h_mu = Variable(torch.zeros(bsize, enc_hidden_size)).cuda()
        h_logvar = Variable(torch.zeros(bsize, enc_hidden_size)).cuda()
        h_dec = Variable(torch.zeros(bsize, dec_hidden_size)).cuda()
        noise = torch.FloatTensor(bsize, nz).cuda()
        c_dec = Variable(torch.zeros(bsize, dec_hidden_size)).cuda()

        mu_t = []
        logvar_t = []

        for seq in range(T):
            sb = F.sigmoid(c)

            x_hat = x - sb

            rd = self.read(x, x_hat, h_dec)  # cat operation

            mu, h_mu, logvar, h_logvar = self.encoder_RNN(rd.view(bsize, -1), h_mu, h_logvar, h_dec)
            ## reparamize
            std = logvar.mul(0.5).exp_()
            noise.normal_(0, 1)
            c, h_dec, c_dec = self.decoder_network(Variable(noise).mul(std).add_(mu), c_dec, h_dec, c)
            mu_t.append(mu)
            logvar_t.append(logvar)
        # print("FORWARD PASS DONE")
        # print("=================")

        return F.sigmoid(c), mu_t, logvar_t