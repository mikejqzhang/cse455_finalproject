import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image

class StackedModel(object):
    def __init__(self, network_arch):
        self.arch = network_arch
        self.model = M1M2(self.arch)

        self.adam_b1 = 0.99
        self.adam_b2 = 0.999
        self.lr = 1e-3

        grad_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = optim.Adam(grad_params, lr=self.lr, betas=(self.adam_b1, self.adam_b2))

    def fit(self, x):
        output = self.model(x)
        self.optimizer.zero_grad()
        output['loss'].backward()
        self.optimizer.step()
        return output

class M1M2(nn.Module):
    def __init__(self, network_arch):
        super(M1M2, self).__init__()

        self.arch = network_arch
        self.m1_x_dim = 784
        self.m1_h_dim = 600
        self.m1_z_dim = 50

        self.m1_x_dropout = nn.Dropout(p=0.2)
        self.m1_en0_layer = nn.Linear(self.x_dim, self.h_dim)

        self.m1_en_mean = nn.Linear(self.h_dim, self.z_dim)
        self.m1_en_lvar = nn.Linear(self.h_dim, self.z_dim)
        self.m1_en_mean_bn = nn.BatchNorm1d(self.z_dim, eps=0.001, momentum=0.001, affine=True)
        self.m1_en_lvar_bn = nn.BatchNorm1d(self.z_dim, eps=0.001, momentum=0.001, affine=True)


        self.z_dropout = nn.Dropout(p=0.2)

        self.m1_dez_layer = nn.Linear(self.z_dim, self.h_dim)
        self.m1_de0_layer = nn.Linear(self.h_dim, self.x_dim)

        self.m2_h_dim = 600
        self.m2_z_dim = 50

        self.m2_x_dropout = nn.Dropout(p=0.2)
        self.m2_en0_layer = nn.Linear(self.x_dim, self.h_dim)

        self.m2_en_mean = nn.Linear(self.h_dim, self.z_dim)
        self.m2_en_lvar = nn.Linear(self.h_dim, self.z_dim)
        self.m2_en_mean_bn = nn.BatchNorm1d(self.z_dim, eps=0.001, momentum=0.001, affine=True)
        self.m2_en_lvar_bn = nn.BatchNorm1d(self.z_dim, eps=0.001, momentum=0.001, affine=True)

        self.m2_dez_layer = nn.Linear(self.z_dim, self.h_dim)
        self.m2_de0_layer = nn.Linear(self.h_dim, self.x_dim)

    def encode_m1(self, x):
        x_do = self.m1_x_dropout(x.view(-1, self.x_dim))
        en0 = F.relu(self.m1_en0_layer(x_do))

        mean = self.m1_en_mean(en0)
        lvar = self.m1_en_lvar(en0)
        mean_bn = self.m1_en_mean_bn(mean)
        lvar_bn = self.m1_en_lvar_bn(lvar)

        return mean_bn, lvar_bn

    def encode_m2(self, x):
        x_do = self.m2_x_dropout(x.view(-1, self.x_dim))
        en0 = F.relu(self.m2_en0_layer(x_do))

        mean = self.m2_en_mean(en0)
        lvar = self.m2_en_lvar(en0)
        mean_bn = self.m2_en_mean_bn(mean)
        lvar_bn = self.m2_en_lvar_bn(lvar)

        return mean_bn, lvar_bn


    def sample(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            z = eps.mul(std).add_(mean)
            z_do = self.z_dropout(z)
            return z_do
        else:
            return mean

    def decode(self, z):
        dez = F.relu(self.dez_layer(z))
        de0 = F.sigmoid(self.de0_layer(dez))
        return de0

    def _loss(self, recon_x, x, mean, lvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.x_dim), size_average=False)
        KLD = -0.5 * torch.sum(1 + lvar - mean.pow(2) - lvar.exp())
        return BCE + KLD

    def forward(self, x):
        mean, lvar = self.encode(x)
        z = self.sample(mean, lvar)
        recon_x = self.decode(z)
        loss = self._loss(recon_x, x, mean, lvar)

        output = {
                'input_x': x,
                'recon_x': recon_x,
                'loss':    loss
                }
        return output

    def gen_samples(self, n_samples=64, filename='output/samples.png'):
        sample = torch.randn(n_samples, self.z_dim)
        sample = self.decode(sample)
        save_image(sample.view(n_samples, 1, 28, 28), filename)
