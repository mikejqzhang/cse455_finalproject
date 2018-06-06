import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image


class M1(object):
    def __init__(self, network_arch):
        self.arch = network_arch
        self.model = M1_ff(self.arch)

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


class M1_ff(nn.Module):
    def __init__(self, network_arch):
        super(M1_ff, self).__init__()

        self.arch = network_arch
        self.x_dim = 784
        self.h_dim = 400
        self.z_dim = 20

        self.x_dropout = nn.Dropout(p=0.2)
        self.z_dropout = nn.Dropout(p=0.2)

        # CNN layers
        self.conv_dim = 8 * 2 * 2

        self.convolve = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )

        self.transpose = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

        # Encoder layers
        self.en0_layer = nn.Linear(self.conv_dim, self.h_dim)
        self.en_mean = nn.Linear(self.h_dim, self.z_dim)
        self.en_lvar = nn.Linear(self.h_dim, self.z_dim)
        self.en_mean_bn = nn.BatchNorm1d(self.z_dim, eps=0.001, momentum=0.001, affine=True)
        self.en_lvar_bn = nn.BatchNorm1d(self.z_dim, eps=0.001, momentum=0.001, affine=True)

        # Decoder layers
        self.dez_layer = nn.Linear(self.z_dim, self.h_dim)
        self.de0_layer = nn.Linear(self.h_dim, self.x_dim)

    def encode(self, x):
        # Use CNN to encode
        x = self.convolve(x)

        # Map CNN to x-dim
        x = x.view(-1, self.conv_dim)

        # Run old model
        x_do = self.x_dropout(x)
        en0 = F.relu(self.en0_layer(x_do))

        mean = self.en_mean(en0)
        lvar = self.en_lvar(en0)
        mean_bn = self.en_mean_bn(mean)
        lvar_bn = self.en_lvar_bn(lvar)

        return mean_bn, lvar_bn

    def sample(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = eps.mul(std).add_(mean)
            z_do = self.z_dropout(z)
            return z_do
        else:
            return mean

    def decode(self, z):
        # Decode
        dez = F.relu(self.dez_layer(z))
        de0 = F.sigmoid(self.de0_layer(dez))

        # TODO: use transpose to ??undo?? the convolutions
        # de0 = self.transpose(de0)

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
            'loss': loss
        }
        return output

    def gen_samples(self, n_samples=64, filename='output/samples.png'):
        sample = torch.randn(n_samples, self.z_dim)
        sample = self.decode(sample)
        save_image(sample.view(n_samples, 1, 28, 28), filename)
