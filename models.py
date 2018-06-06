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
        self.x_dim = 32*32*3
        self.h_dim = 400
        self.z_dim = 60

        self.x_dropout = nn.Dropout(p=0.2)
        self.en0_layer = nn.Linear(self.x_dim, self.h_dim)

        self.en_mean = nn.Linear(self.h_dim, self.z_dim)
        self.en_lvar = nn.Linear(self.h_dim, self.z_dim)
        self.en_mean_bn = nn.BatchNorm1d(self.z_dim, eps=0.001, momentum=0.001, affine=True)
        self.en_lvar_bn = nn.BatchNorm1d(self.z_dim, eps=0.001, momentum=0.001, affine=True)


        self.z_dropout = nn.Dropout(p=0.2)

        self.dez_layer = nn.Linear(self.z_dim, self.h_dim)
        self.de0_layer = nn.Linear(self.h_dim, self.x_dim)

    def encode(self, x):
        x_do = self.x_dropout(x.view(-1, self.x_dim))
        en0 = F.relu(self.en0_layer(x_do))

        mean = self.en_mean(en0)
        lvar = self.en_lvar(en0)
        mean_bn = self.en_mean_bn(mean)
        lvar_bn = self.en_lvar_bn(lvar)

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
        save_image(sample.view(n_samples, 3, 32, 32), filename)





class M2(object):
    def __init__(self, network_arch):
        self.arch = network_arch
        self.model = M2_ff(self.arch)

        self.adam_b1 = 0.99
        self.adam_b2 = 0.999
        self.lr = 1e-3

        grad_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = optim.Adam(grad_params, lr=self.lr, betas=(self.adam_b1, self.adam_b2))

    def fit(self, x, y):
        output = self.model(x, y)
        self.optimizer.zero_grad()
        output['loss'].backward()
        self.optimizer.step()
        return output

class M2_ff(nn.Module):
    def __init__(self, network_arch):
        super(M2_ff, self).__init__()

        self.yolo = 0
        self.arch = network_arch
        self.x_dim = 32*32*3
        self.h_dim = 400
        self.z_dim = 60
        self.clh_dim = 400
        self.n_labels = 10


        self.x_dropout = nn.Dropout(p=0.2)
        self.en0_layer = nn.Linear(self.x_dim, self.h_dim)

        self.en_mean = nn.Linear(self.h_dim, self.z_dim)
        self.en_lvar = nn.Linear(self.h_dim, self.z_dim)
        self.en_mean_bn = nn.BatchNorm1d(self.z_dim, eps=0.001, momentum=0.001, affine=True)
        self.en_lvar_bn = nn.BatchNorm1d(self.z_dim, eps=0.001, momentum=0.001, affine=True)

        self.cl0_layer = nn.Linear(self.x_dim, self.h_dim)
        self.cl1_layer = nn.Linear(self.h_dim, self.n_labels)

        self.z_dropout = nn.Dropout(p=0.2)

        self.dez_layer = nn.Linear(self.z_dim + self.n_labels, self.h_dim)
        self.de0_layer = nn.Linear(self.h_dim, self.x_dim)

    def encode(self, x):
        x_do = self.x_dropout(x.view(-1, self.x_dim))
        en0 = F.relu(self.en0_layer(x_do))

        mean = self.en_mean(en0)
        lvar = self.en_lvar(en0)
        mean_bn = self.en_mean_bn(mean)
        lvar_bn = self.en_lvar_bn(lvar)

        cl0 = F.relu(self.cl0_layer(x_do))
        y_recon = F.softmax(self.cl1_layer(cl0), dim=-1)

        return mean_bn, lvar_bn, y_recon

    def sample(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            z = eps.mul(std).add_(mean)
            z_do = self.z_dropout(z)
            return z_do
        else:
            return mean

    def decode(self, z, y):
        yz = torch.zeros([z.shape[0], self.n_labels])
        yz.scatter_(1, y.unsqueeze(-1), 1)
        dez = torch.cat([z, yz], -1)
        de0 = F.relu(self.dez_layer(dez))
        x_recon = F.sigmoid(self.de0_layer(de0))
        return x_recon

    def _loss(self, recon_x, x, recon_y, y, mean, lvar, alpha=0.1):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.x_dim), size_average=False)
        KLD = -0.5 * torch.sum(1 + lvar - mean.pow(2) - lvar.exp())
        YCE = F.cross_entropy(recon_y, y)
        return (1-alpha) * (BCE + KLD) + alpha * YCE

    def forward(self, x, y):
        mean, lvar, recon_y = self.encode(x)
        z = self.sample(mean, lvar)
        recon_x = self.decode(z, y)
        loss = self._loss(recon_x, x, recon_y, y, mean, lvar)

        output = {
                'x':       x,
                'recon_x': recon_x,
                'loss':    loss,
                'y':       y,
                'recon_y': recon_y.argmax(dim=-1),
                'y_probs': recon_y,
                }
        return output
    def gen_samples(self, n_samples=20, filename='output/longhaul.png'):
        self.yolo += 1
        all_samples = []
        sample = torch.randn(n_samples, self.z_dim)
        for y in range(0, 10):
            y_samples = torch.LongTensor([y for i in range(n_samples)])
            x_sample = self.decode(sample, y_samples)
            all_samples.append(x_sample)
        samples_tensor = torch.cat(all_samples, dim=-1)
        save_image(samples_tensor.view(-1, 3, 32, 32), filename, nrow=10)

