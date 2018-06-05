import sys
import argparse
from tqdm import tqdm

import torch
import torch.utils.data
from torchvision import datasets, transforms
from torchvision.utils import save_image

from models import M1, M2


def get_data(batch_size):
    # kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    kwargs = {}
    # train_loader = torch.utils.data.DataLoader(
    #         datasets.MNIST('data/', train=True, download=True,
    #             transform=transforms.ToTensor()),
    #         batch_size=batch_size, shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(
    #         datasets.MNIST('data/', train=False, transform=transforms.ToTensor()),
    #         batch_size=batch_size, shuffle=True, **kwargs)
    train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('fashiondata/', train=True, download=True,
                transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('fashiondata/', train=False, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader

batch_size = 128
train_loader, test_loader = get_data(batch_size)
network_arch = {}
# M1_model = M1(network_arch)
# M1_model.model.train()
M2_model = M2(network_arch)
M2_model.model.train()

M2_model.model.gen_samples()
for epoch in range(50):
    train_loss = 0
    for i, (x, y) in enumerate(tqdm(train_loader)):
        output = M2_model.fit(x, y)
        train_loss += output['loss'].item()
    print('epoch loss:', train_loss / len(train_loader.dataset))
    M2_model.model.eval()
    M2_model.model.gen_samples()
    M2_model.model.train()
