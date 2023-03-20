import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from utils import *
from vae import VAE


w, h = 28, 28
dx = 20

# load MNIST dataset
train_dataset = DataLoader(datasets.MNIST(root='./mnist_data/', train=True,
                                          transform=transforms.ToTensor(),
                                          download=True),
                           batch_size=128, shuffle=True)

test_dataset = DataLoader(datasets.MNIST(root='./mnist_data/', train=False,
                                         transform=transforms.ToTensor(),
                                         download=False),
                          batch_size=128, shuffle=True)


vae = VAE(dx, w, h)
vae = vae_training(vae, 5, train_dataset)




