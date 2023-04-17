import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from utils import *
from seq_vae import SeqVae, Prior

import matplotlib.pyplot as plt

dx = 2
dh = 256

"load vanderpol data"
data = torch.load('vanderpol.pt')

"extract reference data"
x, y = data['0']
dy_ref = y.shape[2]

prior = Prior(dx)
vae = SeqVae(dx, dy_ref, dh)
vae_tr = vae_training(vae, prior, 5, (x.float(), y.float()))

i=0




