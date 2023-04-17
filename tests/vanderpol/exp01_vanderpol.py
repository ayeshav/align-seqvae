import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from utils import *
from seq_vae import SeqVae, Prior

import matplotlib.pyplot as plt

dx = 2
dh = 256

n_train = 300

"load vanderpol data"
data = torch.load('vanderpol.pt')

"extract reference data"
x, y = data['0']
x_train, y_train = x[:, :n_train, :], y[:, :n_train, :]

data_ref = DataLoader(DataSetTs(x_train.float(), y_train.float()), batch_size=80)

dy_ref = y.shape[2]

prior = Prior(dx)
vae = SeqVae(dx, dy_ref, dh)
vae_tr, prior_tr = vae_training(vae, prior, 100, data_ref)

i=0




