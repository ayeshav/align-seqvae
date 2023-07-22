import os
import numpy as np
from utils import *

from encoders import *
from decoders import *
from priors import *
from seq_vae import *

from align import *
from train import *


if torch.has_mps:
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device='cpu'

print(device)

"load vanderpol data"
data = torch.load('../data/noisy_vanderpol_gaussian.pt')

"adding some global variables that we can later port to a config file"

dx = 2
dh = 64

train_ratio = 0.8

K_vae = 1
K_align = 10
N_epochs_vae = 2_000
batch_size = 100

noise = 'Normal'


def get_training_data(y, y_other, normalize=True):

    K, _, dy = y.shape

    "let's get the reference data"
    N_train = int(np.ceil(train_ratio * K))
    y_train = y[:N_train]
    y_other_train = y_other[:N_train]

    if normalize:
        mu_train = torch.mean(y_train.reshape(-1, dy), 0, keepdim=True)
        sigma_train = torch.std(y_train.reshape(-1, dy), 0, keepdim=True)
        y_train = (y_train - mu_train) / sigma_train

        mu_train = torch.mean(y_other_train.reshape(-1, y_other.shape[-1]), 0, keepdim=True)
        sigma_train = torch.std(y_other_train.reshape(-1, y_other.shape[-1]), 0, keepdim=True)
        y_other_train = (y_other_train - mu_train) / sigma_train

    train_dataloader = SeqDataLoader((y_train,y_other_train), batch_size=batch_size)
    return train_dataloader


def main():

    dy, dy_other = data[0]['y'].shape[-1], data[2]['y'].shape[-1]

    "let's define our vae"
    encoder = Encoder(dy, dx, dh)
    prior = Prior(dx)
    decoder = NormalDecoder(dx, dy)

    align = Align(dy_other, dy, distribution=noise)

    vae = AlignSeqVae(prior, encoder, decoder, align, device, K_vae, K_align)
    train_dataloader = get_training_data(data[0]['y'], data[2]['y'], normalize=True)

    "let's train our vae"
    results = alignvae_training(vae, align, train_dataloader, n_epochs=5, n_epochs_vae=100,
                                            n_epochs_align=50, lr=1e-3)
    torch.save(results, f'results_{noise}_alternate.pt')


if __name__ == '__main__':
    main()






