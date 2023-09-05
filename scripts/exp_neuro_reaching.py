import sys
sys.path.append('../src')

import torch
import numpy as np
from utils import *
from train import *

from encoders import EmbeddingEncoder
from decoders import NormalDecoder
from priors import Prior
from seq_vae import CondSeqVae


if torch.has_mps:
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device='cpu'

print(device)

# load neural data
all_data = torch.load('../data/data_co_reaching.pt')
data = all_data['mihi'][0]

# adding some global variables that we can later port to a config file
dx = 30
dh = 64
d_embed = 64
K, T, dy = data['y'].shape
dy_behav = data['vel'].shape[-1]

u = data['target']
du = u.shape[-1]

train_ratio = 0.7
N_train = int(np.ceil(train_ratio * K))

batch_size = 32

def get_data():

    y = data['rates']
    vel = normalize(data['vel'])

    train_data = y[:N_train], u[:N_train], vel[:N_train]
    train_dataloader = SeqDataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def main():

    train_dataloader = get_data()

    prior = Prior(dx,  du=du, device=device)

    encoder = EmbeddingEncoder(dy + du, dx, dh=dh, d_embed=d_embed, device=device)
    decoder = NormalDecoder(dx, dy, device=device)
    readout_behav = NormalDecoder(dx, dy_behav, device=device)

    vae = CondSeqVae(prior, encoder, decoder, readout_behav=readout_behav, k_step=15, device=device)

    vae, losses_train = condvae_training(vae, train_dataloader, n_epochs=2_000, lr=1e-3,
                                         weight_decay=1e-2)

    torch.save((vae.state_dict(), losses_train), 'neuro_vae_reaching_mihi_0.pt')


if __name__ == '__main__':
    main()