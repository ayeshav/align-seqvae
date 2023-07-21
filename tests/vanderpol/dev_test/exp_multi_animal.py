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

N_epochs_vae = 2_000
batch_size = 100

noise = 'Normal'


def dualvae_training(vae, train_dataloader, n_epochs=100, lr=5e-4,
                 weight_decay=1e-4):
    """
    function that will train a vae
    :param vae: a SeqVae object
    :param train_dataloader: a dataloader object
    :param n_epochs: Number of epochs to train for
    :param lr: learning rate of optimizer
    :param weight_decay: value of weight decay
    :return: trained vae and training losses
    """
    assert isinstance(vae, DualAnimalSeqVae)
    assert train_dataloader.shuffle
    assert isinstance(train_dataloader, SeqDataLoader)

    param_list = list(vae.parameters())

    opt = torch.optim.AdamW(params=param_list, lr=lr, weight_decay=weight_decay)
    training_losses = []
    for _ in tqdm(range(n_epochs)):
        for y, y_other in train_dataloader:
            opt.zero_grad()
            loss = vae(y.to(vae.device), y_other.to(vae.device))
            loss.backward()
            opt.step()

            with torch.no_grad():
                training_losses.append(loss.item())
    return vae, training_losses


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
    encoder = DualAnimalEncoder(dy, dy_other, dx, dh)
    prior = Prior(dx)
    decoder = DualNormalDecoder(dx, dy, dy_other)

    vae = DualAnimalSeqVae(prior, encoder, decoder, device, K_vae)
    train_dataloader = get_training_data(data[0]['y'], data[2]['y'], normalize=True)

    "let's train our vae"
    ref_vae, losses_vae = dualvae_training(vae, train_dataloader, n_epochs=N_epochs_vae, lr=1e-3)
    torch.save((ref_vae, vae), f'results_{noise}.pt')


if __name__ == '__main__':
    main()






