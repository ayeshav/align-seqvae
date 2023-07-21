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
# device='cpu'

print(device)


"load vanderpol data"
data = torch.load('../data/noisy_vanderpol_gaussian.pt')

"adding some global variables that we can later port to a config file"

dx = 2
dh = 64
d_embed = 64

train_ratio = 0.8

K_vae = 1
K_align = 1

N_epochs_vae = 2_000
N_epochs_align = 500

batch_size = 100

wasserstein = False

noise = 'Normal'


def get_training_data(y, normalize=True):

    K, _, dy = y.shape

    "let's get the reference data"
    N_train = int(np.ceil(train_ratio * K))
    y_train = y[:N_train]

    if normalize:
        mu_train = torch.mean(y_train.reshape(-1, dy), 0, keepdim=True)
        sigma_train = torch.std(y_train.reshape(-1, dy), 0, keepdim=True)
        y_train = (y_train - mu_train) / sigma_train

    train_dataloader = SeqDataLoader((y_train,), batch_size=batch_size)
    return train_dataloader


def get_sufficient_statistics(vae, data):
    with torch.no_grad():
        x_samples = vae.encoder.sample(data)[0]

        mu_ref = torch.mean(x_samples.reshape(x_samples.shape[-1], -1), 1, keepdim=True)
        cov_ref = torch.cov(x_samples.reshape(x_samples.shape[-1], -1))

    return mu_ref, cov_ref


def main():

    K, _, dy_ref = data[0]['y'].shape

    "let's define our vae"
    encoder = EmbeddingEncoder(dy_ref, dx, dh, d_embed)
    prior = Prior(dx)
    decoder = BinomialDecoder(dx, dy_ref)

    vae = SeqVae(prior, encoder, decoder, device, K_vae)
    train_dataloader_ref = get_training_data(data[0]['y'], normalize=False)

    "let's train our vae"
    ref_vae, losses_vae = vae_training(vae, train_dataloader_ref, n_epochs=N_epochs_vae, lr=1e-3)

    if wasserstein:
        ref_ss = get_sufficient_statistics(ref_vae, data[0]['y'])
    else:
        ref_ss = None

    results = {'ref_vae': ref_vae,
               'align': [],
               'losses_vae': losses_vae,
               'losses_align': []}

    for i in range(len(data)-1):
        dy = data[i]['y'].shape[-1]

        "let's get data for alignment"
        train_dataloader_new = get_training_data(data[i]['y'], normalize=False)

        "let's define our alignment"
        align = Align(dy, dy_ref, K=K_align, distribution=noise)

        params_align, losses_align = alignment_training(vae, align, train_dataloader_new, ref_ss=ref_ss)

        results['align'].append(params_align)
        results['losses_align'].append(losses_align)

    torch.save(results, f'results_{noise}.pt')


if __name__ == '__main__':
    main()






