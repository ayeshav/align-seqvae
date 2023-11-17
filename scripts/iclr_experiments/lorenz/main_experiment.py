
import sys
sys.path.append('../../../src')
sys.path.append('../../../results')

import torch
import numpy as np
from utils import *

from encoders import Encoder
from priors import Prior
from decoders import NormalDecoder
from seq_vae import SeqVae

from align import *
from train import *
from utils import SeqDataLoader, normalize
from plotting import *

import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

data = torch.load('../../../data/lorenz_gaussian.pt')
vae_sd = torch.load('vae_gaussian_lorenz_sd.pt', map_location=device)

dx = 3
dy = data[0]['y'].shape[-1]
dh = 64

"load the reference model"
encoder = Encoder(dy, dx, dh, device=device)
prior = Prior(dx, device=device)
decoder = NormalDecoder(dx, dy, device=device)

vae = SeqVae(prior, encoder, decoder, device=device)
vae.load_state_dict(vae_sd)


def get_training_data(y, train_ratio, batch_size=100, norm=True):

    N_train = int(np.ceil(train_ratio * y.shape[0]))
    y_train = y[:N_train]

    if norm:
        y_train = normalize(y_train)

    train_dataloader = SeqDataLoader((y_train,), batch_size=batch_size)
    return train_dataloader


"train the alignment"
results = {'align': [], 'losses_align': []}

ks = torch.ones(500)
ks[250:] = torch.from_numpy(np.random.choice(np.arange(1, 3), 250))

for i in range(2):
    dy_new = data[i+1]['y'].shape[-1]
    decoder_new = NormalDecoder(dx, dy_new, device=device)

    train_dataloader = get_training_data(data[i+1]['y'], train_ratio=0.5, norm=False)

    align = Align(dy_new, dy, decoder_new, k_step=2, linear_flag=True, device=device)
    res_align, losses_align = alignment_training(vae, align, train_dataloader, n_epochs=500,
                                                 lr=5e-3, weight_decay=1, ks=ks)

    results['align'].append(res_align)
    results['losses_align'].append(losses_align)


"let's plot the results"
n_test = 1600
vae.encoder.device = 'cpu'
vae.encoder.to('cpu')

vae.device = 'cpu'

y_s = data[0]['y'][n_test:]

with torch.no_grad():
    x_s, _, _ = vae.encoder.sample(y_s)

"sample some trajectories after alignment"
x_ts = []

for i in range(2):
    y_t = data[i+1]['y'][n_test:, :]

    results['align'][i].g.to('cpu')
    with torch.no_grad():
        y_tfm = results['align'][i].g(y_t)
        x_t, _, _ = vae.encoder.sample(y_tfm)

    x_ts.append(x_t)

"plot the trajectories"
fig = plt.figure(figsize=(8, 6))

ax = fig.add_subplot(1, 2, 1, projection='3d')
for i in range(20):
    ax.plot(x_s[i, :, 1], x_s[i, :, 0], zs=x_s[i, :, 2])

colors = ['tab:blue', 'tab:orange']

ax = fig.add_subplot(1, 2, 2, projection='3d')
for j in range(2):
    for i in range(20):
        ax.plot(x_ts[j][i, :, 1], x_ts[j][i, :, 0], zs=x_ts[j][i, :, 2], alpha=0.8, color=colors[j])
plt.show()

"test prediction performance"
T_train = 450
k = 50

r2s = []

for i in range(2):
    y_test = data[i+1]['y'][n_test:]

    y_pred = generate_k_step_pred(vae, y_test, T_train, k, results['align'][i].decoder.to('cpu'),
                                  align=results['align'][i], method='align')

    r2 = compute_prediction_r2(y_test, y_pred, T_train, k)

    r2s.append(r2)

fig = plt.figure(figsize=(6, 6))

plt.plot(np.arange(1, k + 1), torch.hstack(r2s).mean(1), 'o')
plt.ylabel('r2', fontsize=12)
plt.xlabel('prediction step', fontsize=12)
plt.show()






