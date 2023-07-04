import os
import numpy as np
from utils import *
from seq_vae import SeqVae
from align import *

import matplotlib.pyplot as plt

dx = 2
dh = 64

if torch.has_mps:
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = 'cpu'
print(device)

"load vanderpol data"
data = torch.load('data/noisy_vanderpol.pt')


def train_ref_vae():
    "extract reference data"
    # data is batch by time by dimension
    y = data[0]['y'].float()
    dy_ref = y.shape[2]
    N_train = int(np.ceil(0.8 * y.shape[0]))
    y_train = y[:N_train]  # Batch by Time by Dimension

    # Let's whiten the data to make it more numerically stable
    mu_train = torch.mean(y_train.reshape(-1, dy_ref), 0, keepdim=True)
    sigma_train = torch.std(y_train.reshape(-1, dy_ref), 0, keepdim=True)
    y_train = (y_train - mu_train) / sigma_train

    data_ref = SeqDataLoader((y_train,), batch_size=100, shuffle=True)
    vae = SeqVae(dx, dy_ref, dh, device=device)
    res = vae_training(vae, data_ref, lr=1e-3, n_epochs=2_000)

    torch.save(res, 'trained_models/reference_model.pt')


def reuse_dynamics(reference_model, epochs=20):

    vae, _ = torch.load(reference_model)

    y = data[1]['y'].float()
    N_train = int(np.ceil(0.8 * y.shape[0]))
    y_train = y[:N_train]  # Batch by Time by Dimension

    dy_ref = data[1]['y'].float().shape[2]

    # for i in range(1, len(data)):
    y_dataloader = SeqDataLoader((y_train,), batch_size=100, shuffle=True)
    res_alignment = train_invertible_mapping(vae, y_dataloader, dy_ref, epochs)

    return res_alignment


def main():

    model_path = 'trained_models'
    results_path = 'results'

    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    if not os.path.isfile(model_path + '/reference_model.pt'):
        train_ref_vae()

    res_alignment = reuse_dynamics(model_path + '/reference_model.pt', 250)
    torch.save(res_alignment, 'result.pt')


if __name__ == '__main__':
    main()






