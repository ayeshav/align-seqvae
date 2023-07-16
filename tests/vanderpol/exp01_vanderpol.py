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
data = torch.load('data/noisy_vanderpol_bernoulli.pt')


def train_ref_vae(normalize=True, featurize=False):
    "extract reference data"
    # data is batch by time by dimension
    y = data[0]['y'].float()
    dy_ref = y.shape[2]
    N_train = int(np.ceil(0.8 * y.shape[0]))
    y_train = y[:N_train]  # Batch by Time by Dimension

    # Let's whiten the data to make it more numerically stable
    if normalize:
        mu_train = torch.mean(y_train.reshape(-1, dy_ref), 0, keepdim=True)
        sigma_train = torch.std(y_train.reshape(-1, dy_ref), 0, keepdim=True)
        y_train = (y_train - mu_train) / sigma_train

    if featurize:
        net = Mlp(dy_ref, 64, 128, device=device)
    else:
        net = None

    data_ref = SeqDataLoader((y_train,), batch_size=100, shuffle=True)
    vae = SeqVae(dx, 64, dh, dy_ref, likelihood='Bernoulli', k_step=10, device=device)
    res = vae_training(vae, data_ref, inp_tfm=net, lr=1e-3, n_epochs=2_50)

    torch.save(res, 'trained_models/reference_model_bernoulli.pt')


def reuse_dynamics(reference_model, epochs=20):

    vae, _ = torch.load(reference_model, map_location=torch.device('cpu'))
    ref_vae = vae[0].to('cpu')

    y = data[1]['y'].float()
    N_train = int(np.ceil(0.8 * y.shape[0]))
    y_train = y[:N_train]  # Batch by Time by Dimension

    dy_ref = data[0]['y'].float().shape[2]

    # with torch.no_grad():
    #     x_samples, mu, var, _ = vae.encoder(data[0]['y'].float())
    #
    #     mu_ref = torch.mean(x_samples.reshape(x_samples.shape[-1], -1), 1, keepdim=True)
    #     cov_ref = torch.cov(x_samples.reshape(x_samples.shape[-1], -1))

    # for i in range(1, len(data)):
    y_dataloader = SeqDataLoader((y_train,), batch_size=100, shuffle=True)
    res_alignment = train_invertible_mapping(ref_vae, y_dataloader, dy_ref, epochs, dy_out=64, K=10,
                                             distribution='Bernoulli', linear_flag=True)

    return res_alignment


def main():

    model_path = 'trained_models'
    results_path = 'results'

    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    if not os.path.isfile(model_path + '/reference_model_bernoulli.pt'):
        train_ref_vae(normalize=False, featurize=True)

    res_alignment = reuse_dynamics(model_path + '/reference_model_bernoulli.pt', 500)
    torch.save(res_alignment, 'result_bernoulli.pt')


if __name__ == '__main__':
    main()






