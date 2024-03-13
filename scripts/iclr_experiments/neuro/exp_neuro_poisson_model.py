import os
import torch
import pickle
import sys
sys.path.append('../../../src')

from src import *

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


def load_data(filepath, session_id):

    with open(filepath, 'rb') as f:
        data_sessions = pickle.load(f)

    data = data_sessions[session_id]

    rates = torch.from_numpy(data['y']).float()
    velocity = torch.from_numpy(data['velocity']).float()

    target = torch.from_numpy(data['target'].astype(int))
    target_ohe = torch.nn.functional.one_hot(target).unsqueeze(1).repeat(1, rates.shape[1], 1)

    return rates, target_ohe, velocity


def get_train_dataloader(data, batch_size=32, train_ratio=0.8):

    rates, target, velocity = data

    n_train = int(train_ratio * rates.shape[0])

    train_data = rates[:n_train], target[:n_train], velocity[:n_train],
    train_dataloader = SeqDataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_model(params, datapath, session_id):

    data = load_data(datapath, session_id)
    dy, du, dv = data[0].shape[-1], data[1].shape[-1], data[2].shape[-1]
    dx, dh, d_embed = params['dx'], params['dh'], params['d_embed']

    train_dataloader = get_train_dataloader(data, params['batch_size'], params['train_ratio'])

    "define model for training"
    encoder = EmbeddingEncoder(dy, dx, dh, d_embed, du=du, device=device)
    prior = PriorGru(dx + du, dh, device=device)
    decoder = BinomialDecoder(dx, dy, total_count=8, device=device)

    vel_decoder = NormalDecoder(dx, dv)

    vae = CondSeqVae(prior, encoder, decoder, vel_decoder, k_step=10)

    vae, train_losses = vae_training(vae, train_dataloader, n_epochs=params['n_epochs'], lr=params['lr'],
                                     weight_decay=params['weight_decay'])

    return vae, train_losses


if __name__ == "__main__":

    datapath = '../../../data/neuro/processed/co_reaching_move_begins_time.pkl'
    savepath = '../../../results/neuro/trained_model/'

    session_id = 'mihi-03032014'

    "define model and optimization params"
    params = {'dx': 30,
              'd_embed': 64,
              'dh': 64,
              'train_ratio': 0.8,
              'batch_size': 32,
              'n_epochs': 1_000,
              'weight_decay': 1e-2,
              'lr': 1e-3}

    model, loss = train_model(params, datapath, session_id)

    torch.save((model, params, loss), os.path.join(savepath + 'neuro_poisson.pt'))




