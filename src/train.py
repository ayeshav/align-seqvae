import torch
from tqdm import tqdm
from seq_vae import *
from utils import *


def dualvae_training(vae, train_dataloader, n_epochs=100, lr=5e-4,
                     weight_decay=1e-4, beta=1.0, reg_weight=(100, 100)):
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
    training_losses, test_losses = [],[]
    for _ in tqdm(range(n_epochs)):
        for y, y_other in train_dataloader:
            opt.zero_grad()
            loss = vae(y.to(vae.device), y_other.to(vae.device), beta=beta, reg_weight=reg_weight)
            loss.backward()
            opt.step()

            with torch.no_grad():
                training_losses.append(loss.item())
    return vae, training_losses


def dualvae_coordinate_ascent_training(vae, train_dataloader, n_epochs=100, lr=5e-4,
                                       weight_decay=1e-4, beta=1.0, reg_weight=(100, 100)):
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

    training_losses = []

    train_other_animal = True
    for j in tqdm(range(n_epochs)):
        if train_other_animal:
            param_list = list(vae.f_enc.parameters()) + list(vae.other_animal_decoder.parameters())
            train_other_animal = False
        else:
            param_list = list(vae.encoder.parameters()) + list(vae.prior.parameters()) + list(vae.decoder.parameters())
            train_other_animal = True

        opt = torch.optim.AdamW(param_list, lr=lr, weight_decay=weight_decay)

        for y, y_other in train_dataloader:
            opt.zero_grad()
            loss = vae(y.to(vae.device), y_other.to(vae.device), beta=beta, reg_weight=reg_weight)
            loss.backward()
            opt.step()

            with torch.no_grad():
                training_losses.append(loss.item())
    return vae, training_losses


def vae_training(vae, train_dataloader, n_epochs=100, lr=5e-4,
                 weight_decay=1e-4, beta=1.0):
    """
    function that will train a vae
    :param vae: a SeqVae object
    :param train_dataloader: a dataloader object
    :param n_epochs: Number of epochs to train for
    :param lr: learning rate of optimizer
    :param weight_decay: value of weight decay
    :return: trained vae and training losses
    """
    assert isinstance(vae, SeqVae)
    assert train_dataloader.shuffle
    assert isinstance(train_dataloader, SeqDataLoader)

    param_list = list(vae.parameters())

    opt = torch.optim.AdamW(params=param_list, lr=lr, weight_decay=weight_decay)
    training_losses = []
    for _ in tqdm(range(n_epochs)):
        for y, in train_dataloader:
            opt.zero_grad()
            loss = vae(y.to(vae.device), beta=beta)
            loss.backward()
            opt.step()

            with torch.no_grad():
                training_losses.append(loss.item())
    return vae, training_losses


def condvae_training(vae, train_dataloader, n_epochs=100, lr=5e-4,
                     weight_decay=5e-4, beta=1.):
    """
    function that will train a vae
    :param vae: a SeqVae object
    :param train_dataloader: a dataloader object
    :param n_epochs: Number of epochs to train for
    :param lr: learning rate of optimizer
    :param weight_decay: value of weight decay
    :return: trained vae and training losses
    """
    assert isinstance(vae, CondSeqVae)
    assert train_dataloader.shuffle
    assert isinstance(train_dataloader, SeqDataLoader)

    opt = torch.optim.Adam(params=vae.parameters(), lr=lr, weight_decay=weight_decay)
    training_losses = []
    for _ in tqdm(range(n_epochs)):
        for y, u, vel in train_dataloader:
            opt.zero_grad()
            loss = vae(y.to(vae.device), u.to(vae.device), y_behav=vel.to(vae.device), beta=beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(),
                                           max_norm=1., norm_type=2)
            opt.step()

            with torch.no_grad():
                training_losses.append(loss.item())

    return vae, training_losses


def alignment_training(ref_vae, align, train_dataloader, beta=1.0, n_epochs=100, lr=1e-3):
    """
    function for training alignment parameters
    :param ref_vae: pre-trained vae
    :param align: align object
    :param train_dataloader: a dataloader object
    :param ref_ss: sufficient stats for reference latents
    """
    assert isinstance(train_dataloader, SeqDataLoader)
    assert train_dataloader.shuffle

    training_losses = []
    opt = torch.optim.AdamW(params=align.parameters(),
                            lr=lr, weight_decay=1e-4)

    for _ in tqdm(range(n_epochs)):
        for y, in train_dataloader:
            opt.zero_grad()
            loss = align(ref_vae, y.to(ref_vae.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(align.parameters(),
                                           max_norm=1., norm_type=2)
            opt.step()

            with torch.no_grad():
                training_losses.append(loss.item())

    return align, training_losses