import torch
import tqdm
from seq_vae import SeqVae
from utils import *


def vae_training(vae, train_dataloader, n_epochs=100, lr=5e-4,
                 weight_decay=1e-4, n_samples=1):
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
            loss = vae(y.to(vae.device), n_samples=n_samples)
            loss.backward()
            opt.step()

            with torch.no_grad():
                training_losses.append(loss.item())
    return vae, training_losses


def alignment_training(ref_vae, align, train_dataloader, n_epochs=100, lr=1e-3):
    """
    training function for learning linear alignment and updating prior params
    """
    assert isinstance(train_dataloader, SeqDataLoader)
    assert train_dataloader.shuffle

    training_losses = []
    opt = torch.optim.AdamW(params=align.parameters(),
                            lr=lr, weight_decay=1e-4)

    for _ in tqdm(range(n_epochs)):
        for y, in train_dataloader:
            opt.zero_grad()
            loss = align(ref_vae, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(align.parameters(),
                                           max_norm=1., norm_type=2)
            opt.step()

            with torch.no_grad():
                training_losses.append(loss.item())

    return align, training_losses