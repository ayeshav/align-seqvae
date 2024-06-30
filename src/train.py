import torch
from tqdm import tqdm
from src.utils import *
from src.seq_vae import *

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
    assert isinstance(vae, SeqVae) or isinstance(vae, CondSeqVae)
    assert train_dataloader.shuffle
    assert isinstance(train_dataloader, SeqDataLoader)

    param_list = list(vae.parameters())
    opt = torch.optim.AdamW(params=param_list, lr=lr, weight_decay=weight_decay)
    training_losses = []
    for _ in tqdm(range(n_epochs)):
        for batch in train_dataloader:
            opt.zero_grad()
            loss = vae(to_device(batch, vae.device), beta=beta)
            loss.backward()
            opt.step()
            with torch.no_grad():
                training_losses.append(loss.item())
    return vae, training_losses


def alignment_training(ref_vae, align, train_dataloader, beta=1.0, n_epochs=500, lr=1e-3, weight_decay=1e-4, k_step_list=None):
    """
    function for training alignment parameters
    :param ref_vae: pre-trained vae
    :param align: align object
    :param train_dataloader: a dataloader object
    :param k_step_list: list of length n_epochs for stochastic k_step loss
    """
    assert isinstance(train_dataloader, SeqDataLoader)
    assert train_dataloader.shuffle

    training_losses = []
    opt = torch.optim.AdamW(params=align.parameters(),
                            lr=lr, weight_decay=weight_decay)

    for i in tqdm(range(n_epochs)):

        if k_step_list is not None:
            align.k_step = int(k_step_list[i])
        for batch in train_dataloader:
            opt.zero_grad()
            loss = align(ref_vae, to_device(batch, ref_vae.device), beta=beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(align.parameters(),
                                           max_norm=1., norm_type=2)
            opt.step()

            with torch.no_grad():
                training_losses.append(loss.item())

    return align, training_losses