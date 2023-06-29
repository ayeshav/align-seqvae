import numpy as np
import numpy.random as npr
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from seq_vae import SeqVae, Prior
from utils import train_invertible_mapping
# In[]
"let;s load in the data"
data = torch.load('noisy_vanderpol.pt')

# In[]
y, y_ref = data[1]['y'].float(), data[0]['y'].float()

# In[]
# load in vae model
dx = 2
dh = 256
dy_ref = y_ref.shape[2]
vae, prior, _ = torch.load('reference_model.pt')

# In[]
linear_map, training_losses = train_invertible_mapping(100, vae, prior,
                                                       y[:50],
                                                       # y.permute(1, 0, 2),
                                                       dy_ref)

# In[]
plt.plot(training_losses)
plt.show()

# In[]
# let's see how the latent look
y_tfm = y @ linear_map  # apply linear transformation to new dataset
with torch.no_grad():
    encoder_params, likelihood_params, log_prior = vae(y_tfm, prior)  # for the given dataset

x_samples = encoder_params[2]  # sample from the encoder after doing the transformation

# In[]
fig, axs = plt.subplots(1, 2, figsize=(8, 4), dpi=200)
for k in range(x_samples.shape[1]):
    axs[0].plot(x_samples[:, k, 0], x_samples[:, k, 1], alpha=0.3)
    axs[1].plot(data[1]['x'].float()[:, k, 0], data[1]['x'].float()[:, k, 1], alpha=0.3)
axs[0].set_title('Latents from encoder')
axs[1].set_title('True latents')
fig.tight_layout()
fig.show()
# In[]
x_true = data[1]['x'].float()

# In[]
mu_like_tfm, sigma_like_tfm, _ = likelihood_params

# In[]
inv_linear_map = torch.linalg.pinv(linear_map)
mu_like = mu_like_tfm @ inv_linear_map

# In[]
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'black', 'slategray']
counter = 0

fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=200)

for k in range(1):
    for d in range(1):
        counter = d % len(colors)
        ax.plot(y[:100, k, d].detach(), color=colors[counter], alpha=0.3)
        ax.plot(mu_like[:100, k, d].detach(), color=colors[counter], alpha=0.3, linestyle='--')
fig.show()
