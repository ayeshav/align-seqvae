import numpy as np
import torch
import matplotlib.pyplot as plt

noise_type = 'bernoulli'
sigmoid = lambda z: 1 / (1 + np.exp(-z))
softplus = lambda z: np.log(1 + np.exp(z))

data = torch.load(f'data/noisy_vanderpol_{noise_type}.pt')

# In[]
j = 0
x = data[j]['x']
y = data[j]['y']
C = data[j]['C']
b = data[j]['b']

# In[]
if noise_type == 'poisson':
    rates = softplus(x @ C + b)
elif noise_type == 'bernoulli':
    # xs = np.vstack([x[k] for k in range(x.shape[0])])
    # mu = np.mean(xs, 0, keepdims=True)
    # sigma = np.std(xs, 0, keepdims=True)
    #
    # x_normalized = (x - mu) / sigma  # should broadcast correctly
    rates = sigmoid((x @ C + b))


# In[]
fig, ax = plt.subplots(1, 1, dpi=200, figsize=(8, 6))

for k in range(1):
    for d in range(100):
        ax.plot(rates[k, :, d], alpha=0.3)
fig.show()

# In[]
fig, ax = plt.subplots(1, 1, dpi=200, figsize=(8, 6))
ax.matshow(y[0].T, aspect='auto', cmap='coolwarm')
fig.show()

# In[]
y_stacked = np.vstack([y[k] for k in range(y.shape[0])])
y_stacked = np.concatenate((y_stacked, np.ones((y_stacked.shape[0], 1))), -1)
x_stacked = np.vstack([x[k] for k in range(y.shape[0])])
# In[]
A = np.linalg.lstsq(y_stacked, x_stacked)[0]
A = torch.from_numpy(A).float()
# In[]
x_hat = y @ A[:-1] + A[-1]

# In[]
fig, axs = plt.subplots(1, 2, dpi=200)

for k in range(x_hat.shape[0]):
    axs[0].plot(x_hat[k, :, 0], x_hat[k, :, 1], alpha=0.2)
    axs[1].plot(x[k, :, 0], x[k, :, 1], alpha=0.2)
fig.tight_layout()
fig.show()