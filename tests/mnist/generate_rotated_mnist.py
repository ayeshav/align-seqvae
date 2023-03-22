import numpy as np
import torch
from torchvision import datasets, transforms
import torchvision.transforms.functional as tf

n_rots = 5
angles = [5, 25, 45, 65, 85]

train_dataset = datasets.MNIST(root='./mnist_data/', train=True,
                               transform=transforms.ToTensor(),
                               download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False,
                              transform=transforms.ToTensor(),
                              download=False)

augmented_data = {}

for rot in angles:
    data = {}

    x_rot_tr = [tf.rotate(train_dataset.data[i][np.newaxis], rot) for i in range(60_000)]
    data['train'] = (torch.stack(x_rot_tr), train_dataset.train_labels)

    x_rot_te = [tf.rotate(test_dataset.data[i][np.newaxis], rot) for i in range(10_000)]
    data['test'] = (torch.stack(x_rot_te), test_dataset.test_labels)

    augmented_data[rot] = data

torch.save(augmented_data, 'rotated_mnist.pt')