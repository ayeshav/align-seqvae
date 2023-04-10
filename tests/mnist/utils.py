import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli
from vae import VAE


def compute_elbo(vae, x):

    x_c = torch.flatten(x, start_dim=1)

    vae_output = vae(x_c)

    "compute log prob"
    log_like = torch.sum(Bernoulli(vae_output[3]).log_prob(x_c),-1)

    log_prior = torch.sum(Normal(0,1).log_prob(vae_output[2]),-1)

    log_enc = torch.sum(Normal(vae_output[0], vae_output[1]).log_prob(vae_output[2]),-1)

    elbo = torch.mean(log_like - (log_enc - log_prior))

    return -elbo


def vae_training(vae, epochs, data):

    opt = torch.optim.Adam(vae.parameters())
    for _ in range(epochs):
        for x, y in data:
            opt.zero_grad()
            loss = compute_elbo(vae, torch.bernoulli(x)) # x is of shape batch_size x 1 x 28 x 28
            loss.backward()
            opt.step()
    return vae


def rotate_image(img, angle):
    """
    Rotate the given image file by the given angle in degrees,
    and return the rotated image object.
    """

    # Convert angle to radians
    radians = np.deg2rad(angle)

    # Calculate sin and cosine of angle
    c, s = np.cos(radians), np.sin(radians)

    # Get width and height of image
    w, h = img.size

    # Calculate new image dimensions to include rotated image
    new_w = int(abs(w * c) + abs(h * s))
    new_h = int(abs(w * s) + abs(h * c))

    # Create new image with white background
    rotated_img = Image.new('RGB', (new_w, new_h), color='black')

    # Calculate center point of original image
    center_x = int(w / 2)
    center_y = int(h / 2)

    # Iterate over every pixel in rotated image
    for x in range(new_w):
        for y in range(new_h):
            # Calculate the corresponding pixel in the original image
            orig_x = int((x - new_w / 2) * c - (y - new_h / 2) * s + center_x)
            orig_y = int((x - new_w / 2) * s + (y - new_h / 2) * c + center_y)

            # If the corresponding pixel is within the bounds of the original image
            if 0 <= orig_x < w and 0 <= orig_y < h:
                # Get the pixel color from the original image and set it in the rotated image
                color = img.getpixel((orig_x, orig_y))
                rotated_img.putpixel((x, y), int(color))

    return rotated_img


