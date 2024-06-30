## Neural Time Series Alignment

The code for unsupervised alignment of neural time series observations with the same underlying latent dynamics. For more details see:
> A. Vermani, I. M. Park, and J. Nassar, “Leveraging generative models
>for unsupervised alignment of neural time series data,” in International
> Conference on Learning Representations (ICLR), 2024

#### Running the Code

Example code for training a base sequential VAE model and aligning new observations can be found in `notebooks`. In order to run the code,

1. Install dependencies in `environment.yml`

2. Go to `data/low_d/vdp` and run `generate_vanderpol.py`
