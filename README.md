# Sparse VAE

PyTorch implementation of the Sparse VAE from "Identifiable Deep Generative Models via Sparse Decoding" [(arXiv)](https://arxiv.org/abs/2110.10804).

The Sparse VAE places a spike-and-slab lasso prior on a mask `W` (features × latent factors) in the decoder, so that each observed feature depends on only a few latent factors.

> **Reproducing the paper:** this package is under active development and its results may differ from the paper. The exact code used for the paper (including the baselines, data processing, and experiment scripts) is at the [`v1.0-paper`](https://github.com/gemoran/sparse-vae-code/tree/v1.0-paper) tag:
> ```bash
> git clone https://github.com/gemoran/sparse-vae-code.git
> cd sparse-vae-code && git checkout v1.0-paper
> ```

## Installation

Requires Python ≥ 3.9 and PyTorch ≥ 2.0.

```bash
pip install git+https://github.com/gemoran/sparse-vae.git
```


## Usage

```python
import torch
from torch.utils.data import DataLoader
from sparsevae import SparseVAESpikeSlab, ModelTrainer, sigma_prior

# x: numpy array of shape (n_samples, n_features)
sig_scale, sig_init = sigma_prior(x)

model = SparseVAESpikeSlab(input_dim=x.shape[1], latent_dim=5,
                           lambda0=15., lambda1=0.1,
                           sigma_prior_scale=sig_scale, sigmas_init=sig_init)

loader = DataLoader(torch.tensor(x, dtype=torch.float), batch_size=100, shuffle=True)
ModelTrainer(model).train(loader, epochs=200, lr=0.01)

W = model.W.detach().cpu().numpy()                     # estimated sparse mask
support = model.p_star.detach().cpu().numpy() >= 0.5   # entries of W assigned to the slab
```

`sigma_prior` sets the prior on the per-feature noise standard deviations and initializes each one at 80% of that feature's standard deviation. During training, these are held fixed for the first 80% of epochs and then learned.

The priors on `W` and the noise standard deviations are weighted by the number of training samples, which `ModelTrainer` takes from the data loader (`len(loader.dataset)`). Results therefore do not depend on the batch size, but the same `lambda0` gives weaker sparsity on larger datasets.

The device (CUDA, Apple MPS, or CPU) is chosen automatically; pass `ModelTrainer(model, device=...)` to override.

Main model options (`SparseVAESpikeSlab`):

| Argument | Default | Description |
|---|---|---|
| `latent_dim` | – | number of latent factors (can be larger than the true number; unused factors are switched off) |
| `lambda0`, `lambda1` | `15`, `0.1` | spike and slab penalties on `W` |
| `loss_type` | `'mse'` | `'mse'` (Gaussian), `'binary'`, or `'categorical'` |
| `hidden_dim` | `300` | width of encoder/decoder hidden layers; smaller networks (e.g. `50`) give sparser `W` on small datasets |
| `decode_vec` | `True` | `True` uses a vectorized decoder; `False` uses a per-feature loop |

Training options (`ModelTrainer.train`):

| Argument | Default | Description |
|---|---|---|
| `epochs`, `lr` | `10`, `0.01` | number of epochs and Adam learning rate |
| `freeze_sigmas_frac` | `0.8` | fraction of epochs during which the noise standard deviations are held at their initial values |
| `lambda0_schedule` | `None` | optional per-epoch values of `lambda0`, e.g. `lambda0_ladder(epochs, lambda0=15., lambda1=0.1)`, which increases the spike penalty in steps |
| `n_samples` | `len(loader.dataset)` | training set size, for loaders without a length |

See [`examples/simulated.py`](examples/simulated.py) for a complete example on the simulated data from the paper.

## Citation

```bibtex
@article{
moran2022identifiable,
title={Identifiable Deep Generative Models via Sparse Decoding},
author={Gemma E. Moran and Dhanya Sridhar and Yixin Wang and David Blei},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2022},
url={https://openreview.net/forum?id=vd0onGWZbE},
note={}
}
```

The DCI metric in `sparsevae/dci.py` is adapted from [disentanglement_lib](https://github.com/google-research/disentanglement_lib) (Apache 2.0).
