import numpy as np
import torch
from scipy.stats import chi2


def get_device():
	"""Returns the best available device: CUDA, then Apple MPS, then CPU."""
	if torch.cuda.is_available():
		return torch.device('cuda')
	if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
		return torch.device('mps')
	return torch.device('cpu')


def lambda0_ladder(epochs, lambda0, lambda1, warmup_frac=0.04, middle_frac=0.25, lambda0_start=None):
	"""Step schedule that increases the spike penalty lambda0 during training.

	A weak spike early on lets W learn which features load on which factors
	before strong sparsity is imposed. The schedule has three rungs:

	- ``lambda0_start`` (default: ``lambda1``, i.e. no spike) for the first
	  ``warmup_frac`` of epochs,
	- ``2/3 * lambda0`` for the next ``middle_frac`` of epochs,
	- ``lambda0`` for the remaining epochs.

	Starting above ``lambda1`` is not recommended when ``lambda1`` is small: a
	weak-but-distinct spike classifies every entry of W as spike and the
	inclusion weights thetas collapse before training has started.

	Pass the result to ``ModelTrainer.train(..., lambda0_schedule=...)``.

	Returns:
		np.ndarray of length ``epochs``.
	"""
	if lambda0_start is None:
		lambda0_start = lambda1

	n_warm = int(round(warmup_frac * epochs))
	n_mid = int(round(middle_frac * epochs))

	schedule = np.full(epochs, float(lambda0))
	schedule[:n_warm + n_mid] = 2 / 3 * lambda0
	schedule[:n_warm] = lambda0_start
	return schedule


def sigma_prior(x, sig_quant=0.9, sig_df=3, init_frac=0.8):
	"""Data-driven prior and initial values for the per-feature noise standard deviations (mse loss).

	A rough noise level ``sig_est`` is the 5% quantile of the per-feature standard
	deviations. The scaled-inverse-chi^2 prior on sigma_j^2 puts probability
	``sig_quant`` on sigma_j < sig_est. Each sigma_j is initialized at
	``init_frac`` times the standard deviation of feature j.

	Args:
		x: data array of shape (n_samples, n_features).

	Returns:
		(sigma_prior_scale, sigmas_init) to pass to the model as
		``sigma_prior_scale`` and ``sigmas_init``.
	"""
	sd = np.asarray(x).std(axis=0)

	sig_est = np.quantile(sd, q=0.05)
	if sig_est == 0:
		sig_est = 1e-3

	q_chi = chi2.ppf(1 - sig_quant, sig_df)
	sig_scale = sig_est * sig_est * q_chi / sig_df

	sigmas_init = init_frac * sd
	return sig_scale, sigmas_init
