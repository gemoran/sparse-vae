import sys

import numpy as np
import torch
from torch import optim

from sparsevae.utils import get_device


def _unpack_batch(batch):
	"""Accepts a dict with 'data' (and optionally 'normalized_data'), a tuple/list
	whose first element is the data, or a plain tensor."""
	if isinstance(batch, dict):
		return batch['data'], batch.get('normalized_data')
	if isinstance(batch, (list, tuple)):
		return batch[0], None
	return batch, None


class ModelTrainer:
	"""Trains a sparse VAE with Adam.

	W and log_sigmas are optimized without weight decay (W is regularized by the
	spike-and-slab prior); all other network weights use ``weight_decay``.
	"""

	def __init__(self, model, device=None, verbose=True, print_every=2):
		self.model = model
		self.device = device if device is not None else get_device()
		self.verbose = verbose
		self.print_every = print_every

	def train(self, training_loader, epochs=10, lr=1e-2, weight_decay=0., lambda0_schedule=None, n_samples=None,
			  freeze_sigmas_frac=0.8):
		"""Trains the model and returns a list of per-epoch loss dicts.

		Args:
			lambda0_schedule: optional array of length ``epochs``; ``model.lambda0``
				is set to ``lambda0_schedule[epoch]`` at the start of each epoch
				(see :func:`sparsevae.lambda0_ladder`). If None, lambda0 is fixed.
			n_samples: number of training samples, used to weight the priors on
				W and sigmas against the per-sample likelihood. Defaults to
				``len(training_loader.dataset)``.
			freeze_sigmas_frac: fraction of epochs at the start of training during
				which the noise standard deviations sigmas are held at their initial
				values. Learning sigmas from the start lets features the first factors
				explain get small sigmas (and large weight) while the rest are treated
				as noise, so later factors are never learned. Ignored if the model was
				built with ``sigmas_grad=False`` (sigmas then stay fixed throughout).
		"""
		if n_samples is None:
			try:
				n_samples = len(training_loader.dataset)
			except TypeError:
				raise ValueError("could not determine the training set size from the loader; pass n_samples") from None

		if lambda0_schedule is not None:
			lambda0_schedule = np.asarray(lambda0_schedule, dtype=float)
			if lambda0_schedule.shape != (epochs,):
				raise ValueError(f"lambda0_schedule must have length epochs={epochs}, got shape {lambda0_schedule.shape}")
			if not hasattr(self.model, 'lambda0'):
				raise ValueError("lambda0_schedule given but the model has no lambda0 attribute")

		if not 0. <= freeze_sigmas_frac <= 1.:
			raise ValueError(f"freeze_sigmas_frac must be in [0, 1], got {freeze_sigmas_frac}")
		learn_sigmas = self.model.log_sigmas.requires_grad
		n_frozen = int(round(freeze_sigmas_frac * epochs))

		self.model.to(self.device)

		no_l2 = {id(self.model.W), id(self.model.log_sigmas)}
		params_no_l2 = [p for p in (self.model.W, self.model.log_sigmas) if p.requires_grad]
		params_with_l2 = [p for p in self.model.parameters() if p.requires_grad and id(p) not in no_l2]

		optimizer_l2 = optim.Adam(params_with_l2, lr=lr, weight_decay=weight_decay)
		optimizer_no_l2 = optim.Adam(params_no_l2, lr=lr)

		try:
			history = self._run_epochs(training_loader, epochs, n_samples, lambda0_schedule, learn_sigmas, n_frozen,
									   optimizer_l2, optimizer_no_l2)
		finally:
			self.model.log_sigmas.requires_grad_(learn_sigmas)
		return history

	def _run_epochs(self, training_loader, epochs, n_samples, lambda0_schedule, learn_sigmas, n_frozen,
					optimizer_l2, optimizer_no_l2):
		history = []
		for epoch in range(epochs):
			self.model.train()
			if lambda0_schedule is not None:
				self.model.lambda0 = float(lambda0_schedule[epoch])
			self.model.log_sigmas.requires_grad_(learn_sigmas and epoch >= n_frozen)

			for batch in training_loader:
				optimizer_l2.zero_grad(set_to_none=True)
				optimizer_no_l2.zero_grad(set_to_none=True)

				features, normalized_features = _unpack_batch(batch)
				features = features.to(self.device, dtype=torch.float)
				if normalized_features is not None:
					normalized_features = normalized_features.to(self.device, dtype=torch.float)

				likelihood_loss, kl_loss, w_loss, sigma_loss = self.model.vae_loss(
					features, normalized_x=normalized_features, n_samples=n_samples)
				total_loss = likelihood_loss + kl_loss + sigma_loss + w_loss

				total_loss.backward()
				optimizer_l2.step()
				optimizer_no_l2.step()

				self.model.update_after_step()

			losses = {'total': total_loss.item(), 'likelihood': likelihood_loss.item(), 'kl': kl_loss.item(),
					  'sigma': sigma_loss.item(), 'w': w_loss.item(), 'lambda0': getattr(self.model, 'lambda0', None)}
			history.append(losses)

			if self.verbose and epoch % self.print_every == 0:
				print("Epoch:", epoch, "Total loss:", f"{losses['total']:.3}", "Likelihood:", f"{losses['likelihood']:.3}",
					  "KL loss:", f"{losses['kl']:.3}", "Sigma loss:", f"{losses['sigma']:.3}", "W loss:", f"{losses['w']:.3}")
				sys.stdout.flush()

		return history
