import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class BaseSparseVAE(nn.Module):
	"""VAE whose decoder uses a sparse mask W (input_dim x latent_dim).

	Each observed feature j is generated from z masked by row j of W, so that each
	feature depends on only a subset of the latent factors.

	The model is device-agnostic: move it with ``model.to(device)``.
	"""

	def __init__(self, input_dim, latent_dim, hidden_dim=300,
				 W_init=None, sigmas_init=None, sigma_prior_scale=1., sigma_prior_df=3,
				 loss_type='mse', decode_vec=True, sigmas_grad=True):

		super().__init__()

		if loss_type not in ('mse', 'binary', 'categorical'):
			raise ValueError(f"loss_type must be 'mse', 'binary' or 'categorical', got {loss_type!r}")

		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.latent_dim = latent_dim

		self.sigma_prior_df = sigma_prior_df
		self.sigma_prior_scale = sigma_prior_scale
		self.loss_type = loss_type
		self.decode_vec = decode_vec

		if W_init is not None:
			self.W = nn.Parameter(torch.tensor(W_init, dtype=torch.float))
		else:
			W_np = np.random.normal(loc=1, scale=0.01, size=(self.input_dim, self.latent_dim))
			self.W = nn.Parameter(torch.tensor(W_np, dtype=torch.float))

		if sigmas_init is not None:
			self.log_sigmas = nn.Parameter(torch.log(torch.tensor(sigmas_init, dtype=torch.float)), requires_grad=sigmas_grad)
		else:
			self.log_sigmas = nn.Parameter(torch.zeros(input_dim), requires_grad=sigmas_grad)

		self.q_z = nn.Sequential(nn.Linear(input_dim, hidden_dim),
								 nn.ReLU(),
								 nn.Linear(hidden_dim, hidden_dim),
								 nn.ReLU())

		self.z_mean = nn.Linear(hidden_dim, latent_dim)
		self.z_log_var = nn.Linear(hidden_dim, latent_dim)

		self.generator = nn.Sequential(nn.Linear(latent_dim, hidden_dim, bias=False),
									   nn.ReLU(),
									   nn.Linear(hidden_dim, hidden_dim),
									   nn.ReLU())

		if decode_vec:
			# depthwise convolution: each feature (channel) has its own output kernel.
			self.generator_out = nn.Conv1d(in_channels=input_dim, groups=input_dim, out_channels=input_dim, kernel_size=hidden_dim)
		else:
			self.column_means = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(input_dim)])

	def get_generator_mask(self):
		return self.W

	def encode(self, x):
		q_z = self.q_z(x)
		z_mean = self.z_mean(q_z)
		z_log_var = self.z_log_var(q_z)
		return z_mean, z_log_var

	def reparameterize(self, mean, log_var):
		std = torch.exp(0.5 * log_var)
		eps = torch.randn_like(std)
		return mean + (eps * std)

	def column_decode(self, mask_j, z):
		return self.generator(torch.mul(z, mask_j))

	def decode_vmap(self, z):
		mask = self.get_generator_mask()

		# (input_dim, batch, hidden) -> (batch, input_dim, hidden)
		x_vmap = torch.vmap(self.column_decode, in_dims=(0, None))(mask, z)
		x_vmap = torch.permute(x_vmap, (1, 0, 2))

		# (batch, input_dim, 1) -> (batch, input_dim)
		x_recon = self.generator_out(x_vmap).squeeze(-1)

		if self.loss_type == 'binary':
			x_recon = torch.sigmoid(x_recon)
		elif self.loss_type == 'categorical':
			x_recon = F.softmax(x_recon, dim=-1)

		return x_recon

	def decode_loop(self, z):
		x_reconstructed = torch.zeros(z.shape[0], self.input_dim, device=z.device)
		mask = self.get_generator_mask()

		for j in range(self.input_dim):
			masked_input = torch.mul(z, mask[j, :])
			x_reconstructed[:, j] = self.column_means[j](self.generator(masked_input)).squeeze(-1)

		if self.loss_type == 'binary':
			x_reconstructed = torch.sigmoid(x_reconstructed)
		elif self.loss_type == 'categorical':
			x_reconstructed = F.softmax(x_reconstructed, dim=-1)

		return x_reconstructed

	def decode(self, z):
		if self.decode_vec:
			return self.decode_vmap(z)
		return self.decode_loop(z)

	def kld_z(self, z, mu, log_var):
		"""KL(q(z|x) || N(0, I)), summed over latent dimensions and averaged over the batch."""
		return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()

	def reconstruction_loss(self, x_pred, x):
		"""Negative log-likelihood -log p(x | z), summed over features and averaged over the batch.

		For 'mse' this is the Gaussian NLL with per-feature standard deviations
		sigma_j (up to the constant 0.5 * log(2 pi) per feature).
		"""
		if self.loss_type == 'mse':
			log_sigmas = self.log_sigmas
			nll = 0.5 * ((x - x_pred) * torch.exp(-log_sigmas)) ** 2 + log_sigmas
			return nll.sum(1).mean()

		if self.loss_type == 'binary':
			log_prob = x * torch.log(x_pred + 1e-6) + (1 - x) * torch.log(1 - x_pred + 1e-6)
			return -log_prob.sum(1).mean()

		# categorical
		log_pred = torch.log(x_pred + 1e-6)
		return -(log_pred * x).sum(1).mean()

	def mask_loss(self, n_samples):
		raise NotImplementedError

	def sigma_loss(self, n_samples):
		"""Negative log scaled-inverse-chi^2(df, scale) prior on sigma_j^2, divided by the dataset size."""
		df, scale = self.sigma_prior_df, self.sigma_prior_scale
		neg_log_prior = (df + 2) * self.log_sigmas.sum() + 0.5 * df * scale * torch.exp(-2 * self.log_sigmas).sum()
		return neg_log_prior / n_samples

	def forward(self, x):
		z_mean, z_log_var = self.encode(x)
		z = self.reparameterize(z_mean, z_log_var)
		x_mean = self.decode(z)
		return x_mean, z, z_mean, z_log_var

	def vae_loss(self, x, normalized_x=None, n_samples=None):
		"""Returns (reconstruction_loss, kl_loss, w_loss, sigma_loss).

		The sum of the four terms is an unbiased minibatch estimate of the negative
		log posterior divided by the dataset size: the per-sample terms
		(reconstruction, KL) are averaged over the batch, and the priors on the
		global parameters (W, sigmas) are divided by ``n_samples``.

		Args:
			x: batch of data.
			normalized_x: if given, used as the encoder input, while the
				reconstruction loss is computed against ``x``.
			n_samples: total number of samples in the training set. If None,
				``x`` is treated as the full dataset.
		"""
		x_in = normalized_x if normalized_x is not None else x
		x_mean, z, z_mean, z_log_var = self.forward(x_in)

		if n_samples is None:
			n_samples = x.shape[0]

		kl_loss = self.kld_z(z, z_mean, z_log_var)
		w_loss = self.mask_loss(n_samples)
		reconstruction_loss = self.reconstruction_loss(x_mean, x)

		if self.loss_type == 'mse':
			sigma_loss = self.sigma_loss(n_samples)
		else:
			sigma_loss = torch.zeros((), device=x.device)

		return reconstruction_loss, kl_loss, w_loss, sigma_loss

	def update_after_step(self):
		"""Hook called by the trainer after each optimizer step."""
		pass


class SparseVAESpikeSlab(BaseSparseVAE):
	"""Sparse VAE with a spike-and-slab lasso prior on the decoder mask W.

	Args:
		lambda0: spike penalty (large -> stronger shrinkage to zero).
		lambda1: slab penalty.
		a, b: Beta prior hyperparameters on the inclusion probabilities theta_k
			(b defaults to input_dim).
		All other arguments are passed to :class:`BaseSparseVAE`.
	"""

	def __init__(self, input_dim, latent_dim, lambda0=15., lambda1=0.1, a=1., b=None, **kwargs):
		super().__init__(input_dim, latent_dim, **kwargs)
		self.lambda0 = lambda0
		self.lambda1 = lambda1
		self.a = a
		self.b = b if b is not None else input_dim

		self.p_star = nn.Parameter(0.5 * torch.ones(self.input_dim, self.latent_dim), requires_grad=False)
		self.thetas = nn.Parameter(torch.rand(self.latent_dim), requires_grad=False)

	def mask_loss(self, n_samples):
		"""Expected negative log spike-and-slab lasso prior on W (terms depending on W), divided by the dataset size."""
		loss = (self.lambda1 * self.p_star + self.lambda0 * (1 - self.p_star)) * self.W.abs()
		return loss.sum() / n_samples

	@torch.no_grad()
	def update_spike_slab(self):
		"""Closed-form update of the inclusion probabilities p_star and weights thetas.

		p_star is the conditional probability that each entry of W comes from the slab,
		under Laplace densities psi_lambda(w) = (lambda / 2) exp(-lambda |w|).

		thetas are floored at 1e-10 so they cannot get stuck at exactly 0.
		"""
		W_abs = self.W.abs()
		slab = self.thetas * self.lambda1 * torch.exp(-self.lambda1 * W_abs)
		spike = (1 - self.thetas) * self.lambda0 * torch.exp(-self.lambda0 * W_abs)
		p_star = slab / (slab + spike)

		thetas = (p_star.sum(0) + self.a - 1) / (self.a + self.b + self.input_dim - 2)
		thetas = thetas.clamp_min(1e-10)

		self.p_star.copy_(p_star)
		self.thetas.copy_(thetas)

	def update_after_step(self):
		self.update_spike_slab()
