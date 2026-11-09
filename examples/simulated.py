"""Fit a Sparse VAE to the simulated dataset from the paper (7 features, 2 true factors).

Run with:  python examples/simulated.py
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from sparsevae import ModelTrainer, SparseVAESpikeSlab, compute_dci, sigma_prior


def simulate(N=1000, sigma_true=0.5, seed=0):
	rng = np.random.default_rng(seed)
	z = rng.standard_normal((N, 2))
	x = np.column_stack([
		1 * z[:, 0],
		2 * z[:, 0],
		3 * z[:, 0] ** 2,
		4 * z[:, 1],
		5 * z[:, 1],
		6 * np.sin(z[:, 1]),
		7 * z[:, 0] * z[:, 1],
	])
	x = x + sigma_true * rng.standard_normal(x.shape)
	return x, z


def main():
	torch.manual_seed(0)
	np.random.seed(0)

	x, z_true = simulate()
	sig_scale, sig_init = sigma_prior(x)

	epochs, lambda0, lambda1 = 200, 15., 0.1
	model = SparseVAESpikeSlab(input_dim=x.shape[1], latent_dim=5, hidden_dim=50,
							   lambda0=lambda0, lambda1=lambda1,
							   sigma_prior_scale=sig_scale, sigmas_init=sig_init)

	loader = DataLoader(torch.tensor(x, dtype=torch.float), batch_size=100, shuffle=True)
	trainer = ModelTrainer(model)
	trainer.train(loader, epochs=epochs, lr=0.01)

	np.set_printoptions(precision=2, suppress=True)
	print("\nEstimated mask W (features x latent factors):")
	print(model.W.detach().cpu().numpy())
	print("\nInclusion probabilities p_star >= 0.5:")
	print((model.p_star.cpu().numpy() >= 0.5).astype(int))

	model.eval()
	with torch.no_grad():
		z_mean, _ = model.encode(torch.tensor(x, dtype=torch.float, device=trainer.device))
	codes = z_mean.cpu().numpy().T
	scores = compute_dci(codes, z_true.T, codes, z_true.T)
	print("\nDCI disentanglement:", round(scores["disentanglement"], 3))


if __name__ == "__main__":
	main()
