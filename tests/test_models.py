import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from sparsevae import ModelTrainer, SparseVAESpikeSlab, compute_dci, lambda0_ladder, sigma_prior

INPUT_DIM, LATENT_DIM, HIDDEN_DIM = 7, 3, 16


def make_model(**kwargs):
	return SparseVAESpikeSlab(INPUT_DIM, LATENT_DIM, hidden_dim=HIDDEN_DIM, **kwargs)


@pytest.mark.parametrize("decode_vec", [True, False])
@pytest.mark.parametrize("batch_size", [1, 5])
def test_forward_shapes(decode_vec, batch_size):
	model = make_model(decode_vec=decode_vec)
	x = torch.randn(batch_size, INPUT_DIM)
	x_mean, z, z_mean, z_log_var = model(x)
	assert x_mean.shape == (batch_size, INPUT_DIM)
	assert z.shape == z_mean.shape == z_log_var.shape == (batch_size, LATENT_DIM)


def test_vmap_and_loop_decoders_agree():
	torch.manual_seed(0)
	model = make_model(decode_vec=True)
	z = torch.randn(4, LATENT_DIM)

	# copy the depthwise conv weights into the per-column linear layers
	model.column_means = torch.nn.ModuleList([torch.nn.Linear(HIDDEN_DIM, 1) for _ in range(INPUT_DIM)])
	with torch.no_grad():
		for j, layer in enumerate(model.column_means):
			layer.weight.copy_(model.generator_out.weight[j])
			layer.bias.copy_(model.generator_out.bias[j:j + 1])

	assert torch.allclose(model.decode_vmap(z), model.decode_loop(z), atol=1e-5)


@pytest.mark.parametrize("loss_type", ["mse", "binary", "categorical"])
def test_losses_finite(loss_type):
	model = make_model(loss_type=loss_type)
	if loss_type == "mse":
		x = torch.randn(8, INPUT_DIM)
	else:
		x = (torch.rand(8, INPUT_DIM) > 0.5).float()
	for loss in model.vae_loss(x):
		assert torch.isfinite(loss).all()


def test_unknown_argument_raises():
	with pytest.raises(TypeError):
		make_model(row_normalize=True)


def test_spike_slab_update_in_range():
	model = make_model()
	model.update_spike_slab()
	assert ((model.p_star >= 0) & (model.p_star <= 1)).all()
	assert (model.thetas > 0).all()


def test_trainer_runs_and_updates_W():
	torch.manual_seed(0)
	x = np.random.randn(50, INPUT_DIM)
	sig_scale, sig_init = sigma_prior(x)
	model = make_model(sigma_prior_scale=sig_scale, sigmas_init=sig_init)
	W_before = model.W.detach().clone()

	loader = DataLoader(torch.tensor(x, dtype=torch.float), batch_size=10, shuffle=True)
	history = ModelTrainer(model, device=torch.device("cpu"), verbose=False).train(loader, epochs=2)

	assert len(history) == 2
	assert np.isfinite(history[-1]["total"])
	assert not torch.equal(W_before, model.W.detach())


def test_trainer_accepts_dict_batches():
	x = torch.rand(20, INPUT_DIM)
	data = [{"data": x[i], "normalized_data": x[i] / x[i].sum()} for i in range(20)]
	loader = DataLoader(data, batch_size=5)
	model = make_model(loss_type="categorical")
	ModelTrainer(model, device=torch.device("cpu"), verbose=False).train(loader, epochs=1)


def test_compute_dci_keys():
	rng = np.random.default_rng(0)
	codes = rng.normal(size=(2, 40))
	factors = codes + 0.1 * rng.normal(size=(2, 40))
	scores = compute_dci(codes, factors, codes, factors)
	assert set(scores) == {"informativeness_train", "informativeness_test", "disentanglement", "completeness"}


def test_lambda0_ladder_matches_multistudy_rungs():
	# multi-study code with epochs=100, freeze_lam=0.04, lambda0_max=15 -> 4 x 1, 25 x 10, 71 x 15
	sched = lambda0_ladder(100, lambda0=15., lambda1=0.1, lambda0_start=1.)
	assert sched.shape == (100,)
	assert (sched[:4] == 1).all()
	assert np.allclose(sched[4:29], 10)
	assert (sched[29:] == 15).all()


def test_lambda0_ladder_short_run():
	sched = lambda0_ladder(3, lambda0=10., lambda1=0.5, warmup_frac=0.5, middle_frac=0.9)
	assert sched.shape == (3,)
	assert sched[0] == 0.5 and sched[-1] == pytest.approx(20 / 3)


def test_lambda0_ladder_starts_at_lambda1_by_default():
	sched = lambda0_ladder(100, lambda0=30., lambda1=0.1)
	assert (sched[:4] == 0.1).all() and np.allclose(sched[4:29], 20) and (sched[29:] == 30).all()


def test_trainer_applies_lambda0_schedule():
	x = torch.randn(20, INPUT_DIM)
	loader = DataLoader(x, batch_size=10)
	model = make_model(lambda0=10.)
	sched = lambda0_ladder(4, lambda0=10., lambda1=0.1, warmup_frac=0.25, middle_frac=0.25)
	history = ModelTrainer(model, device=torch.device("cpu"), verbose=False).train(loader, epochs=4, lambda0_schedule=sched)
	assert [h["lambda0"] for h in history] == pytest.approx(list(sched))
	assert model.lambda0 == 10.


def test_trainer_rejects_wrong_length_schedule():
	loader = DataLoader(torch.randn(10, INPUT_DIM), batch_size=5)
	with pytest.raises(ValueError):
		ModelTrainer(make_model(), device=torch.device("cpu"), verbose=False).train(loader, epochs=3, lambda0_schedule=[1., 2.])


def test_p_star_matches_spike_slab_lasso_formula():
	model = make_model(lambda0=10., lambda1=1.)
	with torch.no_grad():
		model.W.copy_(torch.linspace(-1, 1, INPUT_DIM * LATENT_DIM).reshape(INPUT_DIM, LATENT_DIM))
		model.thetas.fill_(0.5)
	model.update_spike_slab()

	w = np.abs(model.W.detach().numpy())
	laplace = lambda lam: lam / 2 * np.exp(-lam * w)
	expected = 0.5 * laplace(1.) / (0.5 * laplace(1.) + 0.5 * laplace(10.))
	assert np.allclose(model.p_star.numpy(), expected, atol=1e-6)


def test_p_star_threshold():
	# with theta = 0.5, p_star = 0.5 exactly at |w| = log(lambda0 / lambda1) / (lambda0 - lambda1)
	model = SparseVAESpikeSlab(1, 1, hidden_dim=4, lambda0=10., lambda1=1.)
	with torch.no_grad():
		model.W.fill_(np.log(10.) / 9.)
		model.thetas.fill_(0.5)
	model.update_spike_slab()
	assert model.p_star.item() == pytest.approx(0.5, abs=1e-5)



def test_thetas_floored_above_zero():
	# a weak-but-distinct spike with W ~ 1 drives p_star (and hence thetas) to 0
	model = make_model(lambda0=1., lambda1=0.1)
	with torch.no_grad():
		model.W.fill_(1.)
	for _ in range(100):
		model.update_spike_slab()
	assert (model.thetas >= 1e-10).all()


def test_mse_reconstruction_is_gaussian_nll():
	model = make_model()
	with torch.no_grad():
		model.log_sigmas.copy_(torch.linspace(-1, 1, INPUT_DIM))
	x, x_pred = torch.randn(6, INPUT_DIM), torch.randn(6, INPUT_DIM)
	normal = torch.distributions.Normal(x_pred, torch.exp(model.log_sigmas))
	expected = -normal.log_prob(x).sum(1).mean() - INPUT_DIM * 0.5 * np.log(2 * np.pi)
	assert model.reconstruction_loss(x_pred, x).item() == pytest.approx(expected.item(), rel=1e-5)


def test_kl_sums_over_latent_dims():
	model = make_model()
	mu, log_var = torch.randn(6, LATENT_DIM), torch.randn(6, LATENT_DIM)
	q = torch.distributions.Normal(mu, torch.exp(0.5 * log_var))
	expected = torch.distributions.kl_divergence(q, torch.distributions.Normal(0., 1.)).sum(1).mean()
	assert model.kld_z(None, mu, log_var).item() == pytest.approx(expected.item(), rel=1e-5)


def test_prior_terms_divided_by_dataset_size_not_batch_size():
	model = make_model()
	_, _, w_small, s_small = model.vae_loss(torch.randn(5, INPUT_DIM), n_samples=1000)
	_, _, w_large, s_large = model.vae_loss(torch.randn(50, INPUT_DIM), n_samples=1000)
	assert w_small.item() == pytest.approx(w_large.item())
	assert s_small.item() == pytest.approx(s_large.item())
	_, _, w_half, _ = model.vae_loss(torch.randn(5, INPUT_DIM), n_samples=500)
	assert w_half.item() == pytest.approx(2 * w_small.item())


def test_trainer_passes_dataset_size():
	seen = []

	class Spy(SparseVAESpikeSlab):
		def vae_loss(self, x, normalized_x=None, n_samples=None):
			seen.append(n_samples)
			return super().vae_loss(x, normalized_x, n_samples)

	model = Spy(INPUT_DIM, LATENT_DIM, hidden_dim=HIDDEN_DIM)
	loader = DataLoader(torch.randn(30, INPUT_DIM), batch_size=7)
	ModelTrainer(model, device=torch.device("cpu"), verbose=False).train(loader, epochs=1)
	assert seen and set(seen) == {30}


def test_sigma_prior_initializes_at_fraction_of_feature_sd():
	x = np.random.default_rng(0).normal(size=(200, INPUT_DIM)) * np.arange(1, INPUT_DIM + 1)
	_, sigmas_init = sigma_prior(x)
	assert np.allclose(sigmas_init, 0.8 * x.std(0))


@pytest.mark.parametrize("frac, sigmas_grad, should_change", [(1.0, True, False), (0.0, True, True),
															  (0.5, True, True), (0.0, False, False)])
def test_trainer_sigma_freezing(frac, sigmas_grad, should_change):
	torch.manual_seed(0)
	model = make_model(sigmas_grad=sigmas_grad)
	before = model.log_sigmas.detach().clone()
	loader = DataLoader(torch.randn(20, INPUT_DIM), batch_size=10)
	ModelTrainer(model, device=torch.device("cpu"), verbose=False).train(loader, epochs=4, freeze_sigmas_frac=frac)
	assert (not torch.equal(before, model.log_sigmas.detach())) == should_change
	assert model.log_sigmas.requires_grad == sigmas_grad
