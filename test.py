import jax
import jax.numpy as jnp
import pickle
from alpaca import ALPaCA, DefaultFeatureMapping
from data_generation import generate_sinusoid_data
import matplotlib.pyplot as plt
from gp import GaussianProcess, rbf

# Generate data
M = 1  # Total number of trajectories
tau = 50  # Number of time steps per trajectory
amplitude_range = (0.1, 5.0)
phase_range = (0.0, jnp.pi)
time_range = (-5, 5)
key = jax.random.key(7)

Dx, Dy = generate_sinusoid_data(M, tau, key, amplitude_range, phase_range, time_range)
Dx = Dx.squeeze(0)
Dy = Dy.squeeze(0)

# Load the model
n_x = Dx.shape[-1]  # Input dimension (should be 1)
n_y = Dy.shape[-1]  # Output dimension (should be 1)
n_phi = 32  # Dimension of feature mapping output

# Define noise covariance
Sigma_eps = jnp.eye(n_y) * 0.02  # Assumed noise covariance

# Instantiate the feature mapping module
phi = DefaultFeatureMapping(n_phi=n_phi)

# Load models
alpaca = ALPaCA(phi=phi, Sigma_eps=Sigma_eps, n_y=n_y, n_x=n_x, n_phi=n_phi)
gp = GaussianProcess(rbf, 1.0, Sigma_eps[0, 0])

# Load parameters
with open("data/model_params.pkl", "rb") as f:
    params = pickle.load(f)
with open("data/model_params_no_meta.pkl", "rb") as f:
    params_no_meta = pickle.load(f)

# Run the online update
context_size = 3
context_indices = jnp.sort(
    jax.random.randint(key, (context_size,), 0 + tau // 10, tau - tau // 10)
)
online_Dx = Dx[context_indices]
online_Dy = Dy[context_indices]
params = alpaca.online_update(params, (online_Dx, online_Dy))
params_no_meta = alpaca.online_update(params_no_meta, (online_Dx, online_Dy))
gp.fit(online_Dx, online_Dy)

# Predict
Ybar_meta, Sigma_meta = alpaca.predict(params, Dx)
Ybar_no_meta, Sigma_no_meta = alpaca.predict(params_no_meta, Dx)
mu_star, cov_star = gp.predict(Dx)

# Plot side-by-side subplots to show meta vs non-meta predictions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# With Meta-Learning
axes[0].plot(Dx, Dy, label="True", color="blue")
axes[0].plot(Dx, Ybar_meta, label="Predicted (Meta)", color="orange")
axes[0].fill_between(
    Dx[:, 0],
    Ybar_meta[:, 0] - 2 * jnp.sqrt(Sigma_meta).flatten(),
    Ybar_meta[:, 0] + 2 * jnp.sqrt(Sigma_meta).flatten(),
    color="orange",
    alpha=0.3,
)
axes[0].plot(online_Dx, online_Dy, "x", color="black", markersize=10, markeredgewidth=2)
axes[0].set_xlim(-5, 5)
axes[0].set_ylim(-5, 5)
axes[0].set_title("ALPaCA")

# Without Meta-Learning
axes[1].plot(Dx, Dy, label="True", color="blue")
axes[1].plot(Dx, Ybar_no_meta, label="Predicted (No Meta)", color="orange")
axes[1].fill_between(
    Dx[:, 0],
    Ybar_no_meta[:, 0] - 2 * jnp.sqrt(Sigma_meta).flatten(),
    Ybar_no_meta[:, 0] + 2 * jnp.sqrt(Sigma_meta).flatten(),
    color="orange",
    alpha=0.3,
)
axes[1].plot(online_Dx, online_Dy, "x", color="black", markersize=10, markeredgewidth=2)
axes[1].set_xlim(-5, 5)
axes[1].set_ylim(-5, 5)
axes[1].set_title("ALPaCA (no meta)")

# GP
axes[2].plot(Dx, Dy, label="True", color="blue")
axes[2].plot(Dx, mu_star, label="Predicted (GP)", color="orange")
axes[2].fill_between(
    Dx[:, 0],
    mu_star.flatten() - 2 * jnp.sqrt(jnp.diag(cov_star)),
    mu_star.flatten() + 2 * jnp.sqrt(jnp.diag(cov_star)),
    color="orange",
    alpha=0.3,
)
axes[2].plot(online_Dx, online_Dy, "x", color="black", markersize=10, markeredgewidth=2)
axes[2].set_xlim(-5, 5)
axes[2].set_ylim(-5, 5)
axes[2].set_title("Gaussian Process")

plt.tight_layout()
plt.savefig("figures/predictions.png")
