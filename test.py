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
gp_scale = 1.0
gp = GaussianProcess(rbf, gp_scale, Sigma_eps[0, 0])

# Load parameters
with open("data/model_params.pkl", "rb") as f:
    params_meta = pickle.load(f)
with open("data/model_params_no_meta.pkl", "rb") as f:
    params_no_meta = pickle.load(f)

# Run the online update
context_sizes = [0, 1, 2, 3, 5]
context_indices_all = jnp.sort(
    jax.random.choice(
        key,
        a=jnp.arange(0 + tau // 10, tau - tau // 10),
        shape=(context_sizes[-1],),
        replace=False,
    )
)

Ybar_metas = []
Sigma_metas = []
Ybar_no_metas = []
Sigma_no_metas = []
mu_stars = []
cov_stars = []
online_Dxs = []
online_Dys = []
for context_size in context_sizes:
    # Online update
    context_indices = context_indices_all[:context_size]
    online_Dx = Dx[context_indices]
    online_Dy = Dy[context_indices]
    params_meta_fit = alpaca.online_update(params_meta, (online_Dx, online_Dy))
    params_no_meta_fit = alpaca.online_update(params_no_meta, (online_Dx, online_Dy))
    gp.fit(online_Dx, online_Dy)

    # Predict
    Ybar_meta, Sigma_meta = alpaca.predict(params_meta, Dx)
    Ybar_no_meta, Sigma_no_meta = alpaca.predict(params_no_meta, Dx)
    mu_star, cov_star = gp.predict(Dx)

    # Store data
    Ybar_metas.append(Ybar_meta)
    Sigma_metas.append(Sigma_meta)
    Ybar_no_metas.append(Ybar_no_meta)
    Sigma_no_metas.append(Sigma_no_meta)
    mu_stars.append(mu_star)
    cov_stars.append(cov_star)
    online_Dxs.append(online_Dx)
    online_Dys.append(online_Dy)

    # Reset GP
    gp = GaussianProcess(rbf, gp_scale, Sigma_eps[0, 0])

# Plot side-by-side subplots to show meta vs non-meta predictions
num_rows = len(context_sizes)
fig, axes = plt.subplots(num_rows, 3, figsize=(15, 1 * num_rows))
if num_rows == 1:
    axes = axes[jnp.newaxis, :]

for i, context_size in enumerate(context_sizes):
    Ybar_meta = Ybar_metas[i]
    Sigma_meta = Sigma_metas[i]
    Ybar_no_meta = Ybar_no_metas[i]
    Sigma_no_meta = Sigma_no_metas[i]
    mu_star = mu_stars[i]
    cov_star = cov_stars[i]
    online_Dx = online_Dxs[i]
    online_Dy = online_Dys[i]

    # With Meta-Learning
    axes[i, 0].plot(Dx, Dy, label="True", color="blue")
    axes[i, 0].plot(Dx, Ybar_meta, label="Predicted (Meta)", color="orange")
    axes[i, 0].fill_between(
        Dx[:, 0],
        Ybar_meta[:, 0] - 2 * jnp.sqrt(Sigma_meta).flatten(),
        Ybar_meta[:, 0] + 2 * jnp.sqrt(Sigma_meta).flatten(),
        color="orange",
        alpha=0.3,
    )
    axes[i, 0].plot(
        online_Dx, online_Dy, "x", color="black", markersize=10, markeredgewidth=2
    )
    axes[i, 0].set_xlim(-5, 5)
    axes[i, 0].set_ylim(-5, 5)
    axes[i, 0].set_yticks([-5, 0, 5])

    # Without Meta-Learning
    axes[i, 1].plot(Dx, Dy, label="True", color="blue")
    axes[i, 1].plot(Dx, Ybar_no_meta, label="Predicted (No Meta)", color="orange")
    axes[i, 1].fill_between(
        Dx[:, 0],
        Ybar_no_meta[:, 0] - 2 * jnp.sqrt(Sigma_no_meta).flatten(),
        Ybar_no_meta[:, 0] + 2 * jnp.sqrt(Sigma_no_meta).flatten(),
        color="orange",
        alpha=0.3,
    )
    axes[i, 1].plot(
        online_Dx, online_Dy, "x", color="black", markersize=10, markeredgewidth=2
    )
    axes[i, 1].set_xlim(-5, 5)
    axes[i, 1].set_ylim(-5, 5)
    axes[i, 1].set_yticks([])

    # GP
    axes[i, 2].plot(Dx, Dy, label="True", color="blue")
    axes[i, 2].plot(Dx, mu_star, label="Predicted (GP)", color="orange")
    axes[i, 2].fill_between(
        Dx[:, 0],
        mu_star.flatten() - 2 * jnp.sqrt(jnp.diag(cov_star)),
        mu_star.flatten() + 2 * jnp.sqrt(jnp.diag(cov_star)),
        color="orange",
        alpha=0.3,
    )
    axes[i, 2].plot(
        online_Dx, online_Dy, "x", color="black", markersize=10, markeredgewidth=2
    )
    axes[i, 2].set_xlim(-5, 5)
    axes[i, 2].set_ylim(-5, 5)
    axes[i, 2].set_yticks([])

    # Cleanup
    if not context_size == context_sizes[-1]:
        axes[i, 0].set_xticks([])
        axes[i, 1].set_xticks([])
        axes[i, 2].set_xticks([])
    else:
        axes[i, 0].set_xticks([-5, 0, 5])
        axes[i, 1].set_xticks([-5, 0, 5])
        axes[i, 2].set_xticks([-5, 0, 5])

axes[0, 0].set_title(f"ALPaCA")
axes[0, 1].set_title(f"ALPaCA (no meta)")
axes[0, 2].set_title(f"GPR")

plt.tight_layout()
plt.savefig("figures/predictions.png")
