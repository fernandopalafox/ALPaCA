import jax
import jax.numpy as jnp
import pickle
from alpaca import ALPaCA, DefaultFeatureMapping
from data_generation import generate_sinusoid_data
import matplotlib.pyplot as plt

# Generate data
M = 1  # Total number of trajectories
tau = 20  # Number of time steps per trajectory
amplitude_range = (0.1, 5.0)
phase_range = (0.0, jnp.pi)
time_range = (-5, 5)
key = jax.random.key(0)

Dx, Dy = generate_sinusoid_data(
    M, tau, key, amplitude_range, phase_range, time_range
)
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

# Instantiate the ALPaCA model
model = ALPaCA(phi=phi, Sigma_eps=Sigma_eps, n_y=n_y, n_x=n_x, n_phi=n_phi)
with open("data/model_params.pkl", "rb") as f:
    params = pickle.load(f)

# Run the online update
context_size = 3
context_indices = jnp.sort(
    jax.random.randint(key, (context_size,), 0 + tau // 10, tau - tau // 10)
)
online_Dx = Dx[context_indices]
online_Dy = Dy[context_indices]
params = model.online_update(params, (online_Dx, online_Dy))

# No meta learning case
with open("data/model_params_no_meta.pkl", "rb") as f:
    params_no_meta = pickle.load(f)

# Predict
Ybar_meta, Sigma_meta = model.predict(params, Dx)
Ybar_no_meta, Sigma_no_meta = model.predict(params_no_meta, Dx)

# Plot side-by-side subplots to show meta vs non-meta predictions
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# With Meta-Learning
axes[0].plot(Dx, Dy, label="True", color="blue")
axes[0].plot(Dx, Ybar_meta, label="Predicted (Meta)", color="orange")
axes[0].fill_between(
    Dx[:,0],
    Ybar_meta[:,0] - 2 * jnp.sqrt(Sigma_meta).flatten(),
    Ybar_meta[:,0] + 2 * jnp.sqrt(Sigma_meta).flatten(),
    color="orange",
    alpha=0.3,
)
axes[0].plot(online_Dx, online_Dy, "x", color="black", markersize=10, markeredgewidth=2)
axes[0].legend()
axes[0].set_xlim(-5, 5)
axes[0].set_ylim(-5, 5)
axes[0].set_title("With Meta-Learning")

# Without Meta-Learning
axes[1].plot(Dx, Dy, label="True", color="blue")
axes[1].plot(Dx, Ybar_no_meta, label="Predicted (No Meta)", color="red")
axes[1].fill_between(
    Dx[:,0],
    Ybar_no_meta[:,0] - 2 * jnp.sqrt(Sigma_meta).flatten(),
    Ybar_no_meta[:,0] + 2 * jnp.sqrt(Sigma_meta).flatten(),
    color="red",
    alpha=0.3,
)
axes[1].plot(online_Dx, online_Dy, "x", color="black", markersize=10, markeredgewidth=2)
axes[1].legend()
axes[1].set_xlim(-5, 5)
axes[1].set_ylim(-5, 5)
axes[1].set_title("Without Meta-Learning")

plt.savefig("figures/predictions.png")
