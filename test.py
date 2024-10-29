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

# Predict
Ybar, Sigma = model.predict(params, Dx)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(Dx[:, 0], Dy[:, 0], label="True")
plt.plot(Dx[:, 0], Ybar[:, 0], label="Predicted")
plt.fill_between(
    Dx[:, 0],
    Ybar[:, 0] - 2 * jnp.sqrt(Sigma.squeeze()),
    Ybar[:, 0] + 2 * jnp.sqrt(Sigma.squeeze()),
    alpha=0.3,
)
plt.plot(online_Dx[:,0], online_Dy[:, 0], "x", color='black', markersize=10, markeredgewidth=2)
plt.legend()
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.savefig("figures/test.png")
