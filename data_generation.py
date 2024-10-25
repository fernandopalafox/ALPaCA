import jax
import jax.numpy as jnp


def generate_sinusoid_data(
    J: int,
    tau: int,
    key: jax.random.PRNGKey,
    amplitude_range=(0.1, 5.0),
    phase_range=(0.0, jnp.pi),
    time_range=(-5, 5),
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generates a dataset of sinusoidal trajectories with specified amplitude and phase ranges.

    Args:
        M (int): Total number of trajectories.
        tau (int): Number of time steps per trajectory.
        key (jax.random.PRNGKey): Random key for reproducibility.
        amplitude_range (tuple): Min and max values for amplitude sampling.
        phase_range (tuple): Min and max values for phase sampling.

    Returns:
        tuple: Dxs and Dys, where:
            - Dxs (jnp.ndarray): Input trajectories of shape (J, tau, 1).
            - Dys (jnp.ndarray): Output trajectories of shape (J, tau, 1).
    """
    # Split the random key
    key, subkey1, subkey2 = jax.random.split(key, 3)

    # Sample amplitudes and phases uniformly within the specified ranges
    amplitudes = jax.random.uniform(
        subkey1, shape=(J,), minval=amplitude_range[0], maxval=amplitude_range[1]
    )
    phases = jax.random.uniform(
        subkey2, shape=(J,), minval=phase_range[0], maxval=phase_range[1]
    )

    # Generate time steps
    t = jnp.linspace(time_range[0], time_range[1], tau)

    # Vectorized computation of trajectories
    def compute_trajectory(A, phi):
        x_j = t.reshape(-1, 1)  # Shape: (tau, 1)
        y_j = A * jnp.sin(x_j + phi)
        return x_j, y_j

    # Apply the function across all amplitudes and phases
    xys = jax.vmap(compute_trajectory)(amplitudes, phases)
    Dxs = xys[0]  # Shape: (J, tau, 1)
    Dys = xys[1]  # Shape: (J, tau, 1)

    return Dxs, Dys


# Example usage
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    J = 100  # Number of trajectories
    tau = 50  # Number of time steps per trajectory

    # You can specify custom ranges if needed
    amplitude_range = (0.1, 5.0)
    phase_range = (0.0, jnp.pi)

    Dxs, Dys = generate_sinusoid_data(J, tau, key, amplitude_range, phase_range)

    # Print the shapes of the generated data
    print(Dxs.shape, Dys.shape)
