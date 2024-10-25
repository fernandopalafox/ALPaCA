import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from data_generation import generate_sinusoid_data
from alpaca import ALPaCA


# Define the feature mapping phi
class FeatureMapping(nn.Module):
    n_phi: int  # Dimension of the feature mapping output

    @nn.compact
    def __call__(self, x):
        # MLP with two hidden layers and 128 units each
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.n_phi)(x)
        return x


def main():
    # Set random seed
    key = jax.random.key(0)

    # Generate data
    M = 100  # Total number of trajectories
    tau = 50  # Number of time steps per trajectory
    amplitude_range = (0.1, 5.0)
    phase_range = (0.0, jnp.pi)
    time_range = (-5, 5)

    Dxs, Dys = generate_sinusoid_data(
        M, tau, key, amplitude_range, phase_range, time_range
    )

    # Define model parameters
    n_x = Dxs.shape[-1]  # Input dimension (should be 1)
    n_y = Dys.shape[-1]  # Output dimension (should be 1)
    n_phi = 16  # Dimension of feature mapping output

    # Define noise covariance
    Sigma_eps = jnp.eye(n_y) * 0.05  # Assumed noise covariance

    # Instantiate the feature mapping module
    phi = FeatureMapping(n_phi=n_phi)

    # Instantiate the ALPaCA model
    model = ALPaCA(phi=phi, Sigma_eps=Sigma_eps, n_y=n_y, n_x=n_x, n_phi=n_phi)

    # Initialize model parameters
    @jax.jit
    def init_model_params(rng_key):
        x_init = jnp.ones((1, n_x))
        params = model.init(rng_key, x_init)
        return params

    params = init_model_params(key)

    # Define optimizer
    learning_rate = 1e-3
    optimizer = optax.adam(learning_rate)

    # Create training state
    class TrainState(train_state.TrainState):
        pass

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )

    # Define training step
    # @jax.jit
    def train_step(state, D, rng_key):
        loss, grads = jax.value_and_grad(model.loss_offline)(state.params, D, rng_key)
        state = state.apply_gradients(grads=grads)
        return state, loss

    # Training parameters
    num_epochs = 100
    J = 32  # Mini-batch size

    # Since the loss function uses the entire dataset, we might need to implement batching
    num_batches = M // J

    for epoch in range(num_epochs):
        # Update random key
        key, subkey_shuffle, subkey_loss = jax.random.split(key, 3)

        # Shuffle data
        permutation = jax.random.permutation(subkey_shuffle, M)
        Dxs_shuffled = Dxs[permutation]
        Dys_shuffled = Dys[permutation]

        epoch_loss = 0.0
        for i in range(num_batches):
            # Prepare batch data
            start = i * J
            end = start + J
            D_batch = (Dxs_shuffled[start:end], Dys_shuffled[start:end])

            # Perform a training step
            state, loss = train_step(state, D_batch, subkey_loss)
            epoch_loss += loss

        epoch_loss /= num_batches

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss}")

if __name__ == "__main__":
    main()
