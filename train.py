import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from data_generation import generate_sinusoid_data
from alpaca import ALPaCA, DefaultFeatureMapping
import matplotlib.pyplot as plt
import time
import datetime
import pickle


def main():
    # Set random seed
    key = jax.random.key(0)

    # Generate data
    M = 100  # Total number of trajectories
    tau = 20  # Number of time steps per trajectory
    amplitude_range = (0.1, 5.0)
    phase_range = (0.0, jnp.pi)
    time_range = (-5, 5)

    Dxs, Dys = generate_sinusoid_data(
        M, tau, key, amplitude_range, phase_range, time_range
    )

    # Define model parameters
    n_x = Dxs.shape[-1]  # Input dimension (should be 1)
    n_y = Dys.shape[-1]  # Output dimension (should be 1)
    n_phi = 32  # Dimension of feature mapping output

    # Define noise covariance
    Sigma_eps = jnp.eye(n_y) * 0.02  # Assumed noise covariance

    # Instantiate the feature mapping module
    phi = DefaultFeatureMapping(n_phi=n_phi)

    # Instantiate the ALPaCA model
    model = ALPaCA(phi=phi, Sigma_eps=Sigma_eps, n_y=n_y, n_x=n_x, n_phi=n_phi)
    params_name = "data/model_params_no_meta.pkl"
    meta_learn = False

    # Initialize model parameters
    x_init = jnp.ones((1, n_x))
    params = model.init(key, x_init)

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
    @jax.jit
    def train_step(state, D, masks):
        loss, grads = jax.value_and_grad(model.loss_offline)(state.params, D, masks)

        # Set Kbar_0 and L0 gradients to zero if not meta-learning
        if not meta_learn:
            grads["params"]["Kbar_0"] = jnp.zeros_like(grads["params"]["Kbar_0"])
            grads["params"]["L0"] = jnp.zeros_like(grads["params"]["L0"])

        state = state.apply_gradients(grads=grads)
        
        return state, loss

    # Training parameters
    num_epochs = 1000
    J = 10  # Mini-batch size

    # Since the loss function uses the entire dataset, we might need to implement batching
    num_batches = M // J

    losses = []
    time_start = time.time()
    try:
        for epoch in range(num_epochs):
            # Update random key
            key, subkey_shuffle, subkey_masks = jax.random.split(key, 3)

            # Shuffle data
            permutation = jax.random.permutation(subkey_shuffle, M)
            Dxs_shuffled = Dxs[permutation]
            Dys_shuffled = Dys[permutation]

            # Sample trajectory lengths and create masks
            trajectory_lengths = jax.random.randint(subkey_masks, (J,), 1, tau + 1)
            indices = jnp.arange(tau)  # (tau,)
            indices = indices[None, :]    # (1, tau)
            masks = (indices < trajectory_lengths[:, None]).astype(jnp.float32) # (J, tau)
            masks = masks[:, :, jnp.newaxis]  # add axis for easy broadcasting later

            epoch_loss = 0.0
            for i in range(num_batches):
                # Prepare batch data
                start = i * J
                end = start + J
                D_batch = (Dxs_shuffled[start:end], Dys_shuffled[start:end])

                # Perform a training step
                state, loss = train_step(state, D_batch, masks)
                epoch_loss += loss

            epoch_loss /= num_batches
            losses.append(epoch_loss)
            print(f"Epoch {epoch}, Loss: {epoch_loss}")

    except KeyboardInterrupt:
        with open(params_name, "wb") as f:
            pickle.dump(state.params, f)

    time_end = time.time()
    total_time = time_end - time_start
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    # Save model
    with open(params_name, "wb") as f:
        pickle.dump(state.params, f)

    # Plot losses
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss (Total Time: {total_time_str})")
    plt.savefig("figures/training_loss.png")


if __name__ == "__main__":
    main()
