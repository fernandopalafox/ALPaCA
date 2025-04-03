import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax
from alpaca import ALPaCA, DefaultFeatureMapping
import matplotlib.pyplot as plt
import time
import datetime
import pickle
import yaml


def main(config, data_loader=None):
    # Set random seed
    key = jax.random.key(config["seed"])

    # Define model parameters
    n_x = config["n_x"]  # Input dimension (should be 1)
    n_y = config["n_y"]  # Output dimension (should be 1)
    n_phi = config["n_phi"]  # Dimension of feature mapping output

    # Define noise covariance
    Sigma_eps = jnp.eye(n_y) * config["sigma_eps"]  # Assumed noise covariance

    # Instantiate the feature mapping module
    phi = DefaultFeatureMapping(n_phi=n_phi)

    # Instantiate the ALPaCA model
    model = ALPaCA(phi=phi, Sigma_eps=Sigma_eps, n_y=n_y, n_x=n_x, n_phi=n_phi)
    model_params_name = "data/" + config["model_params_name"]
    meta_learn = config["meta_learn"]

    # Initialize model parameters
    x_init = jnp.ones((1, n_x))
    params = model.init(key, x_init)

    # Define optimizer
    learning_rate = config["learning_rate"]
    optimizer = optax.adam(learning_rate)

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
    num_epochs = config["num_epochs"]
    J = config["batch_size"]

    # Load data
    if data_loader is None:
        with open(config["data_path"], "rb") as f:
            Dxs, Dys = pickle.load(f)
        M = Dxs.shape[0]  # Total number of trajectories
        tau = Dxs.shape[1]  # Number of time steps per trajectory

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
            indices = indices[None, :]  # (1, tau)
            masks = (indices < trajectory_lengths[:, None]).astype(
                jnp.float32
            )  # (J, tau)
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
            print(f"Epoch {epoch}, Loss: {epoch_loss:.3e}")

    except KeyboardInterrupt:
        with open(model_params_name, "wb") as f:
            pickle.dump(state.params, f)

    time_end = time.time()
    total_time = time_end - time_start
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    # Save model
    with open(model_params_name, "wb") as f:
        pickle.dump(state.params, f)

    # Plot losses
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss (Total Time: {total_time_str})")
    plt.savefig("figures/training_loss.png")


if __name__ == "__main__":
    cfg_filename = "configs/sinusoid.yml"
    with open(cfg_filename, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
    main(config)
