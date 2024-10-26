import jax
from jax import numpy as jnp
from flax import linen as nn

"""
Implementation of ALPaCA as described in https://arxiv.org/abs/1807.08912
Notation is consistent with the paper
"""

class ALPaCA(nn.Module):
    phi: nn.Module 
    Sigma_eps: jnp.ndarray
    n_y: int 
    n_x: int
    n_phi: int

    def setup(self):
        self.L0 = self.param("L0", L0_initializer, (self.n_phi, self.n_phi))
        self.Kbar_0 = self.param(
            "Kbar_0", nn.initializers.normal(stddev=0.1), (self.n_phi, self.n_y)
        )
        nn.share_scope(self, self.phi)

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """Applies feature mapping and returns the model's prediction."""
        phi_x = self.phi(x)
        return (self.Kbar_0.T[None,:,:] @ phi_x[...,None]).squeeze(-1)

    def loss_offline(
        self,
        params: dict,
        D: tuple[jnp.ndarray, jnp.ndarray],
        rng_key: jax.Array
    ) -> jnp.ndarray:
        """
        Offline training of the ALPaCA algorithm.

        Args:
            params (dict): model parameters, 'Kbar_0', 'L0', and NN parameters (w).
            D (tuple): tuple of Dxs and Dys, where:
                - Dxs (jnp.ndarray): input trajectories with shape (J, tau, n_x)
                - Dys (jnp.ndarray): target trajectories with shape (J, tau, n_y)
                - Mini-batch of J trajectories from a dataset of M trajectories.

        Returns:
            jnp.ndarray: averaged loss across sampled trajectories.
        """

        Kbar_0, L0 = params["params"]["Kbar_0"], params["params"]["L0"]

        # split dataset and initialize RNG keys
        Dxs, Dys = D
        J, tau, _ = Dxs.shape
        rng_key, subkey = jax.random.split(rng_key, 2)
        t_js = jax.random.randint(subkey, (J,), 0, tau - 1)

        loss = 0.0
        # I tried vmaping this but trajectory length is variable
        # TODO: figure out if we can pad the data to a fixed length
        for j in range(J):
            # sample data for current index
            Dx_j = Dxs[j, :t_js[j], :]  # (t_j, n_x)
            Y_j = Dys[j, :t_js[j], :]    # (t_j, n_y)
            x_jp1 = Dxs[j, t_js[j], :]   # (n_x)
            y_jp1 = Dys[j, t_js[j], :]   # (n_y)

            # compute NN features
            Phi_j = self.phi.apply(params, Dx_j)  # (t_j, n_phi)
            phi_j = self.phi.apply(params, x_jp1)  # (n_phi)

            # compute Lambda_j and Sigma_j using stable solve instead of inv
            Lambda_0 = L0 @ L0.T
            Lambda_j = Phi_j.T @ Phi_j + Lambda_0
            Lambda_j_inv_phi = jnp.linalg.solve(Lambda_j, phi_j)  # avoid inversion
            Sigma_j = (1 + phi_j.T @ Lambda_j_inv_phi) * self.Sigma_eps

            # compute Kbar_j
            Kbar_j = jnp.linalg.solve(Lambda_j, Phi_j.T @ Y_j + Lambda_0 @ Kbar_0)

            # loss
            y_delta = y_jp1 - Kbar_j.T @ phi_j
            loss += (
                self.n_y * jnp.log(1 + phi_j.T @ Lambda_j_inv_phi)
                + y_delta.T @ jnp.linalg.solve(Sigma_j, y_delta)
            ) / J

        return loss


def L0_initializer(key: jax.random.PRNGKey, shape: tuple, dtype=jnp.float32):
    """
    Custom initializer for the lower triangular matrix L0.

    Args:
        key (jax.random.PRNGKey): random key for initialization.
        shape (tuple): shape of the matrix.
        dtype (jnp.dtype): data type of the matrix.

    """

    n = shape[0]
    assert shape[0] == shape[1], "L0 must be a square matrix"
    # Generate positive values for the diagonal
    diag_key, off_diag_key = jax.random.split(key)
    diag_values = jax.random.uniform(
        diag_key, (n,), minval=0.1, maxval=1.0, dtype=dtype
    )
    # Generate random values for the off-diagonal lower triangular part
    tril_indices = jnp.tril_indices(n, k=-1)
    num_off_diag_elements = len(tril_indices[0])
    off_diag_values = jax.random.normal(
        off_diag_key, (num_off_diag_elements,), dtype=dtype
    )
    # Create a zeros matrix
    L = jnp.zeros((n, n), dtype=dtype)
    # Set the diagonal
    L = L.at[jnp.diag_indices(n)].set(diag_values)
    # Set the off-diagonal lower triangular values
    L = L.at[tril_indices].set(off_diag_values)
    return L
