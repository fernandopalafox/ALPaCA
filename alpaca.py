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
        self.Kbar_0 = self.param('Kbar_0', nn.zeros_init(), (self.n_phi, self.n_y))
        self.L0 = self.param('L0', nn.zeros_init(), (self.n_phi, self.n_phi))
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

        Kbar_0, L0 = params['Kbar_0'], params['L0']

        # split dataset and initialize RNG keys
        Dxs, Dys = D
        J, tau, _ = Dxs.shape
        rng_key, subkey = jax.random.split(rng_key, 3)
        t_js = jax.random.randint(subkey, (J,), 0, tau - 1)

        def compute_loss(j: int) -> jnp.ndarray:
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
            loss = (
                self.n_y * jnp.log(1 + phi_j.T @ Lambda_j_inv_phi)
                + y_delta.T @ jnp.linalg.solve(Sigma_j, y_delta)
            ) / J
            return loss

        # parallel computation of losses (they're all independent)
        losses = jax.vmap(compute_loss)(jnp.arange(J))
        return jnp.sum(losses)