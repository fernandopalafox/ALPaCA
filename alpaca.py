import jax
from jax import numpy as jnp
from flax import linen as nn

"""
Implementation of the ALPaCA algorithm as described in the paper. 
Tried to keep notation close to the paper
"""

class ALPaCA:
    def __init__(
            self,
            phi: nn.Module,
            w: jnp.ndarray,
            Sigma_eps: jnp.ndarray,
            rng_key: int = 0
    ) -> None:
        self.Sigma_eps = Sigma_eps
        self.phi = phi
        self.w = w
        self.rng_key = rng_key
        self.n_phi = None # FIND OUT HOW TO GET THIS
        self.n_y = None # FIND OUT HOW TO GET THIS

    def offline(
        self, D: tuple[jnp.ndarray, jnp.ndarray]
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Offline training of the ALPaCA algorithm

        Args:
            D: jnp.ndarray
                Training data with J trajectories. 
                (traj, time, features)
        """

        # Indices for sampling trajectories and trajectory lengths
        Dxs, Dys = D
        J, tau, _ = Dxs.shape
        self.rng_key, subkey_1, subkey_2 = jax.random.split(self.rng_key, 3)
        t_js = jax.random.randint(subkey_1, (J,), 0, tau-1)
        dataset_indices = jax.random.randint(subkey_2, (J,), 0, J-1)

        def loss_fn(Kbar_0, L0, w): # figure out where this goes
            loss = 0
            for j in range(J):  # TODO: vmap this. Iterations are independent.

                # Index relevant data
                Dx_j = Dxs[dataset_indices[j], :t_js[j], :] # (t_j, n_x)
                Y_j = Dys[dataset_indices[j], :t_js[j], :] # (t_j, n_y)
                x_jp1 = Dxs[dataset_indices[j], t_js[j], :] # (n_x)
                y_jp1 = Dys[dataset_indices[j], t_js[j], :] # (n_y)

                # Apply phi to data
                Phi_j = self.phi.apply(w, Dx_j)  # (t_j, n_phi)
                phi_j = self.phi.apply(w, x_jp1) # (n_phi)

                # Compute Kbar_j, Lambda_j, Sigma_j (eq. 12)
                Lambda_0 = L0 @ L0.T # (n_phi, n_phi)
                Lambda_j = Phi_j.T @ Phi_j + Lambda_0 # (n_phi, n_phi)
                Lambda_j_inv = jnp.linalg.inv(Lambda_j) # TODO: Avoid inversion
                Sigma_j = (1 + phi_j.T @ Lambda_j_inv @ phi_j) @ self.Sigma_eps # (n_y, n_y)
                Kbar_j = Lambda_j_inv @ (Phi_j.T @ Y_j + Lambda_0 @ Kbar_0) # (n_phi, n_y)

                # Update loss (eq. 11)
                y_delta = y_jp1 - Kbar_j.T @ phi_j
                loss += (
                    self.ny * jnp.log(1 + phi_j.T @ Lambda_j_inv @ phi_j)
                    + y_delta.T @ jnp.linalg.inv(Sigma_j) @ y_delta
                ) / J
            return loss    
        return K0, Lamda0, w
    
def ALPaCA(nn.Module):
    phi: nn.Module
    Sigma_eps: jnp.ndarray
    n_y: int
    n_x: int
    n_phi: int

    def setup(self):
        self.Kbar_0 = self.param('Kbar_0', nn.zeros_init(), (self.n_phi, self.n_y))
        self.L0 = self.param('L0', nn.zeros_init(), (self.n_phi, self.n_phi))
        nn.share_scope(self, self.phi)