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
        self.L0 = self.param(
            "L0", nn.initializers.normal(stddev=0.1), (self.n_phi, self.n_phi)
        )
        self.Kbar_0 = self.param(
            "Kbar_0", nn.initializers.normal(stddev=0.1), (self.n_phi, self.n_y)
        )
        nn.share_scope(self, self.phi)

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """Applies feature mapping and returns the model's prediction."""
        phi_x = self.phi(x)
        return (self.Kbar_0.T[None, :, :] @ phi_x[..., None]).squeeze(-1)

    def loss_offline(
        self, params: dict, D: tuple[jnp.ndarray, jnp.ndarray], masks: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Offline training of the ALPaCA algorithm.

        Args:
            params (dict): model parameters, 'Kbar_0', 'L0', and NN parameters (w).
            D (tuple): tuple of Dxs and Dys, where:
                - Dxs (jnp.ndarray): input trajectories with shape (J, tau, n_x)
                - Dys (jnp.ndarray): target trajectories with shape (J, tau, n_y)
                - Mini-batch of J trajectories from a dataset of M trajectories.
            masks (jax.ndarray): mask for the trajectories, shape (J, tau)

        Returns:
            jnp.ndarray: averaged loss across sampled trajectories.
        """

        Kbar_0, L0 = params["params"]["Kbar_0"], params["params"]["L0"]

        # split dataset and initialize RNG keys
        Dxs, Dys = D
        J, tau, _ = Dxs.shape

        loss = 0.0
        # TODO: average loss over entire trajectory length.
        #       See last sentence p. 7 of the paper.
        for j in range(J):
            # sample data for current index
            mask = masks[j]  # (tau, 1)
            prediction_index = jnp.sum(mask, dtype=int)  # (1)
            Dx_j = Dxs[j]  # (tau, n_x)
            Y_j = Dys[j]  # (tau, n_y)
            x_jp1 = Dxs[j, prediction_index, :]  # (n_x)
            y_jp1 = Dys[j, prediction_index, :]  # (n_y)

            # compute NN features
            Phi_j = self.phi.apply(params, Dx_j)  # (tau, n_phi)
            phi_jp1 = self.phi.apply(params, x_jp1)  # (n_phi)

            # compute Lambda_j and Sigma_jp1 using stable solve instead of inv
            Lambda_0 = L0 @ L0.T
            Lambda_j = Phi_j.T @ (mask * Phi_j) + Lambda_0
            Lambda_j_inv_phi = jnp.linalg.solve(Lambda_j, phi_jp1)  # avoid inversion
            Sigma_jp1 = (1 + phi_jp1.T @ Lambda_j_inv_phi) * self.Sigma_eps

            # compute Kbar_j
            Kbar_j = jnp.linalg.solve(Lambda_j, Phi_j.T @ (mask * Y_j) + Lambda_0 @ Kbar_0)

            # loss
            y_delta = y_jp1 - Kbar_j.T @ phi_jp1
            loss += (
                self.n_y * jnp.log(1 + phi_jp1.T @ Lambda_j_inv_phi)
                + y_delta.T @ jnp.linalg.solve(Sigma_jp1, y_delta)
            )
            
        loss /= J

        return loss

    def online_update(self, params: dict, D: tuple[jnp.ndarray, jnp.ndarray]) -> dict:
        """
        Online update of the linear model parameters (Algorithm 2).

        Args:
            params (dict): model parameters, 'Kbar_0', 'L0', and NN parameters (w).
            D (tuple): tuple of Dx and Dy, where:
                - Dx (jnp.ndarray): single input trajectory with shape (tau, n_x)
                - Dy (jnp.ndarray): single output trajectory with shape (tau, n_y)

        Returns:
            params (dict): model parameters with updated 'Kbar_t' and 'Lambda_t_inv'
        """

        Kbar_0, L0 = params["params"]["Kbar_0"], params["params"]["L0"]
        Dx, Dy = D
        tau = Dx.shape[0]
        Lambda_0 = L0 @ L0.T
        Q_tm1 = Lambda_0 @ Kbar_0
        Lambda_tm1_inv = jnp.linalg.inv(Lambda_0)

        for t in range(tau):  # TODO: lax.scan this
            phi_t = self.phi.apply(params, Dx[t])[:, jnp.newaxis]  # (n_phi, 1)
            Lambda_t_inv = (
                Lambda_tm1_inv
                - 1
                / (1 + phi_t.T @ Lambda_tm1_inv @ phi_t)
                * (Lambda_tm1_inv @ phi_t)
                @ (Lambda_tm1_inv @ phi_t).T
            )
            y_t = Dy[t][:, jnp.newaxis]  # (n_y, 1)
            Q_t = phi_t * y_t + Q_tm1
            Kbar_t = Lambda_t_inv @ Q_t

            Lambda_tm1_inv = Lambda_t_inv
            Q_tm1 = Q_t

        if tau > 0:
            params["params"]["Kbar_t"] = Kbar_t
            params["params"]["Lambda_t_inv"] = Lambda_t_inv

        return params

    def predict(self, params: dict, Dx: jnp.ndarray) -> jnp.ndarray:
        """
        Predict output given input trajectory.

        Args:
            params (dict): model parameters
            Dx (jnp.ndarray): input trajectory with shape (tau, n_x)

        Returns:
            Ybar (jnp.ndarray): predicted output trajectory with shape (tau, n_y)
            Sigma (jnp.ndarray): predicted output covariance with shape (tau, n_y, n_y)
        """

        # Extract parameters
        if "Kbar_t" not in params["params"]:
            Kbar_t = params["params"]["Kbar_0"]
        else:
            Kbar_t = params["params"]["Kbar_t"]

        if "Lambda_t_inv" not in params["params"]:
            Lambda_t_inv = jnp.linalg.inv(
                params["params"]["L0"] @ params["params"]["L0"].T
            )
        else:
            Lambda_t_inv = params["params"]["Lambda_t_inv"]

        tau = Dx.shape[0]
        Phi = self.phi.apply(params, Dx)  # (tau, n_phi)
        Ybar = Phi @ Kbar_t  # (tau, n_y)
        Sigma = jnp.zeros((tau, self.n_y, self.n_y))
        for t in range(tau):  # TODO: Nasty. Figure out broadcasting
            phi_t = Phi[t]  # (n_phi)
            Sigma = Sigma.at[t].set(
                (1 + phi_t.T @ Lambda_t_inv @ phi_t) * self.Sigma_eps
            )

        return Ybar, Sigma


# Define the default feature mapping phi
class DefaultFeatureMapping(nn.Module):
    n_phi: int  # Dimension of the feature mapping output

    @nn.compact
    def __call__(self, x):
        # MLP with two hidden layers and 128 units each
        x = nn.Dense(features=128)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=128)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=self.n_phi)(x)
        return x
