import jax
from jax import numpy as jnp
from typing import Callable


class GaussianProcess:
    def __init__(self, kernel: Callable, scale: float, noise: float) -> None:
        self.kernel = kernel
        self.scale = scale
        self.noise = noise

    def fit(self, xs_train: jnp.ndarray, ys_train: jnp.ndarray) -> None:
        self.xs_train = xs_train
        self.ys_train = ys_train
        self.K = self.kernel(xs_train, xs_train, self.scale) + self.noise * jnp.eye(
            len(xs_train)
        )
        self.K_inv = jnp.linalg.inv(self.K)

    def predict(self, xs_test: jnp.ndarray) -> jnp.ndarray:
        K_star = self.kernel(self.xs_train, xs_test, self.scale)
        K_star_star = self.kernel(xs_test, xs_test, self.scale) + self.noise * jnp.eye(
            len(xs_test)
        )
        mu_star = K_star.T @ self.K_inv @ self.ys_train
        cov_star = K_star_star - K_star.T @ self.K_inv @ K_star
        return mu_star, cov_star


def rbf(xi: jnp.ndarray, xj: jnp.ndarray, scale: float) -> jnp.ndarray:
    xi = xi.reshape(-1, 1)
    xj = xj.reshape(-1, 1)
    return jnp.exp(-jnp.abs(xi - xj.T) ** 2 / (2 * scale**2))
