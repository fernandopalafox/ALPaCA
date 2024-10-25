import flax.linen as nn
import jax
from jax import numpy as jnp, random

print(jax.__version__)

class DenseLoRA(nn.Module):
  base: nn.Dense
  rank: int

  def setup(self):
    nn.share_scope(self, self.base)

  @nn.compact
  def __call__(self, x: jax.Array):
    din, dout = x.shape[-1], self.base.features
    A = self.param('A', nn.zeros_init(), (din, self.rank))
    B = self.param('B', nn.zeros_init(), (self.rank, dout))
    return self.base(x) + x @ A @ B

model = DenseLoRA(base=nn.Dense(10), rank=5)

params = model.init(random.key(0), jnp.ones((1, 5)))
print(list(params['params'].keys()))

# call base model 
y = model.base.apply(params, jnp.ones((1, 5)))
print(y.shape)