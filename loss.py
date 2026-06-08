import jax
import jax.numpy as jnp


def mse(pred: jax.Array, target: jax.Array, weights: jax.Array) -> jax.Array:
    # Loss classique MSE purement "data"
    return ((pred - target) ** 2).mean()


def rel_mse(pred: jax.Array, target: jax.Array, weights: jax.Array) -> jax.Array:
    # MSE pondérée par primitives obligatoire car les variables ne s'apprennent pas à la même échelle 
    # (par ex u est triviale, p est plus complexe)
    per_var = ((pred - target) ** 2).mean(0)
    ref     = (target ** 2).mean(0) + 1e-8
    return (per_var / ref).mean()


def shock_weighted_mse(pred: jax.Array, target: jax.Array, weights: jax.Array) -> jax.Array:
    # MSE pondérée par les poids de choc (notamment gradient de pression)
    return (weights[:, None] * (pred - target) ** 2).mean()


def shock_weighted_rel_mse(pred: jax.Array, target: jax.Array, weights: jax.Array) -> jax.Array:
    per_var = (weights[:, None] * (pred - target) ** 2).mean(0)
    ref     = (target ** 2).mean(0) + 1e-8
    return (per_var / ref).mean()

def rel_l2(pred: jax.Array, target: jax.Array) -> jax.Array:
    return jnp.sqrt(((pred - target) ** 2).sum(0) / ((target ** 2).sum(0) + 1e-8))


LOSS_FNS = {
    'mse':                  mse,
    'rel_mse':              rel_mse,
    'shock_weighted_mse':   shock_weighted_mse,
    'shock_weighted_rel_mse': shock_weighted_rel_mse,
}
