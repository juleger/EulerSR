import jax
import jax.numpy as jnp

# Seuil Huber calibré pour des données normalisées 
_HUBER_DELTA = 1.0


def _huber_elementwise(r: jax.Array, delta: float = _HUBER_DELTA) -> jax.Array:
    return jnp.where(jnp.abs(r) < delta,
                     0.5 * r ** 2,
                     delta * (jnp.abs(r) - 0.5 * delta))


def mse(pred: jax.Array, target: jax.Array, weights: jax.Array) -> jax.Array:
    return ((pred - target) ** 2).mean()


def rel_mse(pred: jax.Array, target: jax.Array, weights: jax.Array) -> jax.Array:
    # MSE relative par variable : les primitives n'ont pas la même échelle
    per_var = ((pred - target) ** 2).mean(0)
    ref = (target ** 2).mean(0) + 1e-8
    return (per_var / ref).mean()


def shock_weighted_mse(pred: jax.Array, target: jax.Array, weights: jax.Array) -> jax.Array:
    return (weights[:, None] * (pred - target) ** 2).mean()


def shock_weighted_rel_mse(pred: jax.Array, target: jax.Array, weights: jax.Array) -> jax.Array:
    per_var = (weights[:, None] * (pred - target) ** 2).mean(0)
    ref = (target ** 2).mean(0) + 1e-8
    return (per_var / ref).mean()


def huber(pred: jax.Array, target: jax.Array, weights: jax.Array, delta: float = _HUBER_DELTA) -> jax.Array:
    std = jnp.sqrt((target ** 2).mean(0) + 1e-8)  # (n_vars,) std par variable sur le batch
    r = (pred - target) / std  # résidu relatif : delta en unités de std
    return _huber_elementwise(r, delta).mean()


def shock_weighted_huber(pred: jax.Array, target: jax.Array, weights: jax.Array, delta: float = _HUBER_DELTA) -> jax.Array:
    std = jnp.sqrt((target ** 2).mean(0) + 1e-8)
    r = (pred - target) / std
    return (weights[:, None] * _huber_elementwise(r, delta)).mean()


def sliced_wasserstein_loss(pred: jax.Array, ref: jax.Array, weights: jax.Array) -> jax.Array:
    """Sliced Wasserstein SW2 (trié par variable) sur champ normalisé."""
    p = jnp.sort(pred, axis=0)
    r = jnp.sort(ref, axis=0)
    return jnp.mean((p - r) ** 2)


LOSS_FNS = {
    'mse': mse,
    'rel_mse': rel_mse,
    'shock_weighted_mse': shock_weighted_mse,
    'shock_weighted_rel_mse': shock_weighted_rel_mse,
    'huber': huber,
    'shock_weighted_huber': shock_weighted_huber,
    'sliced_wasserstein': sliced_wasserstein_loss,
}


def grad_p_lsq(pred: jax.Array, knn: dict) -> jax.Array:
    """Gradient de pression LSQ sur le maillage HR"""
    p = pred[:, 3]
    return jnp.einsum('ndk,nk->nd', knn['grad_op'], p[knn['hr_idx']] - p[:, None])


def grad_all_lsq(pred: jax.Array, knn: dict) -> jax.Array:
    """Gradient LSQ de toutes les variables → (N, 2, 4)"""
    diff = pred[knn['hr_idx']] - pred[:, None, :]  # (N, k, 4)
    return jnp.einsum('ndk,nkv->ndv', knn['grad_op'], diff)
