import jax
import jax.numpy as jnp


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


def w2_loss(pred: jax.Array, ref: jax.Array, weights: jax.Array) -> jax.Array:
    """Wasserstein W2 (trié par variable) sur champ normalisé."""
    p = jnp.sort(pred, axis=0)
    r = jnp.sort(ref, axis=0)
    return jnp.mean((p - r) ** 2)


LOSS_FNS = {
    'mse': mse,
    'rel_mse': rel_mse,
    'shock_weighted_mse': shock_weighted_mse,
    'shock_weighted_rel_mse': shock_weighted_rel_mse,
    'w2': w2_loss,
}

# Constantes physiques
from utils.refs import GAMMA as _GAMMA
from preprocessing.dataset import _MACH_MID, _MACH_SCALE
_P_INF = 1.0
_RHO_INF = 1.0


def enthalpy_residual(pred_norm: jax.Array, mach_n: jax.Array,
                      mu: jax.Array, sig: jax.Array, gamma: float = _GAMMA,
                      mach_norm: tuple[float, float] = (_MACH_MID, _MACH_SCALE)) -> jax.Array:
    """Résidu RMS d'enthalpie totale (= 0 pour une solution Euler parfaite), différentiable.

    mach_norm = (mid, scale) résolu pour CE run d'entraînement (cf.
    models/base.py:_aux, dérivé de train_ds.mach_mid/mach_scale)"""
    prim = pred_norm * sig + mu  # dénormalisation
    rho = jnp.maximum(prim[:, 0], 1e-12)
    H = gamma / (gamma - 1) * prim[:, 3] / rho + 0.5 * (prim[:, 1] ** 2 + prim[:, 2] ** 2)
    mach_mid, mach_scale = mach_norm
    mach_in = mach_n * mach_scale + mach_mid
    v_inf = mach_in * jnp.sqrt(gamma * _P_INF / _RHO_INF)
    H_inf = gamma / (gamma - 1) * _P_INF / _RHO_INF + 0.5 * v_inf ** 2
    return jnp.sqrt((((H - H_inf) / H_inf) ** 2).mean() + 1e-12)


def grad_p_lsq(pred: jax.Array, knn: dict) -> jax.Array:
    """Gradient de pression LSQ sur le maillage HR"""
    p = pred[:, 3]
    return jnp.einsum('ndk,nk->nd', knn['grad_op'], p[knn['hr_idx']] - p[:, None])
