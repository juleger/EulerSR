"""
MLP corrector : Perceptron multi-couche simple qui prédit une correction à ajouter aux primitives LR interpolées pour obtenir les primitives HR. Utilise les features HR et les features LR interpolées (optionnellement avec les gradients LR) comme contexte.

Equivalent d'une interpolation IDW "apprise" : le MLP cell-wise ajoute de la non-linéarité pour corriger les erreurs d'interpolation, en particulier près des chocs, même s'il reste assez simple et rapide à entrainer.
"""

import re
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.base import SRModel, MLP, n_params, _knn_path, load_cfg
import euler.jax_fvm.src.mesh  # noqa: F401

_PROC_RE = re.compile(r'^aoa([+-]?\d+\.\d+)_m(\d+\.\d+)$')


class MlpCorrector(SRModel):
    phi: MLP

    def __init__(self, rngs: nnx.Rngs, cfg: dict | None = None):
        arch = (cfg or {}).get('architecture', {})
        hidden      = arch.get('hidden_dims', [128, 128, 64])
        use_lr_grad   = arch.get('use_lr_grad', False)
        lr_feat_dim   = 9 if use_lr_grad else 6   # prim(4)+pos(2)[+grad_p(2)+div_u(1)]
        self.use_residual = arch.get('use_residual', True)

        # MLP simple cell-wise
        self.phi = MLP([4 + lr_feat_dim] + hidden + [4], rngs)

    def predict(self, hr_feat: jax.Array, lr_feat: jax.Array, knn: dict) -> jax.Array:
        # Prédiction des primitives HR à partir des features HR, des features LR interpolées, et des kNN
        idx, dist = knn['idx'], knn['dist']
        w = 1.0 / (dist + 1e-6)
        w = w / w.sum(axis=1, keepdims=True)
        ctx = (w[:, :, None] * lr_feat[idx]).sum(axis=1)
        x = jnp.concatenate([hr_feat, ctx], axis=-1)
        delta = jax.vmap(self.phi)(x)
        return (ctx[:, :4] + delta) if self.use_residual else delta

    def _grad_p(self, hr_feat: jax.Array, lr_feat: jax.Array, knn: dict) -> jax.Array:
        # Calcul du gradient de pression à partir des features HR et LR, pour le weighting de la loss
        idx, dist = knn['idx'], knn['dist']
        w = 1.0 / (dist + 1e-6)
        w = w / w.sum(axis=1, keepdims=True)
        ctx = (w[:, :, None] * lr_feat[idx]).sum(axis=1)
        x = jnp.concatenate([hr_feat, ctx], axis=-1)

        def p_i(pos_i, tail_i):
            return self.phi(jnp.concatenate([pos_i, tail_i]))[3]

        return jax.vmap(jax.grad(p_i))(x[:, :2], x[:, 2:])

    @classmethod
    def load_knn(cls, data_dir: Path, cfg: dict | None = None) -> dict:
        # Chargement des kNN pré-calculés pour la résolution HR/LR et le k spécifiés dans cfg
        res = (cfg or {}).get('resolution', {})
        hr_res = res.get('hr', 0.025)
        lr_res = res.get('lr', 0.1)
        k = (cfg or {}).get('architecture', {}).get('k', 6)
        d = np.load(_knn_path(data_dir, hr_res, lr_res, k))
        return {
            'idx': jnp.array(d['indices']),
            'dist': jnp.array(d['dists']),
        }

