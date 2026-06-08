"""
LocalNet : modèle de super-résolution sur maillage local inspiré du PointNet++

Pour chaque cellule du maillage HR, on récupère les kNN dans le maillage LR, afin d'extraire des features locales à partir des primitives LR, avec maxpooling et gating selon la distance.
Possibilité d'utiliser plusieurs échelles de kNN (ex 6 et 16) pour capturer des contextes à des échelles différentes.
"""

import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.base import SRModel, MLP, n_params, _knn_path


class LocalNet(SRModel):
    phi: MLP
    gate: MLP
    norm: nnx.LayerNorm
    psi: MLP

    def __init__(self, rngs: nnx.Rngs, cfg: dict | None = None):
        arch = (cfg or {}).get('architecture', {})
        embed_dim = arch.get('embed_dim', 64)
        dec_hid = arch.get('decoder_hidden', [128, 128, 64])
        scales = list(arch.get('scales', [6, 16]))
        if len(scales) < 1:
            raise ValueError("Au moins une échelle de kNN est requise")
        if scales != sorted(scales):
            scales = sorted(scales)

        use_lr_grad = arch.get('use_lr_grad', False)
        phi_in_dim = 10 if use_lr_grad else 7       # prim(4)+[grad_p(2)+div_u(1)]+rel(2)+dist(1)
        self.use_residual = arch.get('use_residual', True)
        self.scales = scales
        self.use_lr_grad = use_lr_grad
        dec_in_dim    = embed_dim * len(scales) + 4

        # Phi : MLP partagé pour extraire les features locales à partir des kNN
        self.phi  = MLP([phi_in_dim, embed_dim, embed_dim], rngs)
        self.gate = MLP([1, 16, 1], rngs) # Gate pour le pooling des features locales, en fonction de la distance des voisins.
        self.norm = nnx.LayerNorm(dec_in_dim, rngs=rngs) # Normalisation avant le MLP de décodage (stabilité)

        # Psi : MLP pour décoder les features locales + info HR en correction sur les primitives HR
        self.psi  = MLP([dec_in_dim] + dec_hid + [4], rngs)

    def _embed(self, lr_feat: jax.Array, idx_k: jax.Array, rel_k: jax.Array, dist_k: jax.Array) -> jax.Array:
        # Construit l'embedding local pour chaque cellule HR
        prim_n = lr_feat[:, :4]
        parts = [prim_n[idx_k]]
        if self.use_lr_grad:
            grad_p = lr_feat[:, 6:8]
            div_u  = lr_feat[:, 8:9]
            parts.append(grad_p[idx_k])
            parts.append(div_u[idx_k])
        parts += [rel_k, dist_k[:, :, None]]
        phi_in = jnp.concatenate(parts, axis=-1)
        N_e, k_e, d_e = phi_in.shape
        h = self.phi(phi_in.reshape(N_e * k_e, d_e)).reshape(N_e, k_e, -1)
        g = jax.nn.sigmoid(self.gate(dist_k[:, :, None].reshape(N_e * k_e, 1)).reshape(N_e, k_e, 1))
        return (g * h).max(axis=1)

    def predict(self, hr_feat: jax.Array, lr_feat: jax.Array, knn: dict) -> jax.Array:
        # Prédiction des primitives HR à partir des features HR, des features LR interpolées, et des kNN
        idx, rel, dist = knn['idx'], knn['rel'], knn['dist']

        embeds = [self._embed(lr_feat, idx[:, :k], rel[:, :k], dist[:, :k])
                  for k in self.scales]
        dec_in = self.norm(jnp.concatenate(embeds + [hr_feat], axis=-1))
        delta = jax.vmap(self.psi)(dec_in)

        if not self.use_residual:
            return delta
        k0 = self.scales[0]
        w0 = 1.0 / (dist[:, :k0] + 1e-6)
        w0 /= w0.sum(axis=1, keepdims=True)
        baseline = (w0[:, :, None] * lr_feat[idx[:, :k0], :4]).sum(axis=1)
        return baseline + delta

    def _grad_p(self, hr_feat: jax.Array, lr_feat: jax.Array, knn: dict) -> jax.Array:
        # Renvoie le gradient de pression prédit par le modèle, pour les positions HR, à partir des features LR interpolées et des kNN
        idx, rel, dist = knn['idx'], knn['rel'], knn['dist']
        embeds = [self._embed(lr_feat, idx[:, :k], rel[:, :k], dist[:, :k])
                  for k in self.scales]
        ctx = jnp.concatenate(embeds, axis=-1)

        def p_i(pos_i, tail_i, ctx_i):
            dec = self.norm(jnp.concatenate([ctx_i, pos_i, tail_i]))
            return self.psi(dec)[3]

        return jax.vmap(jax.grad(p_i))(hr_feat[:, :2], hr_feat[:, 2:], ctx)

    @classmethod
    def load_knn(cls, data_dir: Path, cfg: dict | None = None) -> dict:
        # Chargement des kNN précalculés HR sur LR
        res = (cfg or {}).get('resolution', {})
        hr_res = res.get('hr', 0.025)
        lr_res = res.get('lr', 0.1)
        scales = list((cfg or {}).get('architecture', {}).get('scales', [6, 16]))
        k_max = max(scales)
        d = np.load(_knn_path(data_dir, hr_res, lr_res, k_max))
        return {
            'idx': jnp.array(d['indices']),
            'rel': jnp.array(d['rel_pos']),
            'dist': jnp.array(d['dists']),
        }

