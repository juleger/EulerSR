"""
DAM : Deterministic Attentional Mesh Network pour la super-résolution CFD.

Ce module héberge aussi le backbone partagé avec FAM :
- AMNet (Attentional Mesh net) : U multi-échelles unique aux deux modèles. Le flow
  matching active simplement le conditionnement temporel (sinusoïdal de t) et
  l'injection du champ bruité x_t à l'encodeur (paramètres d_t > 0 et n_x > 0).
- build_context / load_hierarchical_knn : contexte + hiérarchie kNN partagés.
FAM importe AMNet, build_context et load_hierarchical_knn depuis ce fichier.

Architecture AMNet :
- niveau 0 (HR)       : encodage par noeud + self-attention locale kNN
- descente l -> l+1   : cross-attention locale (requêtes = noeuds grossiers)
- bottleneck          : self-attention complète (champ réceptif global)
- remontée l+1 -> l   : cross-attention locale résiduelle + skip
- décodeur            : LayerNorm + Linear -> 4 canaux (delta ou vitesse), init à zéro

Conditionnement : FiLM/AdaLN (Mach, AoA, résolution, +t si flow) dans chaque bloc.
Champ LR interpolé IDW (+ gradients optionnels) et features de paroi injectés à
chaque niveau. Prédiction DAM : baseline IDW au niveau HR + delta prédit.
"""

import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.base import SRModel, MLP, Buffer, _ensure_knn
from utils.refs import _res_tag
from utils.layout import DataLayout
from utils.aero import wall_feature_array
from preprocessing.knn import build_hierarchy as _build_hierarchy
from utils.attention import (sinusoidal_embedding, AdaLNBlock, LocalAttnBlock,
                             GlobalAttnBlock, PE_DIM, scalar_fourier_embedding,
                             scalar_embed_dim)

_N_SCALARS = 2  # (Mach, AoA) dans hr_feat[:, 2:4]
# Conditionnement : embedding Fourier de (Mach, AoA) + log(h_lr / _LR_REF) brut
_N_COND = scalar_embed_dim(_N_SCALARS) + 1
_N_WD = 4  # features de paroi : [wd_norm, wd_exp, nx, ny]
_IDW_K = 6  # IDW standard du projet : k=6, p=2
_LR_REF = 0.1  # résolution LR de référence (scalaire de cond. = 0 à cette résolution)


class AMNet(nnx.Module):
    """Attentional Mesh net : U multi-échelles partagé DAM/FAM sur hiérarchie de maillages.

    d_t > 0 et n_x > 0 activent le conditionnement temporel et l'entrée x_t
    (flow matching). À 0 (DAM), le réseau est purement déterministe : pred(HR | LR).
    """

    def __init__(self, n_levels: int, n_c: int, d: int, d_cond: int,
                 n_heads: int, n_global: int, mlp_ratio: int, rngs: nnx.Rngs,
                 d_t: int = 0, n_x: int = 0, use_geom_cond: bool = False,
                 n_geoms: int = 8, n_hr_blocks: int = 1, dtype=None):
        self.use_time = d_t > 0
        self.d_t = d_t
        self.n_x = n_x

        # Entrée encodeur : (x_t si flow) + features LR interpolées + PE + wall
        in0 = n_x + n_c + PE_DIM + _N_WD
        # Requêtes pour le pooling vers les niveaux grossiers : PE + LR + wall du niveau
        # grossier (la géométrie est ainsi injectée à chaque niveau, jusqu'au bottleneck).
        q_in = PE_DIM + n_c + _N_WD
        cond_in = (d_t if self.use_time else 0) + _N_COND

        self._use_geom_cond = use_geom_cond
        self.cond_mlp = MLP([cond_in, d_cond, d_cond], rngs, dtype=dtype)
        if use_geom_cond:
            self.geom_emb = nnx.Embed(n_geoms, d_cond, rngs=rngs)
        self.enc = nnx.Linear(in0, d, rngs=rngs, dtype=dtype)

        # Stack de self-attention locale répétée à HR (raffinement pleine résolution)
        self.self_in = [LocalAttnBlock(d, d, d_cond, n_heads, rngs, dtype=dtype)
                        for _ in range(n_hr_blocks)]
        self.block_in = [AdaLNBlock(d, d_cond, rngs, mlp_ratio=mlp_ratio, dtype=dtype)
                         for _ in range(n_hr_blocks)]
        self.down = [LocalAttnBlock(q_in, d, d_cond, n_heads, rngs, dtype=dtype)
                     for _ in range(n_levels)]
        self.down_mix = [AdaLNBlock(d, d_cond, rngs, mlp_ratio=mlp_ratio, dtype=dtype)
                         for _ in range(n_levels)]
        self.global_blocks = [GlobalAttnBlock(d, d_cond, n_heads, rngs,
                                              mlp_ratio=mlp_ratio, dtype=dtype)
                              for _ in range(n_global)]
        self.up = [LocalAttnBlock(d, d, d_cond, n_heads, rngs, dtype=dtype)
                   for _ in range(n_levels)]
        self.up_mix = [AdaLNBlock(d, d_cond, rngs, mlp_ratio=mlp_ratio, dtype=dtype)
                       for _ in range(n_levels)]
        self.self_out = [LocalAttnBlock(d, d, d_cond, n_heads, rngs, dtype=dtype)
                         for _ in range(n_hr_blocks)]
        self.block_out = [AdaLNBlock(d, d_cond, rngs, mlp_ratio=mlp_ratio, dtype=dtype)
                          for _ in range(n_hr_blocks)]

        self.dec_norm = nnx.LayerNorm(d, rngs=rngs)
        # Init à zéro : delta/vitesse = 0 au départ, prédiction = baseline IDW
        self.dec = nnx.Linear(d, 4, rngs=rngs, dtype=dtype)
        self.dec.kernel.value = jnp.zeros_like(self.dec.kernel.value)
        self.dec.bias.value = jnp.zeros_like(self.dec.bias.value)

    def _cond(self, scal: jax.Array, geom_id: jax.Array,
              t: jax.Array | None) -> jax.Array:
        # Embedding Fourier des scalaires physiques (+ temps si flow) + log(h_lr) brut
        parts = []
        if self.use_time:
            parts.append(sinusoidal_embedding(t, self.d_t))
        parts += [scalar_fourier_embedding(scal[:_N_SCALARS]), scal[_N_SCALARS:]]
        cond = self.cond_mlp(jnp.concatenate(parts))
        if self._use_geom_cond:
            cond = cond + self.geom_emb(geom_id)
        return cond

    def __call__(self, c: list, scal: jax.Array, fm: dict, wall: list,
                 geom_id: jax.Array, x_t: jax.Array | None = None,
                 t: jax.Array | None = None) -> jax.Array:
        """
        c       : liste de features LR interpolées par niveau, c[l] = [N_l, n_c]
        scal    : scalaires de conditionnement [2] = (Mach, AoA)
        fm      : dict hiérarchie kNN (pe, down, up, self, cond)
        wall    : liste de features de paroi par niveau, wall[l] = [N_l, _N_WD]
        geom_id : entier identifiant la géométrie (utilisé si use_geom_cond)
        x_t, t  : champ bruité et temps (flow matching uniquement)
        """
        cond = self._cond(scal, geom_id, t)

        enc_parts = ([x_t] if self.n_x else []) + [c[0], fm['pe'][0], wall[0]]
        h = [None] * (len(self.down) + 1)
        h0 = self.enc(jnp.concatenate(enc_parts, axis=-1))
        for sa, blk in zip(self.self_in, self.block_in):
            h0 = h0 + sa(h0, h0, fm['self']['idx'], fm['self']['rel'], cond)
            h0 = blk(h0, cond)
        h[0] = h0

        for l in range(len(self.down)):
            q = jnp.concatenate([fm['pe'][l + 1], c[l + 1], wall[l + 1]], axis=-1)
            hl = self.down[l](q, h[l], fm['down'][l]['idx'], fm['down'][l]['rel'], cond)
            h[l + 1] = self.down_mix[l](hl, cond)

        for blk in self.global_blocks:
            h[-1] = blk(h[-1], cond)

        for l in reversed(range(len(self.up))):
            hf = h[l] + self.up[l](h[l], h[l + 1],
                                   fm['up'][l]['idx'], fm['up'][l]['rel'], cond)
            h[l] = self.up_mix[l](hf, cond)

        h0 = h[0]
        for sa, blk in zip(self.self_out, self.block_out):
            h0 = h0 + sa(h0, h0, fm['self']['idx'], fm['self']['rel'], cond)
            h0 = blk(h0, cond)
        return self.dec(self.dec_norm(h0))


def build_context(model, hr_feat: jax.Array, lr_feat: jax.Array, knn: dict) -> dict:
    """Construit le contexte commun DAM/FAM : features LR interpolées par niveau,
    baseline IDW, scalaires de conditionnement, features de paroi et geom_id
    """
    fm = knn['fm']
    parts = [lr_feat[:, :4]]
    if model.use_lr_grad:
        parts.append(lr_feat[:, 6:9])  # grad_p (2) + div_u (1)
    feats = jnp.concatenate(parts, axis=-1)
    c = [(cl['w'][:, :, None] * feats[cl['idx']]).sum(axis=1) for cl in fm['cond']]
    # Features de paroi par niveau (liste). Tolère un ancien format mono-niveau
    # (array) ou l'absence (zéros) pour rester robuste aux knn non hiérarchiques.
    wall = knn.get('wall_feat')
    if wall is None:
        wall = [jnp.zeros((p.shape[0], _N_WD), hr_feat.dtype) for p in fm['pe']]
    elif not isinstance(wall, (list, tuple)):
        wall = [wall] + [jnp.zeros((p.shape[0], _N_WD), hr_feat.dtype)
                         for p in fm['pe'][1:]]
    geom_id = hr_feat[0, 4].astype(jnp.int32)
    # Scalaire de conditionnement résolution LR (propriété de branche, dans le knn)
    res_scalar = knn.get('res_scalar', jnp.zeros((1,), hr_feat.dtype))
    scal = jnp.concatenate([hr_feat[0, 2:4], res_scalar])
    return {'c': c, 'baseline': c[0][:, :4],
            'scal': scal, 'wall': wall, 'geom_id': geom_id}


def load_hierarchical_knn(layout: DataLayout, default_levels: list,
                          cfg: dict | None = None, tag: str = 'MESH') -> dict:
    """Hiérarchie kNN + features de paroi, partagée par DAM.load_knn / FAM.load_knn"""
    cfg = cfg or {}
    # Résolutions LR/HR : prises sur le layout (source de vérité par branche),
    # pas sur cfg (partagé entre branches en multi-résolution).
    hr_res = layout.hr_res
    lr_res = layout.lr_res
    arch = cfg.get('architecture', {})
    levels = list(arch.get('levels', default_levels))
    k_pool = arch.get('k_pool', 9)
    k_up = arch.get('k_up', 4)
    k_self = arch.get('k_self', 9)
    k_cond = arch.get('k_cond', _IDW_K)

    d = np.load(_ensure_knn(layout, hr_res, lr_res, max(k_cond, _IDW_K)))
    knn = {'idx': jnp.array(d['indices']),
           'rel': jnp.array(d['rel_pos']),
           'dist': jnp.array(d['dists']),
           'res_scalar': jnp.array([float(np.log(lr_res / _LR_REF))], jnp.float32)}
    if 'grad_op' in d:
        knn['grad_op'] = jnp.array(d['grad_op'])
        knn['hr_idx'] = jnp.array(d['hr_idx'])

    def _mesh(r):
        return np.load(layout.mesh_path(r), allow_pickle=True).item()

    mesh_hr = _mesh(hr_res)
    pos_hr = np.asarray(mesh_hr.barycenter, dtype=np.float64)
    lr_pos = np.asarray(_mesh(lr_res).barycenter, dtype=np.float64)
    lvl_meshes = [mesh_hr] + [_mesh(r) for r in levels]
    lvl_pos = [pos_hr] + [np.asarray(m.barycenter, dtype=np.float64)
                          for m in lvl_meshes[1:]]

    # Features de paroi à chaque niveau (géométrie injectée jusqu'au bottleneck)
    knn['wall_feat'] = [jnp.array(wall_feature_array(m, p, lr_res))
                        for m, p in zip(lvl_meshes, lvl_pos)]
    knn['mesh'] = mesh_hr

    res_tag = '_'.join(_res_tag(r) for r in [hr_res] + levels)
    cache = (layout.knn_dir /
             f'fm_{res_tag}_lr{_res_tag(lr_res)}_kp{k_pool}_ku{k_up}'
             f'_ks{k_self}_kc{k_cond}.npz')
    if cache.exists():
        z = dict(np.load(cache))
    else:
        print(f"  [{tag}] construction hiérarchie kNN -> {cache.name}")
        z = _build_hierarchy(lvl_pos, lr_pos, k_pool, k_up, k_self, k_cond)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **z)

    L = len(levels) + 1
    knn['fm'] = {
        'pe': [jnp.array(z[f'pe{l}']) for l in range(L)],
        'cond': [{'idx': jnp.array(z[f'cond{l}_idx']),
                  'w': jnp.array(z[f'cond{l}_w'])} for l in range(L)],
        'down': [{'idx': jnp.array(z[f'down{l}_idx']),
                  'rel': jnp.array(z[f'down{l}_rel'])} for l in range(L - 1)],
        'up': [{'idx': jnp.array(z[f'up{l}_idx']),
                'rel': jnp.array(z[f'up{l}_rel'])} for l in range(L - 1)],
        'self': {'idx': jnp.array(z['self_idx']),
                 'rel': jnp.array(z['self_rel'])},
    }
    return knn


class DAM(SRModel):
    """Deterministic Attentional Mesh : SR direct avec backbone hiérarchique AMNet."""

    DEFAULT_LEVELS = [0.05, 0.1]

    def __init__(self, rngs: nnx.Rngs, cfg: dict | None = None):
        cfg = cfg or {}
        arch = cfg.get('architecture', {})

        self.levels = list(arch.get('levels', self.DEFAULT_LEVELS))
        self.use_lr_grad = arch.get('use_lr_grad', True)
        self.n_c = 4 + (3 if self.use_lr_grad else 0)

        use_bf16 = arch.get('use_bf16', False)
        compute_dtype = jnp.bfloat16 if use_bf16 else None

        self._use_geom_cond = arch.get('use_geom_cond', False)
        # Stats d'entraînement (mu/sig du dataset primaire, sauvées avec l'état)
        self.mu_train = Buffer(jnp.zeros((4,), jnp.float32))
        self.sig_train = Buffer(jnp.ones((4,), jnp.float32))
        self.net = AMNet(
            n_levels=len(self.levels),
            n_c=self.n_c,
            d=arch.get('embed_dim', 96),
            d_cond=arch.get('cond_dim', 64),
            n_heads=arch.get('n_heads', 4),
            n_global=arch.get('n_global_blocks', 3),
            mlp_ratio=arch.get('mlp_ratio', 4),
            rngs=rngs,
            d_t=0, n_x=0,  # déterministe : ni temps ni champ bruité
            use_geom_cond=self._use_geom_cond,
            n_geoms=arch.get('n_geoms', 8),
            n_hr_blocks=arch.get('n_hr_blocks', 1),
            dtype=compute_dtype)

    def predict(self, hr_feat: jax.Array, lr_feat: jax.Array, knn: dict) -> jax.Array:
        ctx = build_context(self, hr_feat, lr_feat, knn)
        delta = self.net(ctx['c'], ctx['scal'], knn['fm'], ctx['wall'], ctx['geom_id'])
        return ctx['baseline'] + delta

    @classmethod
    def load_knn(cls, layout: DataLayout, cfg: dict | None = None) -> dict:
        return load_hierarchical_knn(layout, cls.DEFAULT_LEVELS, cfg, tag='DAM')
