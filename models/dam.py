"""
DAM : Deterministic Attentional Mesh Network pour la super-résolution CFD.

Backbone U multi-échelles identique à FAM (VMANet), sans flow matching ni variable de temps t.

DAMNet prédit directement HR | LR, Mach, AoA sur hiérarchie de maillages, architecture :
- niveau 0 (HR)       : encodage par noeud + self-attention locale kNN
- descente l -> l+1   : cross-attention locale (requêtes = noeuds grossiers)
- bottleneck          : self-attention complète (champ réceptif global)
- remontée l+1 -> l   : cross-attention locale résiduelle + skip
- décodeur            : LayerNorm + Linear -> delta sur 4 canaux, init à zéro

Conditionnement : FiLM/AdaLN (Mach, AoA) dans chaque bloc, pas de variable de temps t
Champ LR interpolé IDW (+ gradients optionnels) injecté à chaque niveau.
Prédiction : baseline IDW au niveau HR + delta prédit par le réseau.
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

from models.base import SRModel, MLP, _ensure_knn
from utils.refs import _res_tag
from utils.layout import DataLayout
from utils.aero import wall_features as _wall_features
from preprocessing.knn import build_hierarchy as _build_hierarchy
from utils.attention import (AdaLNBlock, LocalAttnBlock, GlobalAttnBlock, PE_DIM)

_N_SCALARS = 2   # (Mach, AoA) dans hr_feat[:, 2:4]
_N_WD = 4        # features de paroi : [wd_norm, wd_exp, nx, ny]
_IDW_K = 6       # IDW standard du projet : k=6, p=2


class DAMNet(nnx.Module):
    """U multi-échelles déterministe : pred(HR | LR, Mach, AoA) sur hiérarchie de maillages.

    Identique au VMANet de FAM sans la variable de temps t :
    - pas d'embedding sinusoïdal de t
    - pas d'entrée x_t (champ bruité)
    - conditioning FiLM sur (Mach, AoA) uniquement
    """

    def __init__(self, n_levels: int, n_c: int, d: int, d_cond: int,
                 n_heads: int, n_global: int, mlp_ratio: int, rngs: nnx.Rngs,
                 use_geom_cond: bool = False, n_geoms: int = 8, dtype=None):
        # Entrée encodeur : features LR interpolées + PE + wall
        in0 = n_c + PE_DIM + _N_WD
        # Requêtes pour le pooling vers les niveaux grossiers
        q_in = PE_DIM + n_c

        self._use_geom_cond = use_geom_cond
        self.cond_mlp = MLP([_N_SCALARS, d_cond, d_cond], rngs, dtype=dtype)
        if use_geom_cond:
            self.geom_emb = nnx.Embed(n_geoms, d_cond, rngs=rngs)
        self.enc = nnx.Linear(in0, d, rngs=rngs, dtype=dtype)

        self.self_in = LocalAttnBlock(d, d, d_cond, n_heads, rngs, dtype=dtype)
        self.block_in = AdaLNBlock(d, d_cond, rngs, mlp_ratio=mlp_ratio, dtype=dtype)
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
        self.self_out = LocalAttnBlock(d, d, d_cond, n_heads, rngs, dtype=dtype)
        self.block_out = AdaLNBlock(d, d_cond, rngs, mlp_ratio=mlp_ratio, dtype=dtype)

        self.dec_norm = nnx.LayerNorm(d, rngs=rngs)
        # Init à zéro : delta = 0 au départ, prédiction = baseline IDW
        self.dec = nnx.Linear(d, 4, rngs=rngs, dtype=dtype)
        self.dec.kernel.value = jnp.zeros_like(self.dec.kernel.value)
        self.dec.bias.value = jnp.zeros_like(self.dec.bias.value)

    def __call__(self, c: list, scal: jax.Array,
                 fm: dict, wall: jax.Array, geom_id: jax.Array) -> jax.Array:
        """
        c       : liste de features LR interpolées par niveau, c[l] = [N_l, n_c]
        scal    : scalaires de conditionnement [2] = (Mach, AoA)
        fm      : dict hiérarchie kNN (pe, down, up, self, cond)
        wall    : features de paroi [N0, _N_WD]
        geom_id : entier identifiant la géométrie (utilisé si use_geom_cond)
        """
        cond = self.cond_mlp(scal)
        if self._use_geom_cond:
            cond = cond + self.geom_emb(geom_id)

        h = [None] * (len(self.down) + 1)
        h0 = self.enc(jnp.concatenate([c[0], fm['pe'][0], wall], axis=-1))
        h0 = h0 + self.self_in(h0, h0, fm['self']['idx'], fm['self']['rel'], cond)
        h[0] = self.block_in(h0, cond)

        for l in range(len(self.down)):
            q = jnp.concatenate([fm['pe'][l + 1], c[l + 1]], axis=-1)
            hl = self.down[l](q, h[l], fm['down'][l]['idx'], fm['down'][l]['rel'], cond)
            h[l + 1] = self.down_mix[l](hl, cond)

        for blk in self.global_blocks:
            h[-1] = blk(h[-1], cond)

        for l in reversed(range(len(self.up))):
            hf = h[l] + self.up[l](h[l], h[l + 1],
                                   fm['up'][l]['idx'], fm['up'][l]['rel'], cond)
            h[l] = self.up_mix[l](hf, cond)

        h0 = h[0]
        h0 = h0 + self.self_out(h0, h0, fm['self']['idx'], fm['self']['rel'], cond)
        h0 = self.block_out(h0, cond)
        return self.dec(self.dec_norm(h0))


class DAM(SRModel):
    """Deterministic Attentional Mesh — SR direct avec backbone hiérarchique FAM."""

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
        self.net = DAMNet(
            n_levels=len(self.levels),
            n_c=self.n_c,
            d=arch.get('embed_dim', 96),
            d_cond=arch.get('cond_dim', 64),
            n_heads=arch.get('n_heads', 4),
            n_global=arch.get('n_global_blocks', 3),
            mlp_ratio=arch.get('mlp_ratio', 4),
            rngs=rngs,
            use_geom_cond=self._use_geom_cond,
            n_geoms=arch.get('n_geoms', 8),
            dtype=compute_dtype)

    def _context(self, hr_feat: jax.Array, lr_feat: jax.Array, knn: dict) -> dict:
        fm = knn['fm']
        parts = [lr_feat[:, :4]]
        if self.use_lr_grad:
            parts.append(lr_feat[:, 6:9])
        feats = jnp.concatenate(parts, axis=-1)
        c = [(cl['w'][:, :, None] * feats[cl['idx']]).sum(axis=1)
             for cl in fm['cond']]
        wall = knn.get('wall_feat')
        if wall is None:
            wall = jnp.zeros((hr_feat.shape[0], _N_WD), hr_feat.dtype)
        geom_id = hr_feat[0, 4].astype(jnp.int32)
        return {'c': c, 'baseline': c[0][:, :4],
                'scal': hr_feat[0, 2:4], 'wall': wall, 'geom_id': geom_id}

    def predict(self, hr_feat: jax.Array, lr_feat: jax.Array, knn: dict) -> jax.Array:
        ctx = self._context(hr_feat, lr_feat, knn)
        delta = self.net(ctx['c'], ctx['scal'], knn['fm'], ctx['wall'], ctx['geom_id'])
        return ctx['baseline'] + delta

    @classmethod
    def load_knn(cls, layout: DataLayout, cfg: dict | None = None) -> dict:
        cfg = cfg or {}
        res = cfg.get('resolution', {})
        hr_res = res.get('hr', 0.025)
        lr_res = res.get('lr', 0.1)
        arch = cfg.get('architecture', {})
        levels = list(arch.get('levels', cls.DEFAULT_LEVELS))
        k_pool = arch.get('k_pool', 9)
        k_up = arch.get('k_up', 4)
        k_self = arch.get('k_self', 9)
        k_cond = arch.get('k_cond', _IDW_K)

        d = np.load(_ensure_knn(layout, hr_res, lr_res, max(k_cond, _IDW_K)))
        knn = {'idx': jnp.array(d['indices']),
               'rel': jnp.array(d['rel_pos']),
               'dist': jnp.array(d['dists'])}
        if 'grad_op' in d:
            knn['grad_op'] = jnp.array(d['grad_op'])
            knn['hr_idx'] = jnp.array(d['hr_idx'])

        def _mesh(r):
            return np.load(layout.mesh_path(r), allow_pickle=True).item()

        mesh_hr = _mesh(hr_res)
        pos_hr = np.asarray(mesh_hr.barycenter, dtype=np.float64)
        lr_pos = np.asarray(_mesh(lr_res).barycenter, dtype=np.float64)
        lvl_pos = [pos_hr] + [np.asarray(_mesh(r).barycenter, dtype=np.float64)
                              for r in levels]

        wd, normals = _wall_features(mesh_hr, pos_hr)
        knn['wall_feat'] = jnp.array(np.column_stack([
            wd / wd.max(),
            np.exp(-wd / (2.0 * lr_res)),
            normals,
        ]).astype(np.float32))
        knn['mesh'] = mesh_hr

        tag = '_'.join(_res_tag(r) for r in [hr_res] + levels)
        cache = (layout.knn_dir /
                 f'fm_{tag}_lr{_res_tag(lr_res)}_kp{k_pool}_ku{k_up}'
                 f'_ks{k_self}_kc{k_cond}.npz')
        if cache.exists():
            z = dict(np.load(cache))
        else:
            print(f"  [DAM] construction hiérarchie kNN -> {cache.name}")
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
