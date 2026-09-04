"""
FAMWall / DAMWall : reconstruction du champ HR complet à partir d'observations
ponctuelles de bord UNIQUEMENT (pas de champ LR volumique).

Réutilise intégralement AMNet (models/dam.py) comme backbone U-net -- code
strictement inchangé, seuls les hyperparamètres de son constructeur changent
(n_levels=1, n_global_blocks=1 par défaut : la hiérarchie garde son rôle de
cohérence spatiale entre nœuds, plus celui d'acheminement de l'info de bord
désormais assuré directement par WallCondBlock à chaque niveau).

Deux passes bien séparées à chaque forward :
  Phase A (calculée UNE FOIS, hors boucle ODE pour FAM) :
    WallEncoder  : self-attention parmi les N_wall observations de bord.
    WallCondBlock (un par niveau) : cross-attention DENSE (pas de troncature
        kNN -- N_wall est petit par construction) des nœuds du niveau vers
        tous les tokens de bord, biais de distance appris ACTIF par défaut
        (module neuf, aucune contrainte de rétrocompatibilité contrairement
        à LocalAttnBlock.learnable_range=False).
    wall_baseline : IDW(bord) mélangé au freestream selon wd_exp -- remplace
        le gather IDW figé sur maillage LR (build_context de dam.py).
  Phase B : AMNet, code inchangé, consomme c[l]=[baseline_l, lecture_l] au lieu
    de l'IDW volumique -- répétée à chaque pas ODE pour FAMWall.
"""
import math
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / 'euler') not in sys.path:
    sys.path.append(str(REPO_ROOT / 'euler'))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.base import SRModel, MLP, Buffer
from models.dam import (AMNet, load_hierarchical_knn, cfg_drop_geom_id,
                        _N_SCALARS, _N_COND, _N_WD, _LR_REF)
from utils.layout import DataLayout
from utils.coords import object_center_scale
from utils.attention import (GlobalAttnBlock, PE_DIM, fourier_pos_enc,
                             scalar_fourier_embedding)
from utils.viz._style import VAR_LABELS
from preprocessing.dataset import _MACH_MID, _MACH_SCALE, _AOA_SCALE

GAMMA = 1.4  # même constante que utils/aero.py (q_inf/cp_field)
_WALL_FEAT_DIM = 9  # [pos(2), normal(2), value(4), s(1)] -- cf. preprocessing/wall_obs.py


def _zeros_linear(din: int, dout: int, rngs: nnx.Rngs, dtype=None) -> nnx.Linear:
    lin = nnx.Linear(din, dout, rngs=rngs, dtype=dtype)
    lin.kernel.value = jnp.zeros_like(lin.kernel.value)
    lin.bias.value = jnp.zeros_like(lin.bias.value)
    return lin


class WallEncoder(nnx.Module):
    """Encode la séquence de N_wall observations de bord en tokens latents,
    via self-attention légère (GlobalAttnBlock réutilisé tel quel -- N_wall
    petit, 32-256, attention complète bon marché, capture les corrélations
    le long de tout le contour : ex. bord de fuite <-> bord d'attaque)."""

    def __init__(self, d: int, d_cond: int, n_heads: int, n_blocks: int,
                 rngs: nnx.Rngs, dtype=None):
        in_dim = 2 * PE_DIM + 2 + 4  # PE(pos) + PE_cyclique(s) + normale(2) + valeur(4)
        self.tok_in = nnx.Linear(in_dim, d, rngs=rngs, dtype=dtype)
        self.blocks = [GlobalAttnBlock(d, d_cond, n_heads, rngs, dtype=dtype)
                       for _ in range(n_blocks)]

    def __call__(self, wall_feat: jax.Array, cond: jax.Array) -> jax.Array:
        pos = wall_feat[:, 0:2]
        normal = wall_feat[:, 2:4]
        value = wall_feat[:, 4:8]
        s = wall_feat[:, 8]
        pe_pos = fourier_pos_enc(pos)
        s_cyc = jnp.stack([jnp.cos(2 * jnp.pi * s), jnp.sin(2 * jnp.pi * s)], axis=-1)
        pe_s = fourier_pos_enc(s_cyc)
        h = self.tok_in(jnp.concatenate([pe_pos, pe_s, normal, value], axis=-1))
        for blk in self.blocks:
            h = blk(h, cond)
        return h


def _dense_cross_attention(Q: jax.Array, K: jax.Array, V: jax.Array, n_heads: int,
                           dist_bias: jax.Array | None = None) -> jax.Array:
    """Cross-attention dense (pas de kNN, pas de troncature) : Q (Nq,d), K/V (Nk,d).
    dist_bias : (n_heads, Nq, Nk) optionnel, additif avant softmax."""
    Nq, d = Q.shape
    Nk = K.shape[0]
    dh = d // n_heads
    Qh = Q.reshape(Nq, n_heads, dh)
    Kh = K.reshape(Nk, n_heads, dh)
    Vh = V.reshape(Nk, n_heads, dh)
    scores = jnp.einsum('qhd,khd->hqk', Qh, Kh) / math.sqrt(dh)
    if dist_bias is not None:
        scores = scores + dist_bias
    w = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum('hqk,khd->qhd', w, Vh)
    return out.reshape(Nq, d)


class WallCondBlock(nnx.Module):
    """Cross-attention DENSE des nœuds d'un niveau vers tous les tokens de bord
    (pas de troncature kNN -- N_wall petit par construction, coût
    N_niveau×N_wall négligeable à tout niveau). Biais de distance appris
    actif par défaut (module neuf, cf. docstring module)."""

    def __init__(self, q_in_dim: int, d_wall: int, d_cond: int, n_heads: int,
                 rngs: nnx.Rngs, dtype=None, dist_bias: bool = True):
        self.q_norm = nnx.LayerNorm(q_in_dim, rngs=rngs)
        self.q_proj = nnx.Linear(q_in_dim, d_wall, rngs=rngs, dtype=dtype)
        self.gamma = _zeros_linear(d_cond, d_wall, rngs, dtype=dtype)
        self.beta = _zeros_linear(d_cond, d_wall, rngs, dtype=dtype)
        self.kv_norm = nnx.LayerNorm(d_wall, rngs=rngs)
        self.wk = nnx.Linear(d_wall, d_wall, use_bias=False, rngs=rngs, dtype=dtype)
        self.wv = nnx.Linear(d_wall, d_wall, use_bias=False, rngs=rngs, dtype=dtype)
        self.out = nnx.Linear(d_wall, 4, rngs=rngs, dtype=dtype)  # -> "primitives lues"
        self.n_heads = n_heads
        self._dist_bias = dist_bias
        if dist_bias:
            self.range_scale = nnx.Param(jnp.zeros((n_heads,), jnp.float32))

    def __call__(self, q_in: jax.Array, wall_tokens: jax.Array,
                 wall_pos_n: jax.Array, level_pos_n: jax.Array,
                 cond: jax.Array) -> jax.Array:
        q = self.q_proj(self.q_norm(q_in))
        q = q * (1.0 + self.gamma(cond)) + self.beta(cond)
        kv = self.kv_norm(wall_tokens)
        K, V = self.wk(kv), self.wv(kv)
        dist_bias = None
        if self._dist_bias:
            d = jnp.linalg.norm(level_pos_n[:, None, :] - wall_pos_n[None, :, :], axis=-1)
            scale = jax.nn.softplus(self.range_scale.value)  # (n_heads,)
            dist_bias = -scale[:, None, None] * d[None, :, :]  # (n_heads, Nq, Nk)
        h = _dense_cross_attention(q, K, V, self.n_heads, dist_bias)
        return self.out(h)


class _WallCond(nnx.Module):
    """Conditionnement de la Phase A (WallEncoder/WallCondBlock) -- même formule
    que AMNet._cond (Fourier(Mach,AoA)+log(res)[+geom]) mais paramètres PROPRES,
    indépendants du cond interne d'AMNet (Phase B) : pas de couplage entre les
    deux passes, pas de dépendance à t (la lecture du bord ne dépend pas du pas
    ODE -- garantie de portée architecturale, pas itérative)."""

    def __init__(self, d_cond: int, rngs: nnx.Rngs, use_geom_cond: bool = False,
                n_geoms: int = 8, dtype=None):
        self.mlp = MLP([_N_COND, d_cond, d_cond], rngs, dtype=dtype)
        self._use_geom_cond = use_geom_cond
        if use_geom_cond:
            self.geom_emb = nnx.Embed(n_geoms + 1, d_cond, rngs=rngs)

    def __call__(self, scal: jax.Array, geom_id: jax.Array) -> jax.Array:
        feat = jnp.concatenate([scalar_fourier_embedding(scal[:_N_SCALARS]), scal[_N_SCALARS:]])
        cond = self.mlp(feat)
        if self._use_geom_cond:
            cond = cond + self.geom_emb(geom_id)
        return cond


def wall_baseline(query_pos_n: jax.Array, wall_pos_n: jax.Array, wall_val_n: jax.Array,
                  wd_exp: jax.Array, freestream_n: jax.Array, k: int = 6) -> jax.Array:
    """baseline(x) = wd_exp(x)*IDW_wall(x) + (1-wd_exp(x))*freestream.
    wall_val_n/freestream_n déjà normalisés (mu,sig) -- cohérence avec la cible."""
    dist = jnp.linalg.norm(query_pos_n[:, None, :] - wall_pos_n[None, :, :], axis=-1)
    w = 1.0 / (dist ** 2 + 1e-6)
    k_eff = min(k, wall_pos_n.shape[0])
    topk_w, topk_idx = jax.lax.top_k(w, k_eff)
    topk_w = topk_w / topk_w.sum(axis=-1, keepdims=True)
    idw = (topk_w[:, :, None] * wall_val_n[topk_idx]).sum(axis=1)
    return wd_exp[:, None] * idw + (1.0 - wd_exp[:, None]) * freestream_n[None, :]


def _freestream_normalized(scal: jax.Array, mu: jax.Array, sig: jax.Array,
                           mach_mid: jax.Array = _MACH_MID,
                           mach_scale: jax.Array = _MACH_SCALE) -> jax.Array:
    """État freestream uniforme déduit de Mach/AoA connus (aucune vérité terrain
    requise), normalisé par les mêmes mu/sig que le reste (rho,u,v,p).

    mach_mid/mach_scale : conditionnement Mach de CE checkpoint (buffers
    model.mach_mid/mach_scale, persistés dans le .pkl -- cf. _build_backbone),
    repli sur l'historique (0.7, 3.0) par défaut."""
    mach_in = scal[0] * mach_scale + mach_mid
    aoa_in = scal[1] * _AOA_SCALE
    c_inf = jnp.sqrt(GAMMA)
    u_inf = mach_in * c_inf * jnp.cos(jnp.deg2rad(aoa_in))
    v_inf = mach_in * c_inf * jnp.sin(jnp.deg2rad(aoa_in))
    freestream = jnp.array([1.0, u_inf, v_inf, 1.0], dtype=jnp.float32)
    return (freestream - mu) / sig


def load_hierarchical_knn_wall(layout: DataLayout, default_levels: list,
                               cfg: dict | None = None, tag: str = 'FAMWall') -> dict:
    """Enveloppe autour de load_hierarchical_knn (models/dam.py, INCHANGÉ) :
    ajoute les positions normalisées par niveau (lvl_pos_n), nécessaires au
    biais de distance de WallCondBlock. coord_norm='object' imposé (positions
    de bord sans repère de domaine LR/HR pertinent).
    """
    cfg = cfg or {}
    arch = cfg.get('architecture', {})
    coord_norm = arch.get('coord_norm', 'object')
    if coord_norm != 'object':
        raise ValueError("FAMWall impose architecture.coord_norm='object' "
                         "(positions de bord).")
    knn = load_hierarchical_knn(layout, default_levels, cfg, tag=tag)
    levels = list(arch.get('levels', default_levels))
    mesh_hr = knn['mesh']
    ctr, scl = object_center_scale(mesh_hr.metadata)
    lvl_res = [layout.hr_res] + levels
    lvl_pos_n = []
    for r in lvl_res:
        m = np.load(layout.mesh_path(r), allow_pickle=True).item()
        pos = np.asarray(m.barycenter, dtype=np.float64)
        lvl_pos_n.append(jnp.array(((pos - ctr) / scl).astype(np.float32)))
    knn['lvl_pos_n'] = lvl_pos_n
    return knn


def build_context_wall(model, hr_feat: jax.Array, wall_feat: jax.Array, knn: dict) -> dict:
    """Construit le contexte FAMWall/DAMWall : c[l] = [baseline_l, lecture_l]
    pour chaque niveau, à partir des seules observations de bord -- remplace
    build_context (models/dam.py, IDW volumique) sans changer la forme d'appel
    vers AMNet.__call__ (inchangé)."""
    fm = knn['fm']
    wall_geom = knn.get('wall_feat')  # features géométriques par nœud/niveau (wd_norm,wd_exp,nx,ny)
    geom_id = hr_feat[0, 4].astype(jnp.int32)
    res_scalar = knn.get('res_scalar', jnp.zeros((1,), hr_feat.dtype))
    scal = jnp.concatenate([hr_feat[0, 2:4], res_scalar])

    cond = model.wall_cond(scal, geom_id)
    wall_tokens = model.wall_encoder(wall_feat, cond)

    wall_pos_n = wall_feat[:, 0:2]
    wall_val_n = wall_feat[:, 4:8]
    freestream_n = _freestream_normalized(scal, model.mu_train.value, model.sig_train.value,
                                          model.mach_mid.value, model.mach_scale.value)

    c = []
    for l, (pe_l, wgeom_l, pos_n_l, block) in enumerate(
            zip(fm['pe'], wall_geom, knn['lvl_pos_n'], model.wall_cond_blocks)):
        wd_exp_l = wgeom_l[:, 1]
        baseline_l = wall_baseline(pos_n_l, wall_pos_n, wall_val_n, wd_exp_l, freestream_n)
        q_in = jnp.concatenate([pe_l, wgeom_l], axis=-1)
        read_l = block(q_in, wall_tokens, wall_pos_n, pos_n_l, cond)
        c.append(jnp.concatenate([baseline_l, read_l], axis=-1))

    return {'c': c, 'baseline': c[0][:, :4], 'scal': scal,
           'wall': wall_geom, 'geom_id': geom_id}


class _WallBackboneMixin:
    """Construction partagée DAMWall/FAMWall : WallEncoder + WallCondBlock par
    niveau + AMNet (backbone inchangé, hyperparamètres redimensionnés)."""

    DEFAULT_LEVELS = [0.1]  # n_levels=1 par défaut (AMNet réduit)

    def _build_backbone(self, rngs: nnx.Rngs, cfg: dict, d_t: int, n_x: int) -> None:
        arch = cfg.get('architecture', {})
        self.levels = list(arch.get('levels', self.DEFAULT_LEVELS))
        n_c = 8  # baseline(4) + lecture WallCondBlock(4) -- pas de use_lr_grad (pas de champ LR)

        use_bf16 = arch.get('use_bf16', False)
        compute_dtype = jnp.bfloat16 if use_bf16 else None

        self._use_geom_cond = arch.get('use_geom_cond', False)
        self.geom_cfg_prob = float(arch.get('geom_cfg_prob', 0.0))
        self._n_geoms = arch.get('n_geoms', 8)
        d_cond = arch.get('cond_dim', 128)
        d_wall = arch.get('wall_embed_dim', d_cond)
        n_heads = arch.get('n_heads', 4)
        dist_bias = arch.get('wall_dist_bias', True)

        # Conditionnement Mach de ce checkpoint
        _mn = cfg.get('mach_norm')
        _mach_mid, _mach_scale = (((_mn[0] + _mn[1]) / 2, (_mn[1] - _mn[0]) / 2)
                                  if _mn is not None else (_MACH_MID, _MACH_SCALE))
        self.mach_mid = Buffer(jnp.array(_mach_mid, jnp.float32))
        self.mach_scale = Buffer(jnp.array(_mach_scale, jnp.float32))

        self.mu_train = Buffer(jnp.zeros((4,), jnp.float32))
        self.sig_train = Buffer(jnp.ones((4,), jnp.float32))

        self.wall_cond = _WallCond(d_cond, rngs, self._use_geom_cond, self._n_geoms, compute_dtype)
        self.wall_encoder = WallEncoder(d_wall, d_cond, n_heads,
                                        arch.get('n_wall_self_blocks', 2), rngs, compute_dtype)
        n_levels_total = len(self.levels) + 1  # HR + niveaux grossiers
        self.wall_cond_blocks = [
            WallCondBlock(PE_DIM + _N_WD, d_wall, d_cond, n_heads, rngs, compute_dtype, dist_bias)
            for _ in range(n_levels_total)
        ]

        self.net = AMNet(
            n_levels=len(self.levels), n_c=n_c,
            d=arch.get('embed_dim', 96), d_cond=d_cond, n_heads=n_heads,
            n_global=arch.get('n_global_blocks', 1),  # décision tranchée : 1 par défaut
            mlp_ratio=arch.get('mlp_ratio', 4), rngs=rngs,
            d_t=d_t, n_x=n_x,
            use_geom_cond=self._use_geom_cond, n_geoms=self._n_geoms,
            n_hr_blocks=arch.get('n_hr_blocks', 2), dtype=compute_dtype)

    @classmethod
    def load_knn(cls, layout: DataLayout, cfg: dict | None = None) -> dict:
        return load_hierarchical_knn_wall(layout, cls.DEFAULT_LEVELS, cfg, tag=cls.__name__)


class DAMWall(_WallBackboneMixin, SRModel):
    """Reconstruction déterministe (delta direct) à partir d'observations de
    bord uniquement, modèle de contrôle pour FAMWall."""

    def __init__(self, rngs: nnx.Rngs, cfg: dict | None = None):
        cfg = cfg or {}
        self._build_backbone(rngs, cfg, d_t=0, n_x=0)

    def _forward(self, ctx: dict, knn: dict) -> jax.Array:
        delta = self.net(ctx['c'], ctx['scal'], knn['fm'], ctx['wall'], ctx['geom_id'])
        return ctx['baseline'] + delta

    def predict(self, hr_feat: jax.Array, wall_feat: jax.Array, knn: dict) -> jax.Array:
        ctx = build_context_wall(self, hr_feat, wall_feat, knn)
        return self._forward(ctx, knn)

    def _sample_loss(self, hr: jax.Array, wall: jax.Array, tg: jax.Array,
                     wt: jax.Array, gp: jax.Array, knn_g: dict,
                     key: jax.Array, aux: dict) -> jax.Array:
        ctx = build_context_wall(self, hr, wall, knn_g)
        if self.geom_cfg_prob > 0:
            ctx = dict(ctx)
            ctx['geom_id'] = cfg_drop_geom_id(ctx['geom_id'], self._n_geoms,
                                              self.geom_cfg_prob, key)
        pred = self._forward(ctx, knn_g)
        return aux['loss_fn'](pred, tg, wt) + self._phys_terms(pred, hr, gp, knn_g, aux)


class FAMWall(_WallBackboneMixin, SRModel):
    """Flow matching conditionnel sur observations de bord uniquement --
    reprend fidèlement la mécanique de FAM (models/fam.py : bruit->résidu,
    intégration ODE Heun, CFG, res_scale/learned_res_scale) en remplaçant
    build_context par build_context_wall."""

    def __init__(self, rngs: nnx.Rngs, cfg: dict | None = None):
        cfg = cfg or {}
        arch = cfg.get('architecture', {})
        flow = cfg.get('flow', {})
        self._build_backbone(rngs, cfg, d_t=arch.get('t_dim', 64), n_x=4)

        self.n_steps = flow.get('n_steps', 16)
        self.use_residual = flow.get('use_residual', True)
        self.n_samples = flow.get('n_samples', 1)
        self.n_val_gen = flow.get('n_val_gen', 16)
        self.sample_seed = flow.get('sample_seed', 0)
        self.cfg_prob = float(flow.get('cfg_prob', 0.0))
        self.cfg_scale = float(flow.get('cfg_scale', 1.0))
        self.geom_cfg_scale = float(flow.get('geom_cfg_scale', 1.0))
        self._t_dist = flow.get('t_dist', 'uniform')
        self._sigma_t = float(flow.get('sigma_t', 1.0))
        self._t_alpha = float(flow.get('t_alpha', 2.0))
        self._t_beta = float(flow.get('t_beta', 2.0))
        self.val_max_samples = self.n_val_gen

        self.res_scale = Buffer(jnp.ones((4,), jnp.float32))
        self._nominal_lr = float((cfg.get('resolution') or {}).get('lr', _LR_REF))

        # learned_res_scale : quasi obligatoire pour FAMWall (HR indisponible au
        # déploiement réel)
        self._learned_res_scale = bool(flow.get('learned_res_scale', True))
        self.lambda_res_scale = float(flow.get('lambda_res_scale', 0.1))
        if self._learned_res_scale:
            scale_hidden = arch.get('scale_head_dim', 64)
            self.scale_mlp = MLP([_N_COND, scale_hidden, 4], rngs)
            self.scale_mlp.layers[-1].kernel.value = jnp.zeros_like(self.scale_mlp.layers[-1].kernel.value)
            self.scale_mlp.layers[-1].bias.value = jnp.zeros_like(self.scale_mlp.layers[-1].bias.value)
            if self._use_geom_cond:
                self.scale_geom_emb = nnx.Embed(self._n_geoms + 1, 4, rngs=rngs)
                self.scale_geom_emb.embedding.value = jnp.zeros_like(self.scale_geom_emb.embedding.value)
            self.log_scale_bias = Buffer(jnp.zeros((4,), jnp.float32))

    def velocity(self, x_t: jax.Array, t: jax.Array, ctx: dict, knn: dict,
                scal: jax.Array | None = None, geom_id: jax.Array | None = None) -> jax.Array:
        s = ctx['scal'] if scal is None else scal
        g = ctx['geom_id'] if geom_id is None else geom_id
        return self.net(ctx['c'], s, knn['fm'], ctx['wall'], g, x_t=x_t, t=t)

    def _predicted_res_scale(self, ctx: dict) -> jax.Array:
        scal = ctx['scal']
        feat = jnp.concatenate([scalar_fourier_embedding(scal[:_N_SCALARS]), scal[_N_SCALARS:]])
        log_scale = self.log_scale_bias.value + self.scale_mlp(feat)
        if self._use_geom_cond:
            log_scale = log_scale + self.scale_geom_emb(ctx['geom_id'])
        return jnp.exp(log_scale)

    def _res_scale(self, knn: dict, ctx: dict | None = None) -> jax.Array:
        if self._learned_res_scale and ctx is not None:
            return self._predicted_res_scale(ctx)
        rs = knn.get('res_scale')
        return self.res_scale.value if rs is None else rs

    def _project(self, field: jax.Array) -> jax.Array:
        prim = field * self.sig_train.value + self.mu_train.value
        prim = prim.at[:, 0].set(jnp.maximum(prim[:, 0], 1e-3))
        prim = prim.at[:, 3].set(jnp.maximum(prim[:, 3], 1e-3))
        return (prim - self.mu_train.value) / self.sig_train.value

    def _guided_velocity(self, x_t: jax.Array, t: jax.Array, ctx: dict, knn: dict) -> jax.Array:
        if self.cfg_scale == 1.0 and self.geom_cfg_scale == 1.0:
            return self.velocity(x_t, t, ctx, knn)
        v_cond = self.velocity(x_t, t, ctx, knn)
        v = v_cond
        if self.cfg_scale != 1.0:
            scal_null = jnp.zeros_like(ctx['scal'])
            v_uncond = self.velocity(x_t, t, ctx, knn, scal=scal_null)
            v = v + (self.cfg_scale - 1.0) * (v_cond - v_uncond)
        if self.geom_cfg_scale != 1.0:
            geom_null = jnp.full_like(ctx['geom_id'], self._n_geoms)
            v_uncond_geom = self.velocity(x_t, t, ctx, knn, geom_id=geom_null)
            v = v + (self.geom_cfg_scale - 1.0) * (v_cond - v_uncond_geom)
        return v

    def _integrate(self, x0: jax.Array, ctx: dict, knn: dict) -> jax.Array:
        # ctx (donc la lecture du bord, Phase A) est construit UNE FOIS avant cette
        # boucle par sample()
        dt = 1.0 / self.n_steps

        def body(k, x):
            t1 = k * dt
            v1 = self._guided_velocity(x, t1, ctx, knn)
            v2 = self._guided_velocity(x + dt * v1, t1 + dt, ctx, knn)
            return x + 0.5 * dt * (v1 + v2)

        return jax.lax.fori_loop(0, self.n_steps, body, x0)

    def sample(self, hr_feat: jax.Array, wall_feat: jax.Array, knn: dict,
              key: jax.Array | None = None, n_samples: int | None = None) -> jax.Array:
        ctx = build_context_wall(self, hr_feat, wall_feat, knn)
        rs = jax.lax.stop_gradient(self._res_scale(knn, ctx))
        n_s = n_samples if n_samples is not None else self.n_samples
        key = key if key is not None else jax.random.PRNGKey(self.sample_seed)
        keys = jax.random.split(key, n_s)

        def one(k):
            x0 = jax.random.normal(k, (hr_feat.shape[0], 4))
            r = self._integrate(x0, ctx, knn)
            field = ctx['baseline'] + r * rs if self.use_residual else r * rs
            return self._project(field)

        return jax.vmap(one)(keys)

    def predict(self, hr_feat: jax.Array, wall_feat: jax.Array, knn: dict) -> jax.Array:
        return self.sample(hr_feat, wall_feat, knn).mean(axis=0)

    def _sample_t(self, key: jax.Array) -> jax.Array:
        if self._t_dist == 'logit_normal':
            return jax.nn.sigmoid(jax.random.normal(key) * self._sigma_t)
        if self._t_dist == 'beta':
            return jax.random.beta(key, self._t_alpha, self._t_beta)
        return jax.random.uniform(key)

    def _branch_res_std(self, ds_b, knn_b) -> np.ndarray:
        n = len(ds_b)
        sub = np.unique(np.linspace(0, n - 1, min(256, n)).astype(int))
        acc = []
        for i in sub:
            hr, wall, tg, *_ = ds_b[int(i)]
            hr, wall, tg = np.asarray(hr), np.asarray(wall), np.asarray(tg)
            ctx = build_context_wall(self, jnp.array(hr), jnp.array(wall), knn_b)
            baseline = np.asarray(ctx['baseline'])
            acc.append(tg - baseline if self.use_residual else tg)
        return np.maximum(np.concatenate(acc, 0).std(0), 1e-6)

    def _pre_fit(self, branch_pairs: list, cfg) -> None:
        _label = "std résiduel" if self.use_residual else "std cible"
        multi = len(branch_pairs) > 1
        for bi, (ds_b, knn_b) in enumerate(branch_pairs):
            res_std = self._branch_res_std(ds_b, knn_b)
            knn_b['res_scale'] = jnp.array(res_std, jnp.float32)
            print(f"  res_scale ({_label}) : "
                  + "  ".join(f"{v}={res_std[i]:.4f}" for i, v in enumerate(VAR_LABELS)))
        self.res_scale.value = jnp.array(branch_pairs[0][1]['res_scale'])
        if self._learned_res_scale:
            self.log_scale_bias.value = jnp.log(jnp.maximum(self.res_scale.value, 1e-6))
            print(f"  res_scale APPRISE (learned_res_scale=True, lambda={self.lambda_res_scale:g}) "
                  "-- tête initialisée sur le buffer ci-dessus, affinée pendant l'entraînement.")

    def _sample_loss(self, hr: jax.Array, wall: jax.Array, tg: jax.Array, wt: jax.Array,
                gp: jax.Array, knn_g: dict, key: jax.Array, aux: dict) -> jax.Array:
        ctx = build_context_wall(self, hr, wall, knn_g)
        rs_raw = self._res_scale(knn_g, ctx)
        rs = jax.lax.stop_gradient(rs_raw)
        r = ((tg - ctx['baseline']) / rs if self.use_residual else tg / rs)
        kt, kx, kcfg, kgeom = jax.random.split(key, 4)
        t = self._sample_t(kt)
        x0 = jax.random.normal(kx, r.shape)
        x_t = (1.0 - t) * x0 + t * r
        if self.cfg_prob > 0:
            drop = jax.random.bernoulli(kcfg, self.cfg_prob)
            scal_used = jnp.where(drop, jnp.zeros_like(ctx['scal']), ctx['scal'])
        else:
            scal_used = ctx['scal']
        if self.geom_cfg_prob > 0:
            ctx = dict(ctx)
            ctx['geom_id'] = cfg_drop_geom_id(ctx['geom_id'], self._n_geoms,
                                              self.geom_cfg_prob, kgeom)
        v = self.velocity(x_t, t, ctx, knn_g, scal=scal_used)
        loss = aux['loss_fn'](v, r - x0, wt)
        if self._learned_res_scale and self.lambda_res_scale > 0:
            # Loss auxiliaire : régresse la prédiction de res_scale vers le std réel du
            # résidu de CE cas, cible calculée sur le vrai geom_id. rs_pred reprend le ctx
            # d'APRÈS le dropout CFG, sans quoi la ligne du token nul de scale_geom_emb
            # ne recevrait jamais de gradient alors que c'est elle qui sert en OOD.
            rs_pred = self._predicted_res_scale(ctx) if self.geom_cfg_prob > 0 else rs_raw
            true_std = jnp.maximum((r * rs).std(axis=0), 1e-6)
            loss = loss + self.lambda_res_scale * jnp.mean(
                (jnp.log(rs_pred) - jnp.log(true_std)) ** 2)
        if aux['lambda_phys'] > 0 or aux['lambda_enthalpy'] > 0 or aux.get('use_endpoint', False):
            r_hat = x_t + (1.0 - t) * v
            if aux.get('use_endpoint', False):
                loss = loss + aux['lambda_endpoint'] * aux['loss_fn'](r_hat, r, wt)
            if aux['lambda_phys'] > 0 or aux['lambda_enthalpy'] > 0:
                field = (ctx['baseline'] + r_hat * rs if self.use_residual else r_hat * rs)
                loss = loss + self._phys_terms(field, hr, gp, knn_g, aux)
        return loss
