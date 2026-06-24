"""
FAM : Flow-based Attention Mesh (Model) :
Flow matching conditionnel en espace complet pour la super-résolution CFD.

Au lieu de prédire directement HR (déterministe) ou de diffuser en latent,
on apprend un champ de vitesse v(x_t, t | cond) qui transporte du bruit
gaussien vers le résidu r = HR - IDW(LR) (normalisé par canal).
    x_t = (1-t)*x0 + t*r,  v* = r - x0,  x0 ~ N(0, I)
L'échantillonnage intègre l'ODE dx/dt = v par schéma de Heun (n_steps pas).
La prédiction finale est baseline IDW + r_hat * res_scale.

Avantage sur la diffusion pour ce genre de problèmes :
- Moins de pas d'intégration (typiquement 1000 pour DDPM a 50/100 pour DDIM) : inférence plus rapide
- L'approche résiduelle rend la trajectoire très courte et plus stable
- Plus naturel pour conditionner la super-résolution

La diffusion modélise la distribution en prédisant le bruit ajouté, ou bien le score
Sans conditionnement, elle est efficace pour générer des images réalistes, mais pour la super-résolution CFD,
Le flow apprend directement les trajectoires de transport vers la distribution cible, plus efficace pour la SR conditionnelle

Backbone U multi-échelles sur hiérarchie de maillages :
- niveau 0 (HR)       : encodage par noeud + self-attention locale kNN
- descente l -> l+1   : cross-attention locale (requêtes = noeuds grossiers)
- bottleneck          : self-attention complète (champ réceptif global)
- remontée l+1 -> l   : cross-attention locale résiduelle + skip
- décodeur            : LayerNorm + Linear -> vitesse (4 canaux), init à zéro
Conditionnement : FiLM (t, Mach, AoA) dans chaque bloc ; champ LR interpolé
IDW (+ gradients) injecté à chaque niveau
Même architecture que DAM, mais avec variable de temps t et intégration ODE pour l'échantillonnage.
"""

import csv
import sys
import time
from pathlib import Path

import yaml

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from preprocessing.dataset import MultiSRDataset
from models.base import (SRModel, MLP, Buffer, TrainConfig, n_params,
                         _ensure_knn, _rel_l2_np, eval_idw, _get_mem)
from loss import LOSS_FNS
from utils.refs import _res_tag
from utils.layout import DataLayout
from utils.aero import wall_features as _wall_features
from preprocessing.knn import build_hierarchy as _build_hierarchy
from utils.attention import (sinusoidal_embedding, AdaLNBlock,
                              LocalAttnBlock, GlobalAttnBlock, PE_DIM)
from utils.metrics import l2_rel, enthalpy_rms
from utils.refs import to_mach
from utils.viz._style import VAR_LABELS
from utils.viz.training import plot_training_curves

_N_SCALARS = 2   # (Mach, AoA) dans hr_feat[:, 2:4]
_N_WD = 4        # features de paroi : [wd_norm, wd_exp, nx, ny]
_IDW_K = 6       # IDW standard du projet : k=6, p=2


class VMANet(nnx.Module):
    """U multi-échelles : v(x_t, t | cond) sur la hiérarchie de maillages"""
    def __init__(self, n_levels: int, n_c: int, d: int, d_cond: int, d_t: int,
                 n_heads: int, n_global: int, mlp_ratio: int, rngs: nnx.Rngs,
                 use_geom_cond: bool = False, n_geoms: int = 8, dtype=None):
        in0 = 4 + n_c + PE_DIM + _N_WD
        q_in = PE_DIM + n_c
        self.d_t = d_t

        self._use_geom_cond = use_geom_cond
        self.cond_mlp = MLP([d_t + _N_SCALARS, d_cond, d_cond], rngs, dtype=dtype)
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
        # Décodeur init à zéro : v = 0 au départ, l'ODE part de la baseline IDW
        self.dec = nnx.Linear(d, 4, rngs=rngs, dtype=dtype)
        self.dec.kernel.value = jnp.zeros_like(self.dec.kernel.value)
        self.dec.bias.value = jnp.zeros_like(self.dec.bias.value)

    def __call__(self, x_t: jax.Array, t: jax.Array, c: list, scal: jax.Array,
                 fm: dict, wall: jax.Array, geom_id: jax.Array) -> jax.Array:
        cond = self.cond_mlp(jnp.concatenate(
            [sinusoidal_embedding(t, self.d_t), scal]))
        if self._use_geom_cond:
            cond = cond + self.geom_emb(geom_id)

        h = [None] * (len(self.down) + 1)
        h0 = self.enc(jnp.concatenate([x_t, c[0], fm['pe'][0], wall], axis=-1))
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


class FAM(SRModel):
    """Flow matching conditionnel multi-échelles pour la super-résolution CFD :
    v(x_t, t | cond) sur hiérarchie de maillages, intégration ODE par Heun.
    """

    # Multi echelles de maillages pour le réserau
    DEFAULT_LEVELS = [0.05, 0.1]

    def __init__(self, rngs: nnx.Rngs, cfg: dict | None = None):
        cfg = cfg or {}
        arch = cfg.get('architecture', {})
        flow = cfg.get('flow', {})

        self.levels = list(arch.get('levels', self.DEFAULT_LEVELS))
        self.use_lr_grad = arch.get('use_lr_grad', True)
        self.n_c = 4 + (3 if self.use_lr_grad else 0)

        use_bf16 = arch.get('use_bf16', False)
        compute_dtype = jnp.bfloat16 if use_bf16 else None

        self._use_geom_cond = arch.get('use_geom_cond', False)
        self.net = VMANet(
            n_levels=len(self.levels),
            n_c=self.n_c,
            d=arch.get('embed_dim', 96),
            d_cond=arch.get('cond_dim', 128),
            d_t=arch.get('t_dim', 64),
            n_heads=arch.get('n_heads', 4),
            n_global=arch.get('n_global_blocks', 3),
            mlp_ratio=arch.get('mlp_ratio', 2),
            rngs=rngs,
            use_geom_cond=self._use_geom_cond,
            n_geoms=arch.get('n_geoms', 8),
            dtype=compute_dtype)

        self.n_steps = flow.get('n_steps', 16)
        self.use_residual = flow.get('use_residual', True)
        self.n_samples = flow.get('n_samples', 1)
        self.ema_decay = flow.get('ema_decay', 0.999)
        self.n_val_gen = flow.get('n_val_gen', 16)
        self.sample_seed = flow.get('sample_seed', 0)

        # Échelle par canal du résiduel (calculée dans fit, sauvée avec l'état)
        self.res_scale = Buffer(jnp.ones((4,), jnp.float32))

    def _context(self, hr_feat: jax.Array, lr_feat: jax.Array, knn: dict) -> dict:
        fm = knn['fm']
        parts = [lr_feat[:, :4]]
        if self.use_lr_grad:
            parts.append(lr_feat[:, 6:9])    # grad_p (2) + div_u (1)
        feats = jnp.concatenate(parts, axis=-1)
        c = [(cl['w'][:, :, None] * feats[cl['idx']]).sum(axis=1)
             for cl in fm['cond']]
        wall = knn.get('wall_feat')
        if wall is None:
            wall = jnp.zeros((hr_feat.shape[0], _N_WD), hr_feat.dtype)
        geom_id = hr_feat[0, 4].astype(jnp.int32)
        return {'c': c, 'baseline': c[0][:, :4],
                'scal': hr_feat[0, 2:4], 'wall': wall, 'geom_id': geom_id}

    def velocity(self, x_t: jax.Array, t: jax.Array, ctx: dict, knn: dict) -> jax.Array:
        return self.net(x_t, t, ctx['c'], ctx['scal'], knn['fm'], ctx['wall'], ctx['geom_id'])

    def _integrate(self, x0: jax.Array, ctx: dict, knn: dict) -> jax.Array:
        dt = 1.0 / self.n_steps

        def body(k, x):
            t1 = k * dt
            v1 = self.velocity(x, t1, ctx, knn)
            v2 = self.velocity(x + dt * v1, t1 + dt, ctx, knn)
            return x + 0.5 * dt * (v1 + v2)

        return jax.lax.fori_loop(0, self.n_steps, body, x0)

    def sample(self, hr_feat: jax.Array, lr_feat: jax.Array, knn: dict,
               key: jax.Array | None = None, n_samples: int | None = None) -> jax.Array:
        """Tire n_samples champs HR normalisés [S, N, 4] (ensemble / UQ)."""
        ctx = self._context(hr_feat, lr_feat, knn)
        n_s = n_samples if n_samples is not None else self.n_samples
        key = key if key is not None else jax.random.PRNGKey(self.sample_seed)
        keys = jax.random.split(key, n_s)

        def one(k):
            x0 = jax.random.normal(k, (hr_feat.shape[0], 4))
            r = self._integrate(x0, ctx, knn)
            if self.use_residual:
                return ctx['baseline'] + r * self.res_scale.value
            else:
                return r * self.res_scale.value

        return jax.vmap(one)(keys)

    def predict(self, hr_feat: jax.Array, lr_feat: jax.Array, knn: dict) -> jax.Array:
        # Moyenne des n samples pour la préd finale
        return self.sample(hr_feat, lr_feat, knn).mean(axis=0)

    def recalibrate_res_scale(self, lr_feats: list, tg_list: list, knn: dict) -> None:
        """Recalibre res_scale pour une résolution LR différente de l'entraînement.

        lr_feats : liste de tableaux (N_LR, >=4),primitives LR normalisées
        tg_list  : liste de tableaux (N_HR, 4), primitives HR normalisées
        knn      : dict kNN construit avec la nouvelle résolution LR
        """
        w0 = np.asarray(knn['fm']['cond'][0]['w'])
        i0 = np.asarray(knn['fm']['cond'][0]['idx'])
        acc = []
        for lr, tg in zip(lr_feats, tg_list):
            if self.use_residual:
                baseline = (w0[:, :, None] * np.asarray(lr)[:, :4][i0]).sum(axis=1)
                acc.append(np.asarray(tg) - baseline)
            else:
                acc.append(np.asarray(tg))
        res_std = np.maximum(np.concatenate(acc, axis=0).std(0), 1e-6)
        self.res_scale.value = jnp.array(res_std, jnp.float32)
        print('  res_scale recalibrée : '
              + '  '.join(f'{v}={res_std[i]:.4f}' for i, v in enumerate(VAR_LABELS)))

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
        lvl_pos = [pos_hr] + [np.asarray(_mesh(r).barycenter, dtype=np.float64) for r in levels]

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
            print(f"  [FM] construction hiérarchie kNN -> {cache.name}")
            z = _build_hierarchy(lvl_pos, lr_pos, k_pool, k_up, k_self, k_cond)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, **z)

        L = len(levels) + 1
        knn['fm'] = {
            'pe': [jnp.array(z[f'pe{l}']) for l in range(L)],
            'cond': [{'idx': jnp.array(z[f'cond{l}_idx']), 'w': jnp.array(z[f'cond{l}_w'])} for l in range(L)],
            'down': [{'idx': jnp.array(z[f'down{l}_idx']), 'rel': jnp.array(z[f'down{l}_rel'])} for l in range(L - 1)],
            'up': [{'idx': jnp.array(z[f'up{l}_idx']), 'rel': jnp.array(z[f'up{l}_rel'])} for l in range(L - 1)],
            'self': {'idx': jnp.array(z['self_idx']), 'rel': jnp.array(z['self_rel'])},
        }
        return knn

    def _with_params(self, params) -> 'FAM':
        graphdef, _, rest = nnx.split(self, nnx.Param, ...)
        return nnx.merge(graphdef, params, rest)

    def fit(self, train_ds, val_ds, knn, cfg: TrainConfig,
            out_dir: str | Path = 'results/checkpoints',
            model_cfg: dict | None = None, run_name: str | None = None,
            pred_callback=None) -> 'FAM':
        _is_multi = isinstance(train_ds, MultiSRDataset)
        if _is_multi:
            _all_knns  = knn
            _all_names = train_ds.names
            _all_val   = val_ds.datasets if isinstance(val_ds, MultiSRDataset) else [val_ds]
            _primary_knn = _all_knns[_all_names[0]]
            _primary_train = train_ds.datasets[0]
        else:
            _all_knns    = {'primary': knn}
            _all_names   = ['primary']
            _all_val     = [val_ds]
            _primary_knn = knn
            _primary_train = train_ds

        if run_name is None:
            run_name = self.__class__.__name__
        out_dir = Path(out_dir) / run_name
        out_dir.mkdir(parents=True, exist_ok=True)

        sep = '─' * 52
        print(f'\n{sep}')
        print(f'  Run : {run_name}')
        print(f'  Out : {out_dir}')
        if model_cfg:
            for line in yaml.dump(model_cfg, default_flow_style=False,
                                  sort_keys=False).rstrip().splitlines():
                print(f'  {line}')
        print(f'  ·  n_train={len(train_ds)}  n_val={len(val_ds)}'
              f'  params={n_params(self):,}')
        print(sep)

        # Échelle du résiduel : std par canal sur un sous-ensemble train (dataset primaire)
        w0 = np.asarray(_primary_knn['fm']['cond'][0]['w'])
        i0 = np.asarray(_primary_knn['fm']['cond'][0]['idx'])
        sub = np.unique(np.linspace(0, len(_primary_train) - 1,
                                    min(256, len(_primary_train))).astype(int))
        acc = []
        for i in sub:
            _, lr, tg, *_ = _primary_train[int(i)]
            if self.use_residual:
                baseline = (w0[:, :, None] * lr[:, :4][i0]).sum(axis=1)
                acc.append(tg - baseline)
            else:
                acc.append(tg)
        res_std = np.maximum(np.concatenate(acc, 0).std(0), 1e-6)
        self.res_scale.value = jnp.array(res_std, jnp.float32)
        _scale_label = "std résiduel" if self.use_residual else "std cible"
        print(f"  res_scale ({_scale_label}) : "
              + "  ".join(f"{v}={res_std[i]:.4f}" for i, v in enumerate(VAR_LABELS)))

        batch_size = max(cfg.batch_size, 1)
        n_batches = (len(train_ds) + batch_size - 1) // batch_size
        n_steps = cfg.epochs * n_batches
        warmup_steps = cfg.warmup_epochs * n_batches
        if cfg.schedule == 'cosine':
            if warmup_steps > 0:
                lr_sched = optax.warmup_cosine_decay_schedule(
                    init_value=0.0, peak_value=cfg.lr,
                    warmup_steps=warmup_steps, decay_steps=n_steps,
                )
            else:
                lr_sched = optax.cosine_decay_schedule(cfg.lr, n_steps)
        else:
            lr_sched = cfg.lr
        tx = optax.adamw(lr_sched, weight_decay=cfg.weight_decay)
        if cfg.grad_clip > 0:
            tx = optax.chain(optax.clip_by_global_norm(cfg.grad_clip), tx)
        optimizer = nnx.Optimizer(self, tx)
        lr_track: list[float] = []
        _n_grad_steps = 0

        decay = self.ema_decay
        _loss_fn = LOSS_FNS[cfg.loss]
        use_wt = 'shock_weighted' in cfg.loss

        _flow = (model_cfg or {}).get('flow', {})
        _t_dist = _flow.get('t_dist', 'uniform')
        _sigma_t = float(_flow.get('sigma_t', 1.0))
        if _t_dist == 'logit_normal':
            def sample_t(key):
                return jax.nn.sigmoid(jax.random.normal(key) * _sigma_t)
        elif _t_dist == 'beta':
            _alpha = float(_flow.get('t_alpha', 2.0))
            _beta = float(_flow.get('t_beta', 2.0))
            def sample_t(key):
                return jax.random.beta(key, _alpha, _beta)
        else:
            def sample_t(key):
                return jax.random.uniform(key)
        ema = (jax.tree.map(jnp.copy, nnx.state(self, nnx.Param))
               if decay > 0 else None)

        def _make_fam_fns(knn_g):
            def _one_loss(m, hr, lr, tg, wt, key):
                ctx = m._context(hr, lr, knn_g)
                r = (tg - ctx['baseline']) / m.res_scale.value if m.use_residual else tg / m.res_scale.value
                kt, kx = jax.random.split(key)
                t = sample_t(kt)
                x0 = jax.random.normal(kx, r.shape)
                x_t = (1.0 - t) * x0 + t * r
                v = m.velocity(x_t, t, ctx, knn_g)
                return _loss_fn(v, r - x0, wt)

            @nnx.jit
            def _train_step(opt, hr_b, lr_b, tg_b, wt_b, valid_b, key):
                keys = jax.random.split(key, hr_b.shape[0])
                def lf(m):
                    losses = jax.vmap(lambda h, l, t, w, k: _one_loss(m, h, l, t, w, k))(
                        hr_b, lr_b, tg_b, wt_b, keys)
                    return (losses * valid_b).sum() / valid_b.sum()
                loss, grads = nnx.value_and_grad(lf)(opt.model)
                leaves = jax.tree_util.tree_leaves(grads)
                grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves if hasattr(g, 'shape')))
                opt.update(grads)
                return loss, grad_norm

            def _do_step(raw, step_key):
                B = len(raw)
                samples = raw + [raw[-1]] * (batch_size - B)
                valid = jnp.array([1.] * B + [0.] * (batch_size - B), jnp.float32)
                hr_b = jnp.array(np.stack([s[0] for s in samples]))
                lr_b = jnp.array(np.stack([s[1] for s in samples]))
                tg_b = jnp.array(np.stack([s[2] for s in samples]))
                wt_b = jnp.array(np.stack([s[3] for s in samples]))
                l, gn = _train_step(optimizer, hr_b, lr_b, tg_b, wt_b, valid, step_key)
                return float(l) * B, B, float(gn)

            @nnx.jit
            def _val_step(model, hr, lr, tg, wt, t, key):
                ctx = model._context(hr, lr, knn_g)
                r = (tg - ctx['baseline']) / model.res_scale.value if model.use_residual else tg / model.res_scale.value
                x0 = jax.random.normal(key, r.shape)
                x_t = (1.0 - t) * x0 + t * r
                v = model.velocity(x_t, t, ctx, knn_g)
                w = wt[:, None] if use_wt else 1.0
                return (w * (v - (r - x0)) ** 2).mean()

            @nnx.jit
            def _gen_step(model, hr, lr):
                return model.predict(hr, lr, knn_g)

            return _do_step, _val_step, _gen_step

        # Compile une fois par géométrie
        _all_fam_fns = {name: _make_fam_fns(_all_knns[name]) for name in _all_names}

        @jax.jit
        def ema_update(e, p):
            return jax.tree.map(lambda a, b: decay * a + (1.0 - decay) * b, e, p)

        rng = np.random.default_rng(cfg.seed)
        base_key = jax.random.PRNGKey(cfg.seed)
        val_key = jax.random.PRNGKey(cfg.seed + 1)

        mu_np  = np.asarray(_all_val[0].mu)
        sig_np = np.asarray(_all_val[0].sig)
        idx_gen = np.unique(np.linspace(0, len(_all_val[0]) - 1,
                                        min(self.n_val_gen, len(_all_val[0]))).astype(int))
        idw_ref = eval_idw(_all_val[0], _primary_knn)

        # Affiche les références IDW pour chaque dataset
        def _idw_line(ref, name=None):
            vals = "  ".join(f"{v}={ref['l2'][i]:.4f}" for i, v in enumerate(VAR_LABELS))
            prefix = f"  IDW {name}:" if name else "  IDW baseline:"
            return f"{prefix}  {vals}  M={ref['mach']:.4f}"
        
        # Si multi-dataset, affiche les références IDW pour chaque dataset (ex Diamond et Naca0012)
        print(_idw_line(idw_ref, _all_names[0] if _is_multi else None))
        if _is_multi:
            for _g_i, (_g_ds, _g_knn) in enumerate(zip(_all_val[1:], list(_all_knns.values())[1:])):
                _ref_g = eval_idw(_g_ds, _g_knn)
                print(_idw_line(_ref_g, _all_names[_g_i + 1]))
        _multi_str = f"  datasets={_all_names}" if _is_multi else ""
        print(f"\nTraining FAM  epochs={cfg.epochs}  lr={cfg.lr}"
              f"  batch={batch_size}  shock_weighted={use_wt}  "
              f"ema={decay}  n_steps(sampling)={self.n_steps}  "
              f"gen_val={len(idx_gen)} samples{_multi_str}")

        def _delta(model_v, idw_v):
            pct = (model_v - idw_v) / (idw_v + 1e-9) * 100
            sign = '-' if pct <= 0 else '+'
            return f"{model_v:.4f}({sign}{abs(pct):.0f}%)"

        def _prims_str(vl2_arr, idw_l2):
            return "  ".join(f"{v}={_delta(float(vl2_arr[i]), float(idw_l2[i]))}" for i, v in enumerate(VAR_LABELS))

        train_losses, val_losses, val_l2s = [], [], []
        best_score, best_state = float('inf'), None
        step = 0
        _ep_gnorm_sum, _ep_gnorm_n, _ep_n_clipped = 0.0, 0, 0
        t0 = time.time()

        _has_flow = hasattr(_all_val[0], 'entries')

        # CSV logger
        _csv_fields = (['epoch', 'time_s', 'train_loss', 'val_fm', 'val_l2_mean', 'lr']
                       + [f'l2_{v}' for v in VAR_LABELS] + ['l2_mach']
                       + [f'idw_{v}' for v in VAR_LABELS] + ['idw_mach']
                       + ['enthalpy', 'enthalpy_gt', 'idw_enthalpy']
                       + [f'huber_l1_{v}' for v in VAR_LABELS]
                       + ['ram_gb', 'gpu_gb', 'gpu_peak_gb', 'is_best'])
        _csv_path = out_dir / f'{run_name}.csv'
        with open(_csv_path, 'w', newline='') as _f:
            csv.DictWriter(_f, fieldnames=_csv_fields).writeheader()

        for epoch in range(1, cfg.epochs + 1):
            ep_loss, n_seen = 0.0, 0
            _ep_gnorm_sum, _ep_gnorm_n, _ep_n_clipped = 0.0, 0, 0

            if _is_multi:
                _batch_iter = train_ds.iter_batches(batch_size, rng)
                _geom_iter  = ((raw, geom_idx) for raw, geom_idx in _batch_iter)
            else:
                idx_perm = rng.permutation(len(train_ds))
                def _single_iter():
                    for bi in range(0, len(idx_perm), batch_size):
                        yield [train_ds[int(i)] for i in idx_perm[bi:bi + batch_size]], 0
                _geom_iter = _single_iter()

            for raw, geom_idx in _geom_iter:
                _ds_fn, _, _ = _all_fam_fns[_all_names[geom_idx]]
                loss_contrib, B_real, gn = _ds_fn(raw, jax.random.fold_in(base_key, step))
                step += 1
                _n_grad_steps += 1
                ep_loss += loss_contrib
                n_seen += B_real
                _ep_gnorm_sum += gn
                _ep_gnorm_n += 1
                if cfg.grad_clip > 0 and gn > cfg.grad_clip:
                    _ep_n_clipped += 1
                if ema is not None:
                    ema = ema_update(ema, nnx.state(optimizer.model, nnx.Param))

            train_losses.append(ep_loss / max(n_seen, 1))
            lr_now = float(lr_sched(_n_grad_steps)) if callable(lr_sched) else float(cfg.lr)
            lr_track.append(lr_now)

            if epoch % cfg.val_every == 0 or epoch == cfg.epochs:
                eval_model = self._with_params(ema) if ema is not None else self

                # Val FM loss (sur toutes les géométries)
                vl = 0.0
                n_val_total = 0
                for i_g, (g_name, val_ds_g) in enumerate(zip(_all_names, _all_val)):
                    _, _vs_g, _ = _all_fam_fns[g_name]
                    for i in range(len(val_ds_g)):
                        hr, lr, tg, wt, *_ = val_ds_g[i]
                        t_i = ((i % 8) + 0.5) / 8.0
                        vl += float(_vs_g(eval_model, jnp.array(hr), jnp.array(lr),
                                          jnp.array(tg), jnp.array(wt),
                                          jnp.float32(t_i),
                                          jax.random.fold_in(val_key, i + i_g * 10000)))
                    n_val_total += len(val_ds_g)
                vl /= n_val_total
                val_losses.append(vl)

                # Métriques L2 physiques par géométrie + agrégat sur le primaire pour save_best
                vl2, vmach, enth_acc, enth_ref_acc = np.zeros(4), 0.0, 0.0, 0.0
                huber_l1 = np.zeros(4)
                _, _, _gs_0 = _all_fam_fns[_all_names[0]]
                for i in idx_gen:
                    hr, lr, tg, *_ = _all_val[0][int(i)]
                    pred = np.asarray(_gs_0(eval_model, jnp.array(hr), jnp.array(lr)))
                    tg_std = np.sqrt((tg ** 2).mean(0) + 1e-8)
                    huber_l1 += (np.abs(pred - tg) / tg_std >= cfg.huber_delta).mean(0)
                    pred_phys = pred * sig_np + mu_np
                    tg_phys = tg * sig_np + mu_np
                    vl2 += _rel_l2_np(pred_phys, tg_phys)
                    vmach += l2_rel(to_mach(pred_phys), to_mach(tg_phys))
                    if _has_flow:
                        mach_in_v = _all_val[0].entries[int(i)][1]
                        enth_acc     += enthalpy_rms(pred_phys, mach_in_v)
                        enth_ref_acc += enthalpy_rms(tg_phys,   mach_in_v)
                vl2 /= len(idx_gen)
                vmach /= len(idx_gen)
                huber_l1 /= len(idx_gen)
                val_l2s.append([*vl2.tolist(), vmach])

                score = float(vl2.mean())
                is_best = cfg.save_best and score < best_score
                if is_best:
                    best_score = score
                    _, best_state = nnx.split(eval_model)

                eval_model.save(out_dir / f'{self.__class__.__name__}.pkl', cfg=model_cfg)
                if epoch % (cfg.val_every * 5) == 0:
                    eval_model.save(out_dir / f'{self.__class__.__name__}_ep{epoch:04d}.pkl', cfg=model_cfg)
                plot_training_curves(train_losses, val_losses, val_l2s,
                                     out_dir / 'training_curves.png',
                                     lr_track=lr_track, idw_l2_ref=idw_ref)
                if pred_callback is not None:
                    pred_callback(eval_model, epoch, _primary_knn)

                # Mémoire GPU et RAM
                _mem = _get_mem()
                _ram, _gpu, _gpup = _mem['ram_gb'], _mem['gpu_gb'], _mem['gpu_peak_gb']
                mem_parts = []
                if not np.isnan(_ram):
                    mem_parts.append(f"RAM={_ram:.1f}G")
                _gpu_show = _gpup if not np.isnan(_gpup) else _gpu
                if not np.isnan(_gpu_show):
                    mem_parts.append(f"GPU={_gpu_show:.1f}G")
                mem_str = ('  ' + '  '.join(mem_parts)) if mem_parts else ''

                # Grad norm info
                _avg_gnorm = _ep_gnorm_sum / max(_ep_gnorm_n, 1)
                if cfg.grad_clip > 0:
                    _clip_pct = _ep_n_clipped / max(_ep_gnorm_n, 1) * 100
                    _gnorm_str = f"  gn={_avg_gnorm:.1e}({_clip_pct:.0f}%clip)"
                else:
                    _gnorm_str = f"  gn={_avg_gnorm:.1e}"

                # Écriture CSV
                _elapsed = round(time.time() - t0, 1)
                _row = {
                    'epoch': epoch, 'time_s': _elapsed,
                    'train_loss': round(train_losses[-1], 8),
                    'val_fm': round(vl, 8),
                    'val_l2_mean': round(float(vl2.mean()), 8),
                    'lr': f'{lr_now:.6e}',
                    **{f'l2_{v}': round(float(vl2[i]), 8) for i, v in enumerate(VAR_LABELS)},
                    'l2_mach': round(float(vmach), 8),
                    **{f'idw_{v}': round(float(idw_ref['l2'][i]), 8) for i, v in enumerate(VAR_LABELS)},
                    'idw_mach': round(float(idw_ref['mach']), 8),
                    'enthalpy':     round(float(enth_acc / len(idx_gen)), 8) if _has_flow else '',
                    'enthalpy_gt':  round(float(enth_ref_acc / len(idx_gen)), 8) if _has_flow else '',
                    'idw_enthalpy': round(float(idw_ref.get('enthalpy', float('nan'))), 8) if _has_flow else '',
                    **{f'huber_l1_{v}': round(float(huber_l1[i]), 6) for i, v in enumerate(VAR_LABELS)},
                    'ram_gb':       '' if np.isnan(_ram)  else round(_ram, 3),
                    'gpu_gb':       '' if np.isnan(_gpu)  else round(_gpu, 3),
                    'gpu_peak_gb':  '' if np.isnan(_gpup) else round(_gpup, 3),
                    'is_best': int(is_best),
                }
                with open(_csv_path, 'a', newline='') as _f:
                    csv.DictWriter(_f, fieldnames=_csv_fields).writerow(_row)

                _ht_str = f"  Ht={enth_acc/len(idx_gen):.3e}(gt={enth_ref_acc/len(idx_gen):.3e})" if _has_flow else ""
                _use_huber = 'huber' in cfg.loss
                print(f"  ep {epoch:4d}/{cfg.epochs}  "
                      f"tr={train_losses[-1]:.4f}  val(fm)={vl:.4f}  "
                      + _prims_str(vl2, idw_ref['l2'])
                      + f"  M={_delta(vmach, idw_ref['mach'])}"
                      + _ht_str
                      + ("  huber_L1=[" + " ".join(f"{v}:{huber_l1[i]:.0%}" for i, v in enumerate(VAR_LABELS)) + "]" if _use_huber else '')
                      + _gnorm_str
                      + mem_str
                      + f"  {_elapsed:.0f}s")

        if cfg.save_best and best_state is not None:
            graphdef, _ = nnx.split(self)
            saved = nnx.merge(graphdef, best_state)
        else:
            saved = self._with_params(ema) if ema is not None else self

        total = time.time() - t0
        print(f"    Entraînement terminé  {total:.1f}s  "
              f"({total / cfg.epochs:.2f}s/epoch)")

        ckpt = out_dir / f'{self.__class__.__name__}.pkl'
        saved.save(ckpt, cfg=model_cfg)
        print(f"    Checkpoint: {ckpt}  (best gen L2={best_score:.4f})"
              if cfg.save_best else f"    Checkpoint: {ckpt}")

        saved._run_dir = out_dir
        return saved
