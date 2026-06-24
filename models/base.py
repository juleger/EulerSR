"""
Module de base pour les réseaux de neurones de super-résolution de champs CFD (prédiction directe de HR à partir de LR)

SRModel : classe de base abstraite pour les modèles de super-résolution. Définit l'interface commune pour la prédiction, le chargement des kNN, l'entraînement et l'évaluation.
MLP : classe utilitaire pour un perceptron multi-couches simple, utilisé dans les modèles de super-résolution.

Hyperparamètres :
- resolution : résolution LR et HR du maillage (ex: 0.1 et 0.025)
- architecture : dict contenant les détails de l'architecture du modèle (ex: dimensions des couches, utilisation des gradients CFD LR, échelles de kNN, etc.)
- training : dict contenant les détails de l'entraînement (ex: nombre d'époques, taux d'apprentissage, poids de la loss physique, etc.)
- datasets : si multi-géométrie, liste des datasets à utiliser pour l'entraînement et la validation, avec leurs propres stats mu/sig pour normaliser les primitives LR et HR.

Entraînement :
- fit() : méthode d'entraînement du modèle sur un dataset de train et de validation, avec évaluation périodique sur la validation, sauvegarde du meilleur modèle, et possibilité de callback pour visualiser les prédictions pendant l'entraînement.
"""

import csv
import datetime
import math
import os
import functools
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / 'euler') not in sys.path:
    sys.path.append(str(REPO_ROOT / 'euler'))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loss import LOSS_FNS, grad_p_lsq
from utils.viz._style import VAR_LABELS
from utils.viz.training import plot_training_curves, plot_val_panels  # noqa: F401
from utils.metrics import enthalpy_rms, l2_rel
from utils.refs import to_mach, _res_tag
from utils.layout import DataLayout
from preprocessing.dataset import MultiSRDataset


def _get_mem() -> dict[str, float]:
    #RAM (psutil) + GPU (JAX memory_stats)
    ram_gb = float('nan')
    gpu_gb = float('nan')
    gpu_peak_gb = float('nan')
    try:
        import psutil
        ram_gb = psutil.Process(os.getpid()).memory_info().rss / 1e9
    except Exception:
        pass
    try:
        ms = jax.devices()[0].memory_stats()
        if ms and isinstance(ms, dict):
            gpu_gb = ms.get('bytes_in_use', float('nan')) / 1e9
            gpu_peak_gb = ms.get('peak_bytes_in_use', float('nan')) / 1e9
    except Exception:
        pass
    return {'ram_gb': ram_gb, 'gpu_gb': gpu_gb, 'gpu_peak_gb': gpu_peak_gb}


def _rel_l2_np(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    # Erreur relative L2 par variable sur un sample
    return np.sqrt(((pred - ref) ** 2).sum(0) / ((ref ** 2).sum(0) + 1e-8))



def eval_idw(ds, knn: dict, k: int = 6) -> dict:
    # Baseline IDW : L2, Mach, enthalpie (avec valeur GT pour comparaison pendant l'entraînement).
    k_use = min(k, np.asarray(knn['idx']).shape[1])
    idx_np  = np.asarray(knn['idx'])[:, :k_use].astype(np.int32)
    dist_np = np.asarray(knn['dist'])[:, :k_use].astype(np.float64)
    w = 1.0 / (dist_np ** 2 + 1e-12)
    w /= w.sum(axis=1, keepdims=True)
    w32 = w.astype(np.float32)

    mu, sig = np.asarray(ds.mu), np.asarray(ds.sig)
    has_flow = hasattr(ds, 'entries')

    l2_acc, mach_acc = np.zeros(4), 0.0
    enth_acc = enth_ref_acc = 0.0
    for i in range(len(ds)):
        _, lr, tg, *_ = ds[i]
        pred_phys = (w32[:, :, None] * lr[:, :4][idx_np]).sum(axis=1) * sig + mu
        tg_phys   = tg * sig + mu
        l2_acc   += _rel_l2_np(pred_phys, tg_phys)
        mach_acc += l2_rel(to_mach(pred_phys), to_mach(tg_phys))
        if has_flow:
            mach_in = ds.entries[i][1]
            enth_acc     += enthalpy_rms(pred_phys, mach_in)
            enth_ref_acc += enthalpy_rms(tg_phys,   mach_in)
    n = len(ds)
    out = {'l2': l2_acc / n, 'mach': mach_acc / n}
    if has_flow:
        out['enthalpy']     = enth_acc / n
        out['enthalpy_ref'] = enth_ref_acc / n
    return out


@dataclass
class TrainConfig:
    epochs: int = 100
    lr: float = 2e-4
    weight_decay: float = 1e-5
    val_every: int = 10
    seed: int = 42
    loss: str = 'rel_mse'
    schedule: str = 'cosine'
    grad_clip: float = 0.0
    save_best: bool = True
    batch_size: int = 1
    lambda_phys: float = 0.0
    warmup_epochs: int = 0
    huber_delta: float = 1.0
    ema_decay: float = 0.0

class Buffer(nnx.Variable):
    """Variable non entrainable pour stocker des valeurs"""


class MLP(nnx.Module):
    # Multilayer Perceptron simple avec activations SiLU et dernière couche linéaire
    def __init__(self, dims: list[int], rngs: nnx.Rngs, dtype=None):
        self.layers = [nnx.Linear(dims[i], dims[i + 1], rngs=rngs, dtype=dtype)
                       for i in range(len(dims) - 1)]

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers[:-1]:
            x = jax.nn.silu(layer(x))
        return self.layers[-1](x)

class SRModel(nnx.Module):
    # Classe de base abstraite des modèles de Super-resolution
    def predict(self, hr_feat: jax.Array, lr_feat: jax.Array, knn: dict) -> jax.Array:
        raise NotImplementedError

    @classmethod
    def load_knn(cls, layout: 'DataLayout') -> dict:
        raise NotImplementedError

    def save(self, path: str | Path, cfg: dict | None = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _, state = nnx.split(self)
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        if cfg is not None:
            with open(path.with_suffix('.yaml'), 'w') as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: str | Path) -> 'SRModel':
        path = Path(path)
        cfg = None
        cfg_path = path.with_suffix('.yaml')
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
        dummy = cls(nnx.Rngs(0), cfg)
        graphdef, _ = nnx.split(dummy)
        with open(path, 'rb') as f:
            state = pickle.load(f)
        return nnx.merge(graphdef, state)

    def _with_params(self, params) -> 'SRModel':
        graphdef, _, rest = nnx.split(self, nnx.Param, ...)
        return nnx.merge(graphdef, params, rest)

    def fit(self, train_ds, val_ds, knn, cfg: TrainConfig, out_dir: str | Path = 'results/checkpoints',
            model_cfg: dict | None = None, run_name: str | None = None, pred_callback=None) -> 'SRModel':
        
        _is_multi = isinstance(train_ds, MultiSRDataset)
        if _is_multi:
            _all_knns  = knn          # dict[name -> knn_dict]
            _all_names = train_ds.names
            _all_val   = val_ds.datasets if isinstance(val_ds, MultiSRDataset) else [val_ds]
            _primary_knn = _all_knns[_all_names[0]]
        else:
            _all_knns    = {'primary': knn}
            _all_names   = ['primary']
            _all_val     = [val_ds]
            _primary_knn = knn

        if run_name is None:
            run_name = self.__class__.__name__

        out_dir = Path(out_dir) / run_name
        out_dir.mkdir(parents=True, exist_ok=True)

        sep = '─' * 52
        print(f'\n{sep}')
        print(f'  Run : {run_name}')
        print(f'  Out : {out_dir}')
        if model_cfg:
            cfg_str = yaml.dump(model_cfg, default_flow_style=False,
                                sort_keys=False).rstrip()
            for line in cfg_str.splitlines():
                print(f'  {line}')
        print(f'  ·  n_train={len(train_ds)}  n_val={len(val_ds)}'
              f'  params={n_params(self):,}')
        print(sep)

        _steps_per_epoch = math.ceil(len(train_ds) / max(cfg.batch_size, 1))
        n_steps = cfg.epochs * _steps_per_epoch
        warmup_steps = cfg.warmup_epochs * _steps_per_epoch

        if cfg.schedule == 'cosine':
            if warmup_steps > 0:
                lr_sched = optax.warmup_cosine_decay_schedule(
                    init_value=0.0, peak_value=cfg.lr,
                    warmup_steps=warmup_steps, decay_steps=n_steps,
                )
            else:
                lr_sched = optax.cosine_decay_schedule(cfg.lr, n_steps)
        else:
            if warmup_steps > 0:
                lr_sched = optax.join_schedules(
                    schedules=[
                        optax.linear_schedule(0.0, cfg.lr, warmup_steps),
                        optax.constant_schedule(cfg.lr),
                    ],
                    boundaries=[warmup_steps],
                )
            else:
                lr_sched = cfg.lr

        # Optimiseur AdamW avec éventuellement du clipping de gradients
        tx = optax.adamw(lr_sched, weight_decay=cfg.weight_decay)
        if cfg.grad_clip > 0:
            tx = optax.chain(optax.clip_by_global_norm(cfg.grad_clip), tx)

        optimizer = nnx.Optimizer(self, tx)
        lr_track: list[float] = []
        _n_grad_steps = 0
        decay = cfg.ema_decay
        ema = (jax.tree.map(jnp.copy, nnx.state(self, nnx.Param))
               if decay > 0 else None)

        @jax.jit
        def _ema_update(e, p):
            return jax.tree.map(lambda a, b: decay * a + (1.0 - decay) * b, e, p)

        _loss_fn = LOSS_FNS[cfg.loss]
        compute_loss = (functools.partial(_loss_fn, delta=cfg.huber_delta)
                        if 'huber' in cfg.loss else _loss_fn)
        batch_size  = cfg.batch_size
        lambda_phys = cfg.lambda_phys
        use_phys_loss = lambda_phys > 0 and 'grad_op' in _primary_knn

        def _make_step_fns(knn_g):
            _uph = lambda_phys > 0 and 'grad_op' in knn_g

            @nnx.jit
            def _ss(opt, hr, lr, tg, wt, gp_tg):
                def lf(m):
                    pred = m.predict(hr, lr, knn_g)
                    loss = compute_loss(pred, tg, wt)
                    if _uph:
                        loss = loss + lambda_phys * (
                            (jnp.arcsinh(grad_p_lsq(pred, knn_g)) - gp_tg) ** 2).mean()
                    return loss
                loss, grads = nnx.value_and_grad(lf)(opt.model)
                leaves = jax.tree_util.tree_leaves(grads)
                grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves if hasattr(g, 'shape')))
                opt.update(grads)
                return loss, grad_norm

            @nnx.jit
            def _sb(opt, hr_b, lr_b, tg_b, wt_b, valid_b, gp_b):
                def lf(m):
                    def one(hr, lr, tg, wt, gp):
                        pred = m.predict(hr, lr, knn_g)
                        loss = compute_loss(pred, tg, wt)
                        if _uph:
                            loss = loss + lambda_phys * (
                                (jnp.arcsinh(grad_p_lsq(pred, knn_g)) - gp) ** 2).mean()
                        return loss
                    losses = jax.vmap(one)(hr_b, lr_b, tg_b, wt_b, gp_b)
                    return (losses * valid_b).sum() / valid_b.sum()
                loss, grads = nnx.value_and_grad(lf)(opt.model)
                leaves = jax.tree_util.tree_leaves(grads)
                grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves if hasattr(g, 'shape')))
                opt.update(grads)
                return loss, grad_norm

            @nnx.jit
            def _ev(model, hr, lr, tg, wt):
                pred = model.predict(hr, lr, knn_g)
                return compute_loss(pred, tg, wt), pred

            return _ss, _sb, _ev

        _all_step_fns = {name: _make_step_fns(_all_knns[name]) for name in _all_names}
        step_single, step_batched, eval_step_val = _all_step_fns[_all_names[0]]

        _has_flow = hasattr(_all_val[0], 'entries')

        def _do_one_batch(raw, sfns):
            _ss, _sb, _ = sfns
            B = len(raw)
            if batch_size > 1:
                B_real = B
                pad = [raw[-1]] * (batch_size - B_real)
                samples = raw + pad
                valid = np.array([1.0] * B_real + [0.0] * (batch_size - B_real), dtype=np.float32)
                hr_b = jnp.array(np.stack([s[0] for s in samples]))
                lr_b = jnp.array(np.stack([s[1] for s in samples]))
                tg_b = jnp.array(np.stack([s[2] for s in samples]))
                wt_b = jnp.array(np.stack([s[3] for s in samples]))
                gp_b = jnp.array(np.stack([s[4] for s in samples]))
                loss, gnorm = _sb(optimizer, hr_b, lr_b, tg_b, wt_b, jnp.array(valid), gp_b)
                return float(loss) * B_real, B_real, float(gnorm)
            else:
                hr, lr, tg, wt, gp = raw[0]
                loss, gnorm = _ss(optimizer, jnp.array(hr), jnp.array(lr),
                                  jnp.array(tg), jnp.array(wt), jnp.array(gp))
                return float(loss), 1, float(gnorm)

        rng = np.random.default_rng(cfg.seed)
        train_losses, val_losses, val_l2s = [], [], []
        best_val, best_state = float('inf'), None
        _ep_gnorm_sum, _ep_gnorm_n, _ep_n_clipped = 0.0, 0, 0
        t0 = time.time()

        # logger CSV
        _csv_fields = (['epoch', 'time_s', 'train_loss', 'val_loss', 'lr']
                       + [f'l2_{v}' for v in VAR_LABELS] + ['l2_mach']
                       + [f'idw_{v}' for v in VAR_LABELS] + ['idw_mach']
                       + ['enthalpy', 'enthalpy_gt', 'idw_enthalpy']
                       + ['ram_gb', 'gpu_gb', 'gpu_peak_gb', 'is_best'])
        _csv_path = out_dir / f'{run_name}.csv'
        with open(_csv_path, 'w', newline='') as _f:
            csv.DictWriter(_f, fieldnames=_csv_fields).writeheader()

        extra_loss = ""
        if use_phys_loss:
            extra_loss += f"  lambda_phys={lambda_phys}"
        warmup_str = f"  warmup={cfg.warmup_epochs}ep" if cfg.warmup_epochs > 0 else ""
        _ema_str = f"  ema={decay}" if decay > 0 else ""
        _multi_str = f"  datasets={_all_names}" if _is_multi else ""
        print(f"\nTraining {self.__class__.__name__}  "
              f"epochs={cfg.epochs}  lr={cfg.lr}  loss={cfg.loss}{extra_loss}  "
              f"schedule={cfg.schedule}{warmup_str}  grad_clip={cfg.grad_clip}  "
              f"batch={batch_size}{_ema_str}{_multi_str}")

        # Baseline IDW sur le jeu de validation — une référence par géométrie
        mu_np  = np.asarray(_all_val[0].mu)
        sig_np = np.asarray(_all_val[0].sig)

        _idw_refs = {_all_names[0]: eval_idw(_all_val[0], _primary_knn)}
        if _is_multi:
            for _g_i, (_g_ds, _g_knn) in enumerate(zip(_all_val[1:], list(_all_knns.values())[1:])):
                _idw_refs[_all_names[_g_i + 1]] = eval_idw(_g_ds, _g_knn)
        idw_ref = _idw_refs[_all_names[0]]

        def _idw_line(ref, name=None):
            vals = "  ".join(f"{v}={ref['l2'][i]:.4f}" for i, v in enumerate(VAR_LABELS))
            prefix = f"  IDW [{name}]:" if name else "  IDW baseline:"
            return f"{prefix}  {vals}  M={ref['mach']:.4f}"
        print(_idw_line(idw_ref, _all_names[0] if _is_multi else None))
        if _is_multi:
            for _g_name in _all_names[1:]:
                print(_idw_line(_idw_refs[_g_name], _g_name))

        def _delta(model_v, idw_v):
            pct = (model_v - idw_v) / (idw_v + 1e-9) * 100
            sign = '-' if pct <= 0 else '+'
            return f"{model_v:.4f}({sign}{abs(pct):.0f}%)"

        def _prims_str(vl2_arr, idw_l2):
            return "  ".join(f"{v}={_delta(float(vl2_arr[i]), float(idw_l2[i]))}" for i, v in enumerate(VAR_LABELS))

        for epoch in range(1, cfg.epochs + 1):
            ep_loss, n = 0.0, 0
            _ep_gnorm_sum, _ep_gnorm_n, _ep_n_clipped = 0.0, 0, 0

            if _is_multi:
                # Mode multi-dataset : batches mono-géométrie routés vers le bon knn
                for raw, geom_idx in train_ds.iter_batches(batch_size, rng):
                    sfns = _all_step_fns[_all_names[geom_idx]]
                    loss_c, B_r, gn = _do_one_batch(raw, sfns)
                    ep_loss += loss_c; n += B_r; _n_grad_steps += 1
                    _ep_gnorm_sum += gn; _ep_gnorm_n += 1
                    if cfg.grad_clip > 0 and gn > cfg.grad_clip:
                        _ep_n_clipped += 1
                    if ema is not None:
                        ema = _ema_update(ema, nnx.state(optimizer.model, nnx.Param))
            else:
                idx = rng.permutation(len(train_ds))
                sfns = _all_step_fns['primary']
                for bi in range(0, len(idx), batch_size):
                    raw = [train_ds[int(i)] for i in idx[bi:bi + batch_size]]
                    loss_c, B_r, gn = _do_one_batch(raw, sfns)
                    ep_loss += loss_c; n += B_r; _n_grad_steps += 1
                    _ep_gnorm_sum += gn; _ep_gnorm_n += 1
                    if cfg.grad_clip > 0 and gn > cfg.grad_clip:
                        _ep_n_clipped += 1
                    if ema is not None:
                        ema = _ema_update(ema, nnx.state(optimizer.model, nnx.Param))

            train_losses.append(ep_loss / max(n, 1))
            lr_now = float(lr_sched(_n_grad_steps)) if callable(lr_sched) else float(cfg.lr)
            lr_track.append(lr_now)

            if epoch % cfg.val_every == 0 or epoch == cfg.epochs:
                eval_model = self._with_params(ema) if ema is not None else self
                # Validation : loss + métriques L2 physiques (dénormalisées)
                vl, vl2, vmach = 0.0, np.zeros(4), 0.0
                enth_acc = enth_ref_acc = 0.0
                n_val = 0
                _per_geom_metrics: dict = {}

                for i_g, (g_name, val_ds_g) in enumerate(zip(_all_names, _all_val)):
                    _, _, _ev_g = _all_step_fns[g_name]
                    knn_g  = _all_knns[g_name]
                    mu_g   = np.asarray(train_ds.datasets[i_g].mu if _is_multi else train_ds.mu)
                    sig_g  = np.asarray(train_ds.datasets[i_g].sig if _is_multi else train_ds.sig)
                    _hfl_g = hasattr(val_ds_g, 'entries')

                    vl_g = 0.0
                    vl2_g, vmach_g = np.zeros(4), 0.0
                    enth_g = enth_ref_g = 0.0
                    for i in range(len(val_ds_g)):
                        hr, lr, tg, wt, *_ = val_ds_g[i]
                        l, pred = _ev_g(eval_model, jnp.array(hr), jnp.array(lr), jnp.array(tg), jnp.array(wt))
                        vl_g += float(l)
                        pred_phys = np.asarray(pred) * sig_g + mu_g
                        tg_phys   = tg * sig_g + mu_g
                        vl2_g   += _rel_l2_np(pred_phys, tg_phys)
                        vmach_g += l2_rel(to_mach(pred_phys), to_mach(tg_phys))
                        if _hfl_g:
                            mach_in_v = val_ds_g.entries[i][1]
                            enth_g     += enthalpy_rms(pred_phys, mach_in_v)
                            enth_ref_g += enthalpy_rms(tg_phys,   mach_in_v)
                    n_g = len(val_ds_g)
                    _per_geom_metrics[g_name] = {
                        'vl':       vl_g / max(n_g, 1),
                        'vl2':      vl2_g / max(n_g, 1),
                        'vmach':    vmach_g / max(n_g, 1),
                        'enth':     enth_g / max(n_g, 1) if _hfl_g else None,
                        'enth_ref': enth_ref_g / max(n_g, 1) if _hfl_g else None,
                    }
                    vl          += vl_g
                    vl2         += vl2_g
                    vmach       += vmach_g
                    enth_acc    += enth_g
                    enth_ref_acc += enth_ref_g
                    n_val       += n_g

                vl /= n_val; vl2 /= n_val; vmach /= n_val
                val_losses.append(vl)
                val_l2s.append([*vl2.tolist(), vmach])

                is_best = cfg.save_best and vl < best_val
                if is_best:
                    best_val = vl
                    _, best_state = nnx.split(eval_model)

                eval_model.save(out_dir / f'{self.__class__.__name__}.pkl', cfg=model_cfg)

                # Export des courbes en direct à chaque validation
                plot_training_curves(train_losses, val_losses, val_l2s,
                                     out_dir / 'training_curves.png',
                                     lr_track=lr_track, idw_l2_ref=idw_ref)

                if pred_callback is not None:
                    pred_callback(eval_model, epoch, _primary_knn)

                _mem = _get_mem()
                _ram, _gpu, _gpup = _mem['ram_gb'], _mem['gpu_gb'], _mem['gpu_peak_gb']
                mem_parts = []
                if not np.isnan(_ram):
                    mem_parts.append(f"RAM={_ram:.1f}G")
                _gpu_show = _gpup if not np.isnan(_gpup) else _gpu
                if not np.isnan(_gpu_show):
                    mem_parts.append(f"GPU={_gpu_show:.1f}G")
                mem_str = ('  ' + '  '.join(mem_parts)) if mem_parts else ''

                # Écriture CSV
                _elapsed = round(time.time() - t0, 1)
                _row = {
                    'epoch': epoch, 'time_s': _elapsed,
                    'train_loss': round(train_losses[-1], 8),
                    'val_loss': round(vl, 8),
                    'lr': f'{lr_now:.6e}',
                    **{f'l2_{v}': round(float(vl2[i]), 8) for i, v in enumerate(VAR_LABELS)},
                    'l2_mach': round(float(vmach), 8),
                    **{f'idw_{v}': round(float(idw_ref['l2'][i]), 8) for i, v in enumerate(VAR_LABELS)},
                    'idw_mach': round(float(idw_ref['mach']), 8),
                    'enthalpy':     round(float(enth_acc / n_val), 8) if _has_flow else '',
                    'enthalpy_gt':  round(float(enth_ref_acc / n_val), 8) if _has_flow else '',
                    'idw_enthalpy': round(float(idw_ref.get('enthalpy', float('nan'))), 8) if _has_flow else '',
                    'ram_gb':       '' if np.isnan(_ram)  else round(_ram, 3),
                    'gpu_gb':       '' if np.isnan(_gpu)          else round(_gpu, 3),
                    'gpu_peak_gb':  '' if np.isnan(_gpup)         else round(_gpup, 3),
                    'is_best': int(is_best),
                }
                with open(_csv_path, 'a', newline='') as _f:
                    csv.DictWriter(_f, fieldnames=_csv_fields).writerow(_row)

                # Indicateur grad clip : norme moy des gradients bruts + % d'étapes clippées
                _avg_gnorm = _ep_gnorm_sum / max(_ep_gnorm_n, 1)
                if cfg.grad_clip > 0:
                    _clip_pct = _ep_n_clipped / max(_ep_gnorm_n, 1) * 100
                    _gnorm_str = f"  gn={_avg_gnorm:.1e}({_clip_pct:.0f}%clip)"
                else:
                    _gnorm_str = f"  gn={_avg_gnorm:.1e}"

                if _is_multi:
                    # Ligne de résumé global, puis une ligne par géométrie
                    print(f"  ep {epoch:4d}/{cfg.epochs}  "
                          f"tr={train_losses[-1]:.4f}  val={vl:.4f}"
                          + _gnorm_str + mem_str + f"  {_elapsed:.0f}s")
                    _w = max(len(n) for n in _all_names)
                    for g_name in _all_names:
                        gm  = _per_geom_metrics[g_name]
                        ref = _idw_refs[g_name]
                        _ht_g = (f"  Ht={gm['enth']:.3e}(gt={gm['enth_ref']:.3e})"
                                 if gm['enth'] is not None else "")
                        print(f"    [{g_name:<{_w}}]  val={gm['vl']:.4f}  "
                              + _prims_str(gm['vl2'], ref['l2'])
                              + f"  M={_delta(gm['vmach'], ref['mach'])}"
                              + _ht_g)
                else:
                    _ht_str = f"  Ht={enth_acc/n_val:.3e}(gt={enth_ref_acc/n_val:.3e})" if _has_flow else ""
                    print(f"  ep {epoch:4d}/{cfg.epochs}  "
                        f"tr={train_losses[-1]:.4f}  val={vl:.4f}  "
                        + _prims_str(vl2, idw_ref['l2'])
                        + f"  M={_delta(vmach, idw_ref['mach'])}"
                        + _ht_str
                        + _gnorm_str
                        + mem_str
                        + f"  {_elapsed:.0f}s")

        # Si besoin, sauvegarde le meilleur modèle trouvé pendant l'entraînement (en fonction de la val)
        if cfg.save_best and best_state is not None:
            graphdef, _ = nnx.split(self)
            saved = nnx.merge(graphdef, best_state)
        else:
            saved = self._with_params(ema) if ema is not None else self

        total = time.time() - t0
        print(f"    Entraînement terminé  {total:.1f}s  "
              f"({total / cfg.epochs:.2f}s/epoch  "
              f"{total / (cfg.epochs * len(train_ds)) * 1000:.1f}ms/sample)")

        # Sauvegarde du modèle final et des courbes d'entraînement
        ckpt = out_dir / f'{self.__class__.__name__}.pkl'
        saved.save(ckpt, cfg=model_cfg)
        print(f"    Checkpoint: {ckpt}  (best_val={best_val:.4f})" if cfg.save_best
              else f"    Checkpoint: {ckpt}")

        saved._run_dir = out_dir
        return saved


def _knn_path(layout: DataLayout, hr_res: float, lr_res: float, k: int) -> Path:
    lr_tag, hr_tag = _res_tag(lr_res), _res_tag(hr_res)
    p = layout.knn_dir / f'knn_{lr_tag}_to_{hr_tag}_k{k}.npz'
    if not p.exists() and k == 6:
        p = layout.knn_dir / f'knn_{lr_tag}_to_{hr_tag}.npz'
    return p


def _ensure_knn(layout: DataLayout, hr_res: float, lr_res: float, k: int) -> Path:
    p = _knn_path(layout, hr_res, lr_res, k)
    if p.exists():
        return p

    if str(Path(__file__).resolve().parents[1] / 'preprocessing') not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'preprocessing'))
    from knn import build

    mesh_hr = np.load(layout.mesh_path(hr_res), allow_pickle=True).item()
    mesh_lr = np.load(layout.mesh_path(lr_res), allow_pickle=True).item()

    c_hr = np.asarray(mesh_hr.barycenter, dtype=np.float32)
    c_lr = np.asarray(mesh_lr.barycenter, dtype=np.float32)

    layout.knn_dir.mkdir(parents=True, exist_ok=True)
    print(f"    kNN manquant, construction automatique: {p}")
    build(c_hr, c_lr, k=k, save_path=p)
    return p


def load_cfg(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def n_params(model: nnx.Module) -> int:
    _, state = nnx.split(model)
    leaves = jax.tree_util.tree_leaves(state)
    total = 0
    for leaf in leaves:
        val = getattr(leaf, 'value', leaf)
        if hasattr(val, 'size'):
            total += int(val.size)
    return total
