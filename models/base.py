"""
Modèles de base pour la super-résolution de champs CFD (prédiction HR à partir de LR).

SRModel : classe abstraite définissant l'interface commune (predict, load_knn, fit).
MLP : perceptron multi-couches utilitaire.

Config via dict (architecture/training/datasets) ; fit() gère l'entraînement,
la validation périodique, le meilleur checkpoint et un callback de visualisation.
"""

import csv
import math
import os
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

from loss import LOSS_FNS, grad_p_lsq, enthalpy_residual
from utils.viz._style import VAR_LABELS
from utils.viz.training import plot_training_curves
from utils.metrics import enthalpy_rms, l2_rel, idw_weights
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
    idx_np = np.asarray(knn['idx'])[:, :k_use].astype(np.int32)
    w32 = idw_weights(np.asarray(knn['dist'])[:, :k_use]).astype(np.float32)

    mu, sig = np.asarray(ds.mu), np.asarray(ds.sig)
    has_flow = hasattr(ds, 'entries')

    l2_acc, mach_acc = np.zeros(4), 0.0
    enth_acc = enth_ref_acc = 0.0
    for i in range(len(ds)):
        _, lr, tg, *_ = ds[i]
        pred_phys = (w32[:, :, None] * lr[:, :4][idx_np]).sum(axis=1) * sig + mu
        tg_phys = tg * sig + mu
        l2_acc += _rel_l2_np(pred_phys, tg_phys)
        mach_acc += l2_rel(to_mach(pred_phys), to_mach(tg_phys))
        if has_flow:
            mach_in = ds.entries[i][1]
            enth_acc += enthalpy_rms(pred_phys, mach_in)
            enth_ref_acc += enthalpy_rms(tg_phys, mach_in)
    n = len(ds)
    out = {'l2': l2_acc / n, 'mach': mach_acc / n}
    if has_flow:
        out['enthalpy'] = enth_acc / n
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
    lambda_enthalpy: float = 0.0
    lambda_endpoint: float = 0.0
    lambda_endpoint_warmup_epochs: int = 0  # rampe lineaire 0 -> lambda_endpoint (0 = pas de warmup)
    warmup_epochs: int = 0
    ema_decay: float = 0.0
    gradmix: bool = False           # multi-branche : melange les gradients de toutes les
                                    # branches par meta-step au lieu d'un update sequentiel
                                    # mono-branche (cf. fit)
    gradnorm: bool = False          # ponderation dynamique GradNorm des branches (Chen et al.
                                    # 2018) au lieu des poids statiques 'weight:', cf. fit
    gradnorm_alpha: float = 1.5     # force de rappel vers l'equilibrage (0 = pas de rappel)
    gradnorm_lr: float = 0.025      # pas de descente sur les poids w_i (SGD, 4-8 scalaires)

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

def _build_schedule(cfg: 'TrainConfig', n_steps: int, warmup_steps: int):
    """Schedule de learning rate (cosine/constant, avec warmup éventuel)."""
    if cfg.schedule == 'cosine':
        if warmup_steps > 0:
            return optax.warmup_cosine_decay_schedule(
                init_value=0.0, peak_value=cfg.lr,
                warmup_steps=warmup_steps, decay_steps=n_steps,
            )
        return optax.cosine_decay_schedule(cfg.lr, n_steps)
    if warmup_steps > 0:
        return optax.join_schedules(
            schedules=[
                optax.linear_schedule(0.0, cfg.lr, warmup_steps),
                optax.constant_schedule(cfg.lr),
            ],
            boundaries=[warmup_steps],
        )
    return cfg.lr


def _delta(model_v, idw_v) -> str:
    pct = (model_v - idw_v) / (idw_v + 1e-9) * 100
    sign = '-' if pct <= 0 else '+'
    return f"{model_v:.4f}({sign}{abs(pct):.0f}%)"


def _prims_str(vl2_arr, idw_l2) -> str:
    return "  ".join(f"{v}={_delta(float(vl2_arr[i]), float(idw_l2[i]))}"
                     for i, v in enumerate(VAR_LABELS))


def _validate(eval_model, all_names, all_val, all_step_fns, train_ds, is_multi,
              max_samples: int | None = None) -> dict:
    """Boucle de validation : loss + métriques L2 physiques (dénormalisées) par géométrie"""
    vl, vl2, vmach = 0.0, np.zeros(4), 0.0
    enth_acc = enth_ref_acc = 0.0
    n_val = 0
    per_geom: dict = {}

    for i_g, (g_name, val_ds_g) in enumerate(zip(all_names, all_val)):
        ev_g = all_step_fns[g_name]['ev']
        mu_g = np.asarray(train_ds.datasets[i_g].mu if is_multi else train_ds.mu)
        sig_g = np.asarray(train_ds.datasets[i_g].sig if is_multi else train_ds.sig)
        has_flow_g = hasattr(val_ds_g, 'entries')

        if max_samples is not None and len(val_ds_g) > max_samples:
            idxs = np.unique(np.linspace(0, len(val_ds_g) - 1, max_samples).astype(int))
        else:
            idxs = range(len(val_ds_g))

        vl_g = 0.0
        vl2_g, vmach_g = np.zeros(4), 0.0
        enth_g = enth_ref_g = 0.0
        for i in idxs:
            hr, lr, tg, wt, *_ = val_ds_g[i]
            l, pred = ev_g(eval_model, jnp.array(hr), jnp.array(lr),
                           jnp.array(tg), jnp.array(wt))
            vl_g += float(l)
            pred_phys = np.asarray(pred) * sig_g + mu_g
            tg_phys = tg * sig_g + mu_g
            vl2_g += _rel_l2_np(pred_phys, tg_phys)
            vmach_g += l2_rel(to_mach(pred_phys), to_mach(tg_phys))
            if has_flow_g:
                mach_in_v = val_ds_g.entries[i][1]
                enth_g += enthalpy_rms(pred_phys, mach_in_v)
                enth_ref_g += enthalpy_rms(tg_phys, mach_in_v)
        n_g = len(idxs)
        per_geom[g_name] = {
            'vl': vl_g / max(n_g, 1),
            'vl2': vl2_g / max(n_g, 1),
            'vmach': vmach_g / max(n_g, 1),
            'enth': enth_g / max(n_g, 1) if has_flow_g else None,
            'enth_ref': enth_ref_g / max(n_g, 1) if has_flow_g else None,
        }
        vl += vl_g
        vl2 += vl2_g
        vmach += vmach_g
        enth_acc += enth_g
        enth_ref_acc += enth_ref_g
        n_val += n_g

    vl /= n_val; vl2 /= n_val; vmach /= n_val
    return {'vl': vl, 'vl2': vl2, 'vmach': vmach,
            'enth_acc': enth_acc, 'enth_ref_acc': enth_ref_acc,
            'n_val': n_val, 'per_geom': per_geom}


def _log_epoch(epoch, cfg, metrics, *, train_loss, lr_now, elapsed, is_best, mem,
               gnorm_sum, gnorm_n, n_clipped, idw_ref, idw_refs,
               all_names, is_multi, has_flow, csv_path, csv_fields, gn_w=None) -> None:
    """Écrit la ligne CSV de l'epoch et imprime le résumé console."""
    vl, vl2, vmach = metrics['vl'], metrics['vl2'], metrics['vmach']
    enth_acc, enth_ref_acc = metrics['enth_acc'], metrics['enth_ref_acc']
    n_val, per_geom = metrics['n_val'], metrics['per_geom']

    ram, gpu, gpup = mem['ram_gb'], mem['gpu_gb'], mem['gpu_peak_gb']
    mem_parts = []
    if not np.isnan(ram):
        mem_parts.append(f"RAM={ram:.1f}G")
    gpu_show = gpup if not np.isnan(gpup) else gpu
    if not np.isnan(gpu_show):
        mem_parts.append(f"GPU={gpu_show:.1f}G")
    mem_str = ('  ' + '  '.join(mem_parts)) if mem_parts else ''

    row = {
        'epoch': epoch, 'time_s': elapsed,
        'train_loss': round(train_loss, 8),
        'val_loss': round(vl, 8),
        'lr': f'{lr_now:.6e}',
        **{f'l2_{v}': round(float(vl2[i]), 8) for i, v in enumerate(VAR_LABELS)},
        'l2_mach': round(float(vmach), 8),
        **{f'idw_{v}': round(float(idw_ref['l2'][i]), 8) for i, v in enumerate(VAR_LABELS)},
        'idw_mach': round(float(idw_ref['mach']), 8),
        'enthalpy': round(float(enth_acc / n_val), 8) if has_flow else '',
        'enthalpy_gt': round(float(enth_ref_acc / n_val), 8) if has_flow else '',
        'idw_enthalpy': round(float(idw_ref.get('enthalpy', float('nan'))), 8) if has_flow else '',
        **({f'val_{nm}': round(float(per_geom[nm]['vl']), 8) for nm in all_names} if is_multi else {}),
        **({f'gn_w_{nm}': round(float(gn_w[nm]), 5) for nm in all_names} if gn_w is not None else {}),
        'ram_gb': '' if np.isnan(ram) else round(ram, 3),
        'gpu_gb': '' if np.isnan(gpu) else round(gpu, 3),
        'gpu_peak_gb': '' if np.isnan(gpup) else round(gpup, 3),
        'is_best': int(is_best),
    }
    with open(csv_path, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=csv_fields).writerow(row)

    avg_gnorm = gnorm_sum / max(gnorm_n, 1)
    if cfg.grad_clip > 0:
        clip_pct = n_clipped / max(gnorm_n, 1) * 100
        gnorm_str = f"  gn={avg_gnorm:.1e}({clip_pct:.0f}%clip)"
    else:
        gnorm_str = f"  gn={avg_gnorm:.1e}"

    if is_multi:
        # Ligne de résumé global, puis une ligne par géométrie
        print(f"  ep {epoch:4d}/{cfg.epochs}  "
              f"tr={train_loss:.4f}  val={vl:.4f}"
              + gnorm_str + mem_str + f"  {elapsed:.0f}s")
        w = max(len(nm) for nm in all_names)
        for g_name in all_names:
            gm = per_geom[g_name]
            ref = idw_refs[g_name]
            ht_g = (f"  Ht={gm['enth']:.3e}(gt={gm['enth_ref']:.3e})"
                    if gm['enth'] is not None else "")
            gn_g = f"  w={gn_w[g_name]:.2f}" if gn_w is not None else ""
            print(f"    [{g_name:<{w}}]  val={gm['vl']:.4f}  "
                  + _prims_str(gm['vl2'], ref['l2'])
                  + f"  M={_delta(gm['vmach'], ref['mach'])}"
                  + ht_g + gn_g)
    else:
        ht_str = f"  Ht={enth_acc/n_val:.3e}(gt={enth_ref_acc/n_val:.3e})" if has_flow else ""
        print(f"  ep {epoch:4d}/{cfg.epochs}  "
              f"tr={train_loss:.4f}  val={vl:.4f}  "
              + _prims_str(vl2, idw_ref['l2'])
              + f"  M={_delta(vmach, idw_ref['mach'])}"
              + ht_str
              + gnorm_str
              + mem_str
              + f"  {elapsed:.0f}s")


class SRModel(nnx.Module):
    # Classe de base abstraite des modèles de Super-resolution

    # Sous-échantillonnage de la validation pour les modèles à inférence coûteuse
    val_max_samples: int | None = None

    def predict(self, hr_feat: jax.Array, lr_feat: jax.Array, knn: dict) -> jax.Array:
        raise NotImplementedError

    def _pre_fit(self, branch_pairs: list, cfg: 'TrainConfig') -> None:
        """Hook appelé une fois avant l'entraînement (ex: calibration res_scale FAM)
        """

    def _phys_terms(self, pred: jax.Array, hr: jax.Array, gp: jax.Array,
                    knn_g: dict, aux: dict) -> jax.Array:
        """Termes physiques soft (résidu grad_p + enthalpie) ajoutés à la loss"""
        extra = 0.0
        if aux['lambda_phys'] > 0 and 'grad_op' in knn_g:
            extra = extra + aux['lambda_phys'] * (
                (jnp.arcsinh(grad_p_lsq(pred, knn_g)) - gp) ** 2).mean()
        if aux['lambda_enthalpy'] > 0 and 'mu' in knn_g:
            extra = extra + aux['lambda_enthalpy'] * enthalpy_residual(
                pred, hr[0, 2], knn_g['mu'], knn_g['sig'], mach_norm=aux['mach_norm'])
        return extra

    def _sample_loss(self, hr: jax.Array, lr: jax.Array, tg: jax.Array,
                     wt: jax.Array, gp: jax.Array, knn_g: dict,
                     key: jax.Array, aux: dict) -> jax.Array:
        """Loss d'un sample (déterministe par défaut : data + physique).

        Surchargée par les modèles génératifs (ex: FAM, flow matching)
        """
        pred = self.predict(hr, lr, knn_g)
        return aux['loss_fn'](pred, tg, wt) + self._phys_terms(pred, hr, gp, knn_g, aux)

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
            _all_knns = knn  # dict[name -> knn_dict]
            _all_names = train_ds.names
            _all_val = val_ds.datasets if isinstance(val_ds, MultiSRDataset) else [val_ds]
            _primary_knn = _all_knns[_all_names[0]]
        else:
            _all_knns = {'primary': knn}
            _all_names = ['primary']
            _all_val = [val_ds]
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

        # Sauvegarde des stats d'entraînement pour la normalisation en éval OOD géométrique
        _primary_ds = train_ds.datasets[0] if _is_multi else train_ds
        if hasattr(self, 'mu_train'):
            self.mu_train.value = jnp.array(_primary_ds.mu, jnp.float32)
            self.sig_train.value = jnp.array(_primary_ds.sig, jnp.float32)

        # Stats de dénormalisation par branche + calibration
        _name_to_ds = {nm: (train_ds.datasets[i] if _is_multi else train_ds)
                       for i, nm in enumerate(_all_names)}
        for nm in _all_names:
            ds_b = _name_to_ds[nm]
            _all_knns[nm]['mu'] = jnp.array(ds_b.mu, jnp.float32)
            _all_knns[nm]['sig'] = jnp.array(ds_b.sig, jnp.float32)
        _branch_pairs = [(_name_to_ds[nm], _all_knns[nm]) for nm in _all_names]

        self._pre_fit(_branch_pairs, cfg)

        if isinstance(train_ds, MultiSRDataset) and cfg.gradmix:
            # gradmix : un meta-step = un micro-batch de CHAQUE branche (iter_grouped_batches),
            # donc le nombre de steps/epoque cale sur la plus grosse branche.
            _steps_per_epoch = max(math.ceil(len(ds) / max(cfg.batch_size, 1))
                                   for ds in train_ds.datasets)
        elif isinstance(train_ds, MultiSRDataset):
            # sequentiel : un update = un micro-batch d'UNE SEULE branche (iter_batches),
            # donc le nombre de steps/epoque est la somme sur toutes les branches.
            _steps_per_epoch = sum(math.ceil(len(ds) / max(cfg.batch_size, 1))
                                   for ds in train_ds.datasets)
        else:
            _steps_per_epoch = math.ceil(len(train_ds) / max(cfg.batch_size, 1))
        n_steps = cfg.epochs * _steps_per_epoch
        warmup_steps = cfg.warmup_epochs * _steps_per_epoch

        lr_sched = _build_schedule(cfg, n_steps, warmup_steps)

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
        # Modèle d'évaluation EMA construit UNE seule fois (évite fuite ram)
        eval_model = self._with_params(ema) if ema is not None else self

        @jax.jit
        def _ema_update(e, p):
            return jax.tree.map(lambda a, b: decay * a + (1.0 - decay) * b, e, p)

        @nnx.jit
        def _apply_update(opt, grads):
            opt.update(grads)

        _loss_fn = LOSS_FNS[cfg.loss]
        batch_size = cfg.batch_size
        lambda_phys = cfg.lambda_phys
        lambda_enth = cfg.lambda_enthalpy
        lambda_endpoint = cfg.lambda_endpoint

        # aux statique capturé par les closures jit (loss + poids physiques)
        # use_endpoint : flag STATIQUE (python bool, connu a la compilation) qui decide
        # si le terme d'endpoint est calcule ; independant de sa ponderation courante
        # (lambda_endpoint ci-dessous, traceee/dynamique pour le warmup). Ne jamais
        # brancher un `if` python sur aux['lambda_endpoint'] dans le modele : sa valeur
        # est un jnp scalaire trace, pas un float -- cf. TracerBoolConversionError.
        _aux = {'loss_fn': _loss_fn, 'lambda_phys': lambda_phys,
                'lambda_enthalpy': lambda_enth, 'lambda_endpoint': lambda_endpoint,
                'use_endpoint': lambda_endpoint > 0,
                'mach_norm': (float(train_ds.mach_mid), float(train_ds.mach_scale))}
        use_phys_loss = lambda_phys > 0 and 'grad_op' in _primary_knn

        def _make_step_fns(knn_g):
            @nnx.jit
            def _ss(opt, hr, lr, tg, wt, gp_tg, key, lam_ep):
                def lf(m):
                    aux_ep = {**_aux, 'lambda_endpoint': lam_ep}
                    return m._sample_loss(hr, lr, tg, wt, gp_tg, knn_g, key, aux_ep)
                loss, grads = nnx.value_and_grad(lf)(opt.model)
                leaves = jax.tree_util.tree_leaves(grads)
                grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves if hasattr(g, 'shape')))
                opt.update(grads)
                return loss, grad_norm

            @nnx.jit
            def _sb(opt, hr_b, lr_b, tg_b, wt_b, valid_b, gp_b, keys, lam_ep):
                def lf(m):
                    aux_ep = {**_aux, 'lambda_endpoint': lam_ep}
                    losses = jax.vmap(
                        lambda hr, lr, tg, wt, gp, k:
                            m._sample_loss(hr, lr, tg, wt, gp, knn_g, k, aux_ep)
                    )(hr_b, lr_b, tg_b, wt_b, gp_b, keys)
                    return (losses * valid_b).sum() / valid_b.sum()
                loss, grads = nnx.value_and_grad(lf)(opt.model)
                leaves = jax.tree_util.tree_leaves(grads)
                grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves if hasattr(g, 'shape')))
                opt.update(grads)
                return loss, grad_norm

            @nnx.jit
            def _ev(model, hr, lr, tg, wt):
                pred = model.predict(hr, lr, knn_g)
                return _loss_fn(pred, tg, wt), pred

            # Variantes "gradient seul" (pas d'opt.update interne) : accumule les
            # gradients de toutes les branches d'un meta-step avant un update unique.
            @nnx.jit
            def _gs(model, hr, lr, tg, wt, gp_tg, key, lam_ep):
                def lf(m):
                    aux_ep = {**_aux, 'lambda_endpoint': lam_ep}
                    return m._sample_loss(hr, lr, tg, wt, gp_tg, knn_g, key, aux_ep)
                return nnx.value_and_grad(lf)(model)

            @nnx.jit
            def _gb(model, hr_b, lr_b, tg_b, wt_b, valid_b, gp_b, keys, lam_ep):
                def lf(m):
                    aux_ep = {**_aux, 'lambda_endpoint': lam_ep}
                    losses = jax.vmap(
                        lambda hr, lr, tg, wt, gp, k:
                            m._sample_loss(hr, lr, tg, wt, gp, knn_g, k, aux_ep)
                    )(hr_b, lr_b, tg_b, wt_b, gp_b, keys)
                    return (losses * valid_b).sum() / valid_b.sum()
                return nnx.value_and_grad(lf)(model)

            return {'ss': _ss, 'sb': _sb, 'ev': _ev, 'gs': _gs, 'gb': _gb}

        _all_step_fns = {name: _make_step_fns(_all_knns[name]) for name in _all_names}

        _has_flow = hasattr(_all_val[0], 'entries')

        def _do_one_batch(raw, sfns, step_key, lam_ep):
            _ss, _sb = sfns['ss'], sfns['sb']
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
                keys = jax.random.split(step_key, batch_size)
                loss, gnorm = _sb(optimizer, hr_b, lr_b, tg_b, wt_b, jnp.array(valid), gp_b, keys, lam_ep)
                return float(loss) * B_real, B_real, float(gnorm)
            else:
                hr, lr, tg, wt, gp = raw[0]
                loss, gnorm = _ss(optimizer, jnp.array(hr), jnp.array(lr),
                                  jnp.array(tg), jnp.array(wt), jnp.array(gp), step_key, lam_ep)
                return float(loss), 1, float(gnorm)

        def _do_one_batch_grad(raw, sfns, step_key, lam_ep):
            """Comme _do_one_batch, mais retourne les gradients au lieu d'appliquer
            un update -- pour accumulation multi-branche (cf. iter_grouped_batches)."""
            _gs, _gb = sfns['gs'], sfns['gb']
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
                keys = jax.random.split(step_key, batch_size)
                loss, grads = _gb(optimizer.model, hr_b, lr_b, tg_b, wt_b,
                                  jnp.array(valid), gp_b, keys, lam_ep)
                return float(loss) * B_real, B_real, grads
            else:
                hr, lr, tg, wt, gp = raw[0]
                loss, grads = _gs(optimizer.model, jnp.array(hr), jnp.array(lr),
                                  jnp.array(tg), jnp.array(wt), jnp.array(gp), step_key, lam_ep)
                return float(loss), 1, grads

        def _tree_grad_norm(grads) -> float:
            leaves = jax.tree_util.tree_leaves(grads)
            return float(jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves if hasattr(g, 'shape'))))

        def _shared_grad_norm(grads) -> float:
            """Norme L2 du gradient sur le tronc partage : tout sauf les embeddings
            specifiques a la geometrie (geom_emb, scale_geom_emb). Reference de
            GradNorm ; le conditionnement geometrique etant fondu partout, on prend
            tout le tronc plutot qu'une couche 'derniere partagee'.
            """
            leaves = jax.tree_util.tree_flatten_with_path(grads)[0]
            total = 0.0
            for path, leaf in leaves:
                if leaf is None or 'geom_emb' in jax.tree_util.keystr(path):
                    continue
                total = total + jnp.sum(jnp.asarray(leaf) ** 2)
            return float(jnp.sqrt(total))

        rng = np.random.default_rng(cfg.seed)
        base_key = jax.random.PRNGKey(cfg.seed)
        train_losses, val_losses, val_l2s = [], [], []
        # Etat GradNorm : w_i par branche, cote host (4-8 scalaires, inutile de passer
        # par jit). L_i(0) rempli au tout premier meta-step.
        _gradmix = _is_multi and cfg.gradmix
        _gradnorm = _is_multi and cfg.gradnorm
        if _gradnorm and not cfg.gradmix:
            raise ValueError(
                "gradnorm=true necessite gradmix=true (ponderation dynamique calculee sur le "
                "meta-step groupe de iter_grouped_batches, sans equivalent en mode sequentiel).")
        _gn_w = np.ones(len(_all_names), dtype=np.float64) if _gradnorm else None
        _gn_l0 = np.full(len(_all_names), np.nan, dtype=np.float64) if _gradnorm else None
        best_val, best_state = float('inf'), None
        _ep_gnorm_sum, _ep_gnorm_n, _ep_n_clipped = 0.0, 0, 0
        t0 = time.time()

        # logger CSV
        _csv_fields = (['epoch', 'time_s', 'train_loss', 'val_loss', 'lr']
                       + [f'l2_{v}' for v in VAR_LABELS] + ['l2_mach']
                       + [f'idw_{v}' for v in VAR_LABELS] + ['idw_mach']
                       + ['enthalpy', 'enthalpy_gt', 'idw_enthalpy']
                       + ([f'val_{nm}' for nm in _all_names] if _is_multi else [])  # val_loss par branche
                       + ([f'gn_w_{nm}' for nm in _all_names] if _gradnorm else [])  # poids GradNorm
                       + ['ram_gb', 'gpu_gb', 'gpu_peak_gb', 'is_best'])
        _csv_path = out_dir / f'{run_name}.csv'
        with open(_csv_path, 'w', newline='') as _f:
            csv.DictWriter(_f, fieldnames=_csv_fields).writeheader()

        extra_loss = ""
        if use_phys_loss:
            extra_loss += f"  lambda_phys={lambda_phys}"
        if lambda_endpoint > 0:
            ep_warm = (f" (warmup {cfg.lambda_endpoint_warmup_epochs}ep)"
                      if cfg.lambda_endpoint_warmup_epochs > 0 else "")
            extra_loss += f"  lambda_endpoint={lambda_endpoint}{ep_warm}"
        warmup_str = f"  warmup={cfg.warmup_epochs}ep" if cfg.warmup_epochs > 0 else ""
        _ema_str = f"  ema={decay}" if decay > 0 else ""
        _multi_str = (f"  datasets={_all_names}  mode={'gradmix' if _gradmix else 'sequentiel'}"
                     if _is_multi else "")
        _gn_str = f"  gradnorm=alpha{cfg.gradnorm_alpha}/lr{cfg.gradnorm_lr}" if _gradnorm else ""
        print(f"\nTraining {self.__class__.__name__}  "
              f"epochs={cfg.epochs}  lr={cfg.lr}  loss={cfg.loss}{extra_loss}  "
              f"schedule={cfg.schedule}{warmup_str}  grad_clip={cfg.grad_clip}  "
              f"batch={batch_size}{_ema_str}{_multi_str}{_gn_str}")

        # Baseline IDW sur le jeu de validation -- une référence par géométrie.
        # eval_idw suppose un champ LR volumique, absent pour les modèles bord-seul
        # (FAMWall/DAMWall) : repli NaN plutôt que de faire planter fit().
        def _safe_eval_idw(ds, knn):
            try:
                return eval_idw(ds, knn)
            except Exception as e:
                print(f"  [IDW baseline non disponible : {e!r} -- attendu pour un modèle "
                      "bord-seul, pas de champ LR volumique]")
                return {'l2': np.full(4, np.nan), 'mach': float('nan')}

        _idw_refs = {_all_names[0]: _safe_eval_idw(_all_val[0], _primary_knn)}
        if _is_multi:
            for _g_i, (_g_ds, _g_knn) in enumerate(zip(_all_val[1:], list(_all_knns.values())[1:])):
                _idw_refs[_all_names[_g_i + 1]] = _safe_eval_idw(_g_ds, _g_knn)
        idw_ref = _idw_refs[_all_names[0]]

        def _idw_line(ref, name=None):
            vals = "  ".join(f"{v}={ref['l2'][i]:.4f}" for i, v in enumerate(VAR_LABELS))
            prefix = f"  IDW [{name}]:" if name else "  IDW baseline:"
            return f"{prefix}  {vals}  M={ref['mach']:.4f}"
        print(_idw_line(idw_ref, _all_names[0] if _is_multi else None))
        if _is_multi:
            for _g_name in _all_names[1:]:
                print(_idw_line(_idw_refs[_g_name], _g_name))

        for epoch in range(1, cfg.epochs + 1):
            ep_loss, n = 0.0, 0
            _ep_gnorm_sum, _ep_gnorm_n, _ep_n_clipped = 0.0, 0, 0

            # Nouveau N_wall tire par epoque (cf. training.wall_k_range), jamais par
            # batch : un batch JAX exige un nombre de points de bord uniforme. No-op
            # sur les datasets qui n'exposent pas roll_wall_k.
            if hasattr(train_ds, 'roll_wall_k'):
                train_ds.roll_wall_k(rng)

            if cfg.lambda_endpoint_warmup_epochs > 0:
                lam_ep_val = lambda_endpoint * min(1.0, epoch / cfg.lambda_endpoint_warmup_epochs)
            else:
                lam_ep_val = lambda_endpoint
            lam_ep = jnp.asarray(lam_ep_val, jnp.float32)

            if _gradmix:
                # gradmix : un meta-step = un micro-batch de CHAQUE branche, gradients
                # moyennes (ponderes) puis un seul optimizer.update. Opt-in : l'ablation
                # DAM/FAM 2geo n'a pas confirme de gain net face au mode sequentiel.
                for group in train_ds.iter_grouped_batches(batch_size, rng):
                    step_key = jax.random.fold_in(base_key, _n_grad_steps)
                    grads_list, geom_idxs, branch_losses = [], [], []
                    loss_acc, B_acc = 0.0, 0
                    for raw, geom_idx in group:
                        sfns = _all_step_fns[_all_names[geom_idx]]
                        b_key = jax.random.fold_in(step_key, geom_idx)
                        loss_c, B_r, grads = _do_one_batch_grad(raw, sfns, b_key, lam_ep)
                        grads_list.append(grads)
                        geom_idxs.append(geom_idx)
                        branch_losses.append(loss_c / max(B_r, 1))
                        loss_acc += loss_c; B_acc += B_r

                    if _gradnorm:
                        # w_i mis a jour avant de moyenner ce meta-step, comme dans le
                        # papier. Indexation par geom_idxs et non par position : un
                        # meta-step de fin d'epoque peut ne pas contenir toutes les branches.
                        idxs_arr = np.array(geom_idxs)
                        Lt = np.array(branch_losses, dtype=np.float64)
                        missing_l0 = np.isnan(_gn_l0[idxs_arr])
                        if missing_l0.any():
                            _gn_l0[idxs_arr[missing_l0]] = Lt[missing_l0] + 1e-8
                        g_i = np.array([_shared_grad_norm(g) for g in grads_list])
                        r_i = Lt / _gn_l0[idxs_arr]
                        r_i = r_i / (r_i.mean() + 1e-8)          # taux d'entrainement relatif
                        w_cur = _gn_w[idxs_arr]
                        Gw = w_cur * g_i                          # norme de gradient ponderee courante
                        target = Gw.mean() * (r_i ** cfg.gradnorm_alpha)  # traite comme constante
                        grad_w = np.sign(Gw - target) * g_i       # d|Gw_i - target_i|/dw_i (L1)
                        w_new = np.clip(w_cur - cfg.gradnorm_lr * grad_w, 1e-3, None)
                        w_new *= len(idxs_arr) / w_new.sum()      # renormalise sum(w)=n_branches presentes
                        _gn_w[idxs_arr] = w_new
                        w_list = list(w_new)
                    else:
                        w_list = [train_ds.weights[gi] for gi in geom_idxs]
                    wsum = sum(w_list)
                    w_norm = [w / wsum for w in w_list]
                    avg_grads = jax.tree.map(
                        lambda *gs: sum(w * g for w, g in zip(w_norm, gs)), *grads_list)
                    gn = _tree_grad_norm(avg_grads)
                    _apply_update(optimizer, avg_grads)
                    ep_loss += loss_acc; n += B_acc; _n_grad_steps += 1
                    _ep_gnorm_sum += gn; _ep_gnorm_n += 1
                    if cfg.grad_clip > 0 and gn > cfg.grad_clip:
                        _ep_n_clipped += 1
                    if ema is not None:
                        ema = _ema_update(ema, nnx.state(optimizer.model, nnx.Param))
            elif _is_multi:
                # Mode sequentiel : un update = un micro-batch d'UNE SEULE branche par
                # step, sfns choisi selon la geometrie du batch tire.
                for raw, geom_idx in train_ds.iter_batches(batch_size, rng):
                    sfns = _all_step_fns[_all_names[geom_idx]]
                    step_key = jax.random.fold_in(base_key, _n_grad_steps)
                    loss_c, B_r, gn = _do_one_batch(raw, sfns, step_key, lam_ep)
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
                    step_key = jax.random.fold_in(base_key, _n_grad_steps)
                    loss_c, B_r, gn = _do_one_batch(raw, sfns, step_key, lam_ep)
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
                if ema is not None:
                    nnx.update(eval_model, ema)
                metrics = _validate(eval_model, _all_names, _all_val,
                                    _all_step_fns, train_ds, _is_multi,
                                    max_samples=self.val_max_samples)
                vl = metrics['vl']
                val_losses.append(vl)
                val_l2s.append([*metrics['vl2'].tolist(), metrics['vmach']])

                # save_best sur la L2 physique (vraie métrique cible), pas la loss val
                score = float(metrics['vl2'].mean())
                is_best = cfg.save_best and score < best_val
                if is_best:
                    best_val = score
                    _, best_state = nnx.split(eval_model)

                eval_model.save(out_dir / f'{self.__class__.__name__}.pkl', cfg=model_cfg)

                # Export des courbes en direct à chaque validation
                plot_training_curves(train_losses, val_losses, val_l2s, out_dir / 'training_curves.png',
                                     lr_track=lr_track, idw_l2_ref=idw_ref)

                if pred_callback is not None:
                    pred_callback(eval_model, epoch, _primary_knn)

                _elapsed = round(time.time() - t0, 1)
                _log_epoch(epoch, cfg, metrics, train_loss=train_losses[-1], lr_now=lr_now, elapsed=_elapsed,
                        is_best=is_best, mem=_get_mem(), gnorm_sum=_ep_gnorm_sum, gnorm_n=_ep_gnorm_n,
                        n_clipped=_ep_n_clipped, idw_ref=idw_ref, idw_refs=_idw_refs, all_names=_all_names,
                        is_multi=_is_multi, has_flow=_has_flow, csv_path=_csv_path, csv_fields=_csv_fields,
                        gn_w=(dict(zip(_all_names, _gn_w.tolist())) if _gradnorm else None))

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

    from preprocessing.knn import build

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
