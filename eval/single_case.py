"""Orchestration "cas unique" : inférence de chaque modèle sur EXACTEMENT un
cas, timing robuste (médiane, warm-up JIT exclu), métriques complètes.
Retourne un dict de même forme que eval.runner._run_ref_cases (un seul cas)
pour réutiliser tel quel les fonctions de plot de utils.viz.eval.
"""
from __future__ import annotations
from typing import Callable

import numpy as np
import jax

from eval.loader import ModelEntry
from eval.testset import TestSet
from eval.core import build_features, predict_det, predict_ensemble, predict_idw
from utils.refs import to_mach
from utils.metrics import compute_field_errors, l2_rel, aero_metrics


def _train_stats(entry: ModelEntry, ts_stats: dict) -> tuple[np.ndarray, np.ndarray]:
    """Mêmes règles que eval.runner._train_stats (fonction privée du module
    runner, dupliquée ici plutôt qu'importée pour ne pas alourdir l'import)."""
    cfg = entry.cfg or {}
    if cfg.get('datasets'):
        return ts_stats['mu'].astype(np.float32), ts_stats['sig'].astype(np.float32)
    m = entry.model
    if hasattr(m, 'mu_train') and hasattr(m, 'sig_train'):
        return np.asarray(m.mu_train.value), np.asarray(m.sig_train.value)
    return ts_stats['mu'].astype(np.float32), ts_stats['sig'].astype(np.float32)


def _step_info(model) -> str | None:
    """Résumé de la config d'échantillonnage (nb de pas/échantillons), pour
    contextualiser le temps d'inférence : un DAM à 1 appel n'est pas
    comparable à un FAM/SIAM à plusieurs pas ODE/SDE."""
    parts = []
    n_steps = getattr(model, 'n_steps', None)
    n_sde_steps = getattr(model, 'n_sde_steps', None)
    n_samples = getattr(model, 'n_samples', None)
    sampler = getattr(model, 'sampler', None)
    if sampler == 'sde' and n_sde_steps is not None:
        parts.append(f'{n_sde_steps} pas SDE')
    elif n_steps is not None:
        parts.append(f'{n_steps} pas ODE')
    if n_samples is not None and n_samples > 1:
        parts.append(f'{n_samples} éch.')
    return ', '.join(parts) if parts else None


def _timed_repeat(call_fn: Callable[[], tuple], n_repeat: int):
    """Appelle call_fn() -> (résultat, t_ms) n_repeat fois (warm-up déjà fait
    en amont). Retourne (résultat du 1er appel, liste des n_repeat temps) —
    la médiane de cette liste est le temps représentatif à afficher, plus
    robuste au bruit qu'un tir unique."""
    times = []
    result0 = None
    for i in range(max(1, n_repeat)):
        result, t_ms = call_fn()
        if i == 0:
            result0 = result
        times.append(t_ms)
    return result0, times


def run_single_case(models: list[ModelEntry], ts: TestSet, case: dict, d: dict,
                     idw_knn: dict | None, knn_map: dict[str, dict],
                     n_repeat: int = 5) -> dict:
    """Inférence + métriques de tous les modèles sur un cas. 'case' : dict
    {'path', 'mach_in', 'aoa_in', 'label', ...}. 'd' : échantillon déjà chargé
    via utils.layout.load_sample."""
    wc = ts.wc
    dxy_edges = np.linalg.norm(
        wc.bary[wc.cell_adj_edges[:, 0]] - wc.bary[wc.cell_adj_edges[:, 1]], axis=1
    ).astype(np.float32)

    hr_prim = d['hr_primitives'].astype(np.float32) if 'hr_primitives' in d else None
    mach_ref = to_mach(hr_prim) if hr_prim is not None else None

    def _field_errs(prim):
        if hr_prim is None:
            return None, None, None
        er = compute_field_errors(prim, hr_prim, wc.cell_adj_edges, wc.bary, dxy_edges)
        return er['linf_mach'], er['l2w_mach'], er['w2_mach']

    def _aero(prim):
        if hr_prim is None:
            return None, None
        am = aero_metrics(prim, hr_prim, wc, case['mach_in'])
        return am['wall_mach_pred'], am

    print(f"\n── Cas unique : {ts.geometry}  M={case['mach_in']:.2f}  "
          f"AoA={case['aoa_in']:+.1f}°  ({case.get('label', '')}) ──")

    # Warm-up JIT par modèle, exclu du timing (cf. eval/runner.py:145-154)
    for entry in models:
        gid = ts.geom_id_for(entry)
        mu_e, sig_e = _train_stats(entry, ts.stats)
        coord_norm = (entry.cfg or {}).get('architecture', {}).get('coord_norm', 'domain')
        hf, lf, *_ = build_features(d, case['mach_in'], case['aoa_in'],
                                     {'mu': mu_e, 'sig': sig_e}, gid,
                                     coord_norm, ts.hr_mesh_meta)
        print(f"  [{entry.name}] warmup JIT...")
        jax.block_until_ready(entry.model.predict(hf, lf, knn_map[entry.name]))

    # Baseline LR -> HR par IDW
    idw_row = None
    if idw_knn is not None and 'idx' in idw_knn:
        idx_idw = np.asarray(idw_knn['idx'])[:, :6].astype(np.int32)
        dist_idw = np.asarray(idw_knn['dist'])[:, :6].astype(np.float32)
        lr_p = d['lr_primitives'].astype(np.float32)

        prim0, times = _timed_repeat(
            lambda: predict_idw(lr_p, idx_idw, dist_idw, k=6), n_repeat)
        mach = to_mach(prim0)
        li, lw, w2v = _field_errs(prim0)
        wm, am = _aero(prim0)
        idw_row = {
            'name': 'LR IDW', 'prim_preds': [prim0], 'mach_preds': [mach],
            'l2': [l2_rel(mach, mach_ref) if mach_ref is not None else None],
            'linf': [li], 'l2w': [lw], 'w2': [w2v], 'time_ms': times,
            'wall_mach': [wm], 'aero': [am],
        }
        l2_disp = f'{idw_row["l2"][0]:.4f}' if idw_row['l2'][0] is not None else 'N/A'
        print(f"  [LR IDW]  L2={l2_disp}  médiane {np.median(times):.3f}ms "
              f"({n_repeat} tirs)")

    # Modèles
    model_rows = []
    for entry in models:
        gid = ts.geom_id_for(entry)
        mu_e, sig_e = _train_stats(entry, ts.stats)
        stats_e = {'mu': mu_e, 'sig': sig_e}
        is_sde = getattr(entry.model, 'sampler', 'ode') == 'sde'

        if is_sde:
            def _call(entry=entry, stats_e=stats_e, gid=gid):
                mean, _std, t_ms = predict_ensemble(
                    entry, d, case['mach_in'], case['aoa_in'], stats_e,
                    knn=knn_map[entry.name], geom_id=gid, mesh_meta=ts.hr_mesh_meta)
                return mean, t_ms
        else:
            def _call(entry=entry, stats_e=stats_e, gid=gid):
                return predict_det(entry, d, case['mach_in'], case['aoa_in'],
                                    stats_e, knn=knn_map[entry.name], geom_id=gid,
                                    mesh_meta=ts.hr_mesh_meta)

        prim0, times = _timed_repeat(_call, n_repeat)
        mach = to_mach(prim0)
        li, lw, w2v = _field_errs(prim0)
        wm, am = _aero(prim0)
        row = {
            'name': ts.display_name(entry), 'prim_preds': [prim0], 'mach_preds': [mach],
            'l2': [l2_rel(mach, mach_ref) if mach_ref is not None else None],
            'linf': [li], 'l2w': [lw], 'w2': [w2v], 'time_ms': times,
            'wall_mach': [wm], 'aero': [am], 'step_info': _step_info(entry.model),
        }
        model_rows.append(row)
        l2_disp = f'{row["l2"][0]:.4f}' if row['l2'][0] is not None else 'N/A'
        print(f"  [{entry.name}]  L2={l2_disp}  médiane {np.median(times):.2f}ms/cas "
              f"({n_repeat} tirs)")

    return {
        'cases': [case | {'mach_ref': mach_ref, 'hr_prim': hr_prim}],
        'rows': model_rows,
        'idw': idw_row,
    }
