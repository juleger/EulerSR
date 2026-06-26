"""evaluate() unifié : cas de référence + sweep complet pour n'importe quel TestSet."""
from __future__ import annotations
import csv
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from eval.loader import ModelEntry
from eval.testset import TestSet
from eval.core import (mesh_geom_from_case, build_hr_feat, build_lr_feat, build_features,
                       predict_det, predict_idw, make_batch_predict, run_batched, _BATCH_SIZE)
from utils.refs import to_mach
from utils.metrics import (compute_field_errors, l2_rel, aero_metrics,
                            enthalpy_rms, entropy_violation, fvm_euler_rms,
                            idw_weights)
from utils.aero import aero_coeffs
from utils.gt_cache import load_or_build_gt_cache

_PRIM_NAMES = ['rho', 'u', 'v', 'p']


def _train_stats(entry: ModelEntry, ts_stats: dict) -> tuple[np.ndarray, np.ndarray]:
    """Retourne (mu, sig) à utiliser pour normaliser les données.

    Règles :
    - Modèle multi-géométrie : chaque géométrie a ses propres stats stockées dans le dataset (ts_stats)
    - Modèle single-géométrie : les stats d'entraînement (mu_train/sig_train stockées dans le modèle) sont utilisées
    """
    cfg = entry.cfg or {}
    if cfg.get('datasets'):
        # Multi-géo : ts_stats est la bonne référence pour la géométrie courante
        return ts_stats['mu'].astype(np.float32), ts_stats['sig'].astype(np.float32)
    # Single-géo : utiliser les stats d'entraînement stockées dans le modèle
    m = entry.model
    if hasattr(m, 'mu_train') and hasattr(m, 'sig_train'):
        return np.asarray(m.mu_train.value), np.asarray(m.sig_train.value)
    return ts_stats['mu'].astype(np.float32), ts_stats['sig'].astype(np.float32)


def _maybe_recalibrate(models: list[ModelEntry], ts: TestSet, knn_map: dict[str, dict], n_calib: int = 32) -> None:
    """Recalibre res_scale des modèles FAM si OOD résolution ou OOD géométrique.

    Utilise les stats d'entraînement du modèle (mu_train/sig_train) pour normaliser les features et les targets.
    """
    calib_cases = (ts.test_cases or ts.ref_cases)[:n_calib]
    if not calib_cases:
        return
    for entry in models:
        if type(entry.model).__name__ != 'FAM':
            continue
        ood_kind = ts.ood_kind(entry)
        if ood_kind == 'indistrib':
            continue
        mu, sig = _train_stats(entry, ts.stats)
        trained_lr = (entry.cfg or {}).get('resolution', {}).get('lr', 0.1)
        print(f"\n── Recalibration res_scale [{entry.name}]"
              f"  ({ood_kind})  LR train={trained_lr} → eval={ts.lr_res} ──")
        lr_feats, tg_list = [], []
        for c in calib_cases:
            d = dict(np.load(c['path']))
            if 'hr_primitives' not in d:
                continue
            lr_feats.append(build_lr_feat(mesh_geom_from_case(d), d, mu, sig))
            tg_list.append((d['hr_primitives'].astype(np.float32) - mu) / sig)
        if lr_feats:
            entry.model.recalibrate_res_scale(lr_feats, tg_list, knn_map[entry.name])


def evaluate(models: list[ModelEntry], ts: TestSet, out_dir: Path,
             n_eval: int = 0, batch_size: int = _BATCH_SIZE) -> dict:
    #Évalue tous les modèles sur un TestSet : cas de référence + sweep complet métriques
    out = Path(out_dir) / ts.tag
    out.mkdir(parents=True, exist_ok=True)

    _print_header(ts, models)

    # KNN par modèle pour cette géométrie/résolution
    print("\n── Construction des KNN ────────────────────────────────────")
    knn_map: dict[str, dict] = {}
    for entry in models:
        knn_map[entry.name] = ts.build_knn(entry)
        print(f"  [{entry.name}]  {ts.ood_label(entry)}")

    # KNN IDW dédié : toujours reconstruit sur la géométrie/résolution du TestSet
    print("  [IDW]  construction kNN simple...")
    idw_knn = ts.build_idw_knn(k=6)

    _maybe_recalibrate(models, ts, knn_map)

    ref_results = None
    sweep_results = None

    ts_label = ts.ts_title(models)

    if ts.ref_cases:
        print("\n── Cas de référence ────────────────────────────────────────")
        ref_results = _run_ref_cases(models, ts, knn_map, idw_knn)
        ref_results['_ts_label'] = ts_label
        _plot_ref(ref_results, ts, out)

    if ts.test_cases:
        cases = ts.test_cases[:n_eval] if n_eval > 0 else ts.test_cases
        print(f"\n── Sweep test set ({len(cases)} cas) ──"
              f"───────────────────────────────")
        mesh_hr = np.load(ts.layout.mesh_path(ts.hr_res), allow_pickle=True).item()
        mu_gt = ts.stats['mu'].astype(np.float64)
        gt_cache = load_or_build_gt_cache(ts.layout, cases, ts.wc,
                                          mesh=mesh_hr, mu=mu_gt)
        sweep_results = _run_sweep(models, ts, knn_map, cases, gt_cache, batch_size, idw_knn)
        sweep_results['_ts_label'] = ts_label
        _plot_sweep(sweep_results, cases, ts, out)
        _save_summary_csv(ref_results, sweep_results, out / 'summary.csv')

    print(f"\nDone — {out}/")
    return {'ref': ref_results, 'sweep': sweep_results}


def _run_ref_cases(models: list[ModelEntry], ts: TestSet, knn_map: dict[str, dict], idw_knn: dict) -> dict:
    """Inférence + métriques sur les cas de référence."""
    wc = ts.wc

    dxy_edges = np.linalg.norm(
        wc.bary[wc.cell_adj_edges[:, 0]] - wc.bary[wc.cell_adj_edges[:, 1]], axis=1
    ).astype(np.float32)

    # Pré-chargement des cas
    case_data = []
    for c in ts.ref_cases:
        d = np.load(c['path'])
        hr_prim = d['hr_primitives'].astype(np.float32) if 'hr_primitives' in d else None
        mach_ref = to_mach(hr_prim) if hr_prim is not None else None
        wm_hr = (aero_coeffs(hr_prim, wc, c['mach_in'])['wall_mach']
                    if hr_prim is not None else None)
        case_data.append({'meta': c, 'd': d, 'hr_prim': hr_prim,
                          'mach_ref': mach_ref, 'wall_mach_hr': wm_hr})

    # Warmup JIT sur le premier cas (geom_id par modèle)
    if case_data:
        _d0, _c0 = case_data[0]['d'], case_data[0]['meta']
        for entry in models:
            _gid = ts.geom_id_for(entry)
            _mu, _sig = _train_stats(entry, ts.stats)
            _hf, _lf, *_ = build_features(_d0, _c0['mach_in'], _c0['aoa_in'],
                                           {'mu': _mu, 'sig': _sig}, _gid)
            print(f"  [{entry.name}] warmup JIT ({ts.tag})...")
            jax.block_until_ready(entry.model.predict(_hf, _lf, knn_map[entry.name]))

    def _field_errs(prim, cd):
        if cd['hr_prim'] is None:
            return None, None
        er = compute_field_errors(prim, cd['hr_prim'],
                                  wc.cell_adj_edges, wc.bary, dxy_edges)
        return er['linf_mach'], er['l2w_mach']

    def _aero(prim, cd):
        if cd['hr_prim'] is None:
            return None, None
        am = aero_metrics(prim, cd['hr_prim'], wc, cd['meta']['mach_in'])
        return am['wall_mach_pred'], am

    # IDW baseline
    idw_row = {'name': 'LR IDW', 'prim_preds': [], 'mach_preds': [],
                'l2': [], 'linf': [], 'l2w': [], 'time_ms': [],
                'wall_mach': [], 'aero': []}
    if idw_knn is not None and 'idx' in idw_knn:
        idx_idw = np.asarray(idw_knn['idx'])[:, :6].astype(np.int32)
        w_idw = idw_weights(np.asarray(idw_knn['dist'])[:, :6]).astype(np.float32)
        for cd in case_data:
            lr_p = cd['d']['lr_primitives'].astype(np.float32)
            t0 = time.perf_counter()
            prim = (w_idw[:, :, None] * lr_p[idx_idw]).sum(axis=1)
            t = (time.perf_counter() - t0) * 1e3
            mach = to_mach(prim)
            li, lw = _field_errs(prim, cd)
            wm, am = _aero(prim, cd)
            idw_row['prim_preds'].append(prim)
            idw_row['mach_preds'].append(mach)
            idw_row['l2'].append(l2_rel(mach, cd['mach_ref'])
                                 if cd['mach_ref'] is not None else None)
            idw_row['linf'].append(li); idw_row['l2w'].append(lw)
            idw_row['time_ms'].append(t)
            idw_row['wall_mach'].append(wm); idw_row['aero'].append(am)

    # Modèles
    model_rows = []
    for entry in models:
        gid = ts.geom_id_for(entry)
        _mu_e, _sig_e = _train_stats(entry, ts.stats)
        _stats_e = {'mu': _mu_e, 'sig': _sig_e}
        row = {'name': ts.display_name(entry), 'prim_preds': [], 'mach_preds': [],
               'l2': [], 'linf': [], 'l2w': [], 'time_ms': [],
               'wall_mach': [], 'aero': []}
        for cd in case_data:
            c = cd['meta']
            prim, t = predict_det(entry, cd['d'], c['mach_in'], c['aoa_in'],
                                   _stats_e, knn=knn_map[entry.name], geom_id=gid)
            mach = to_mach(prim)
            li, lw = _field_errs(prim, cd)
            wm, am = _aero(prim, cd)
            row['prim_preds'].append(prim)
            row['mach_preds'].append(mach)
            row['l2'].append(l2_rel(mach, cd['mach_ref'])
                             if cd['mach_ref'] is not None else None)
            row['linf'].append(li); row['l2w'].append(lw)
            row['time_ms'].append(t)
            row['wall_mach'].append(wm); row['aero'].append(am)
        model_rows.append(row)
        l2s = [f'{v:.3f}' if v is not None else 'N/A' for v in row['l2']]
        print(f"  [{entry.name}]  L2={l2s}  moy {np.mean(row['time_ms']):.1f}ms/cas")

    return {
        'cases': [cd['meta'] | {'mach_ref': cd['mach_ref'],
                                 'hr_prim': cd['hr_prim'],
                                 'wall_mach_hr': cd['wall_mach_hr']}
                  for cd in case_data],
        'rows': model_rows,
        'idw': idw_row if (idw_knn is not None and 'idx' in idw_knn) else None,
    }


def _run_sweep(models: list[ModelEntry], ts: TestSet,
               knn_map: dict[str, dict], cases: list[dict],
               gt_cache: dict | None, batch_size: int,
               idw_knn: dict | None = None) -> dict:
    """Inférence batch + métriques sur le test set complet."""
    wc = ts.wc
    mu = ts.stats['mu'].astype(np.float32)
    sig = ts.stats['sig'].astype(np.float32)
    n = len(cases)

    det_models = [m for m in models if m.kind == 'det']
    method_names = (['LR IDW'] if idw_knn is not None else []) +                   [ts.display_name(m) for m in models]

    results: dict[str, dict] = {
        nm: {'times': [], 'CL': [], 'CD': [], 'grad_p_max': [],
             'mach_max': [], 'mach_min': [], 'l2_mach': [], 'linf_mach': [],
             'l2w_mach': [], 'sw2_mach': [], 'l2_prims': [], 'linf_prims': [],
             'wall_mach': [], 'has_ref': [], 'enthalpy': [], 'entropy': [],
             'euler_fvm': []}
        for nm in method_names
    }
    results['HR'] = {'CL': [], 'CD': [], 'grad_p_max': [],
                     'mach_max': [], 'mach_min': [], 'wall_mach': [],
                     'enthalpy_hr': [], 'entropy_hr': [], 'euler_fvm_hr': []}

    _knn_mesh = next((knn_map[m.name] for m in det_models
                      if knn_map.get(m.name) and 'mesh' in knn_map[m.name]), None)
    _has_mesh = _knn_mesh is not None
    _mesh_obj = _knn_mesh['mesh'] if _has_mesh else None

    print(f"  Chargement parallèle de {n} fichiers...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        all_data: list[dict] = list(ex.map(lambda c: dict(np.load(c['path'])), cases))

    print("  Pré-calcul des features...")
    geom = mesh_geom_from_case(all_data[0])
    # lr_feats avec ts.stats (pour IDW et modèles in-distrib), les modèles
    # OOD géométrique reconstruisent leurs propres features avec leurs training stats.
    lr_feats = [build_lr_feat(geom, d, mu, sig) for d in all_data]

    dxy_edges = np.linalg.norm(
        wc.bary[wc.cell_adj_edges[:, 0]] - wc.bary[wc.cell_adj_edges[:, 1]], axis=1
    ).astype(np.float32)

    # IDW
    idw_preds: list[np.ndarray] = []
    idw_t_ms_per_case: float = 0.0
    if idw_knn is not None:
        idx_idw = np.asarray(idw_knn['idx'])[:, :6].astype(np.int32)
        w_idw = idw_weights(np.asarray(idw_knn['dist'])[:, :6]).astype(np.float32)
        t0 = time.perf_counter()
        for d in all_data:
            lr_p = d['lr_primitives'].astype(np.float32)
            idw_preds.append((w_idw[:, :, None] * lr_p[idx_idw]).sum(1))
        idw_t_ms_per_case = (time.perf_counter() - t0) * 1e3 / max(n, 1)

    # Inférence batch par modèle, hr_feats et stats calculés par modèle
    det_preds: dict[str, list[np.ndarray]] = {}
    det_t_ms: dict[str, float] = {}
    for entry in det_models:
        nm = ts.display_name(entry)
        gid = ts.geom_id_for(entry)
        mu_e, sig_e = _train_stats(entry, ts.stats)

        # Temps de construction des features
        t0_feat = time.perf_counter()
        if np.allclose(mu_e, mu) and np.allclose(sig_e, sig):
            lrf = lr_feats
            t0_lr = time.perf_counter()
            _ = build_lr_feat(geom, all_data[0], mu_e, sig_e)
            feat_ms_per_case = (time.perf_counter() - t0_lr) * 1e3
        else:
            lrf = [build_lr_feat(geom, d, mu_e, sig_e) for d in all_data]
            feat_ms_per_case = (time.perf_counter() - t0_feat) * 1e3 / n
        t0_hr = time.perf_counter()
        hr_feats = [build_hr_feat(geom, c['mach_in'], c['aoa_in'], gid) for c in cases]
        feat_ms_per_case += (time.perf_counter() - t0_hr) * 1e3 / n

        print(f"  [{entry.name}] warmup + inférence batch (B={batch_size})...")
        bfn = make_batch_predict(entry.model, knn_map[entry.name])
        dup = max(0, batch_size - n)
        warm_h = hr_feats[:min(batch_size, n)] + [hr_feats[0]] * dup
        warm_l = lrf[:min(batch_size, n)] + [lrf[0]] * dup
        dummy_h = jnp.array(np.stack(warm_h))
        dummy_l = jnp.array(np.stack(warm_l))
        bfn(dummy_h, dummy_l).block_until_ready()
        preds, infer_ms = run_batched(bfn, hr_feats, lrf, batch_size, mu_e, sig_e)
        det_preds[nm] = preds
        det_t_ms[nm] = infer_ms + feat_ms_per_case
        print(f"    → {infer_ms:.2f} ms/cas inférence  +  {feat_ms_per_case:.2f} ms/cas features"
              f"  =  {det_t_ms[nm]:.2f} ms/cas total")

    def _fill(nm: str, prim: np.ndarray, t_ms: float, hr_prim: np.ndarray | None, has_ref: bool,
               mach_in: float, aoa_in: float = 0.0):
        r = results[nm]
        ac = aero_coeffs(prim, wc, mach_in)
        r['times'].append(t_ms)
        r['has_ref'].append(has_ref)
        r['CL'].append(ac['CL']); r['CD'].append(ac['CD'])
        r['grad_p_max'].append(ac['grad_p_max'])
        r['mach_max'].append(ac['mach_max']); r['mach_min'].append(ac['mach_min'])
        r['wall_mach'].append(ac['wall_mach'])
        r['enthalpy'].append(enthalpy_rms(prim, mach_in))
        r['entropy'].append(entropy_violation(prim))
        r['euler_fvm'].append(fvm_euler_rms(prim, _mesh_obj, mach_in, aoa_in, mu)
                              if _has_mesh else np.nan)
        if has_ref:
            er = compute_field_errors(prim, hr_prim, wc.cell_adj_edges, wc.bary, dxy_edges)
            r['l2_mach'].append(er['l2_mach'])
            r['linf_mach'].append(er['linf_mach'])
            r['l2w_mach'].append(er['l2w_mach'])
            r['sw2_mach'].append(er['sw2_mach'])
            r['l2_prims'].append([er[f'l2_{p2}'] for p2 in _PRIM_NAMES])
            r['linf_prims'].append([er[f'linf_{p2}'] for p2 in _PRIM_NAMES])
        else:
            for k2 in ('l2_mach', 'linf_mach', 'l2w_mach', 'sw2_mach'):
                r[k2].append(None)
            r['l2_prims'].append(None)
            r['linf_prims'].append(None)

    print("  Calcul métriques + aéro...")
    gc = gt_cache
    for i, (c, d) in enumerate(zip(cases, all_data)):
        if (i + 1) % 200 == 0 or i == 0:
            print(f"    {i+1}/{n}...")
        aoa_in = c.get('aoa_in', 0.0)

        if gc is not None and gc['has_ref'][i]:
            hr_prim, has_ref = gc['hr_prim'][i], True
        else:
            raw = d.get('hr_primitives')
            has_ref = raw is not None
            hr_prim = raw.astype(np.float32) if has_ref else None

        gt = results['HR']
        if has_ref:
            if gc is not None:
                for k2 in ('CL', 'CD', 'grad_p_max', 'mach_max', 'mach_min'):
                    gt[k2].append(float(gc[k2][i]))
                gt['wall_mach'].append(gc['wall_mach'][i])
                gt['enthalpy_hr'].append(float(gc.get('enthalpy_hr', [np.nan] * n)[i]))
                gt['entropy_hr'].append(float(gc.get('entropy_hr', [np.nan] * n)[i]))
            else:
                ac = aero_coeffs(hr_prim, wc, c['mach_in'])
                for k2 in ('CL', 'CD', 'grad_p_max', 'mach_max', 'mach_min'):
                    gt[k2].append(ac[k2])
                gt['wall_mach'].append(ac['wall_mach'])
                gt['enthalpy_hr'].append(enthalpy_rms(hr_prim, c['mach_in']))
                gt['entropy_hr'].append(entropy_violation(hr_prim))
            gt['euler_fvm_hr'].append(fvm_euler_rms(hr_prim, _mesh_obj, c['mach_in'], aoa_in, mu)
                if _has_mesh else np.nan)
        else:
            for k2 in ('CL', 'CD', 'grad_p_max', 'mach_max', 'mach_min'):
                gt[k2].append(None)
            gt['wall_mach'].append(np.array([], dtype=np.float32))
            gt['enthalpy_hr'].append(np.nan)
            gt['entropy_hr'].append(np.nan)
            gt['euler_fvm_hr'].append(np.nan)

        if idw_knn is not None:
            _fill('LR IDW', idw_preds[i], idw_t_ms_per_case,
                  hr_prim, has_ref, c['mach_in'], aoa_in)
        for entry in det_models:
            nm = ts.display_name(entry)
            _fill(nm, det_preds[nm][i], det_t_ms[nm],
                  hr_prim, has_ref, c['mach_in'], aoa_in)

    # Conversion listes en arrays
    for nm, r in results.items():
        if not isinstance(r, dict):
            continue
        for k2 in list(r.keys()):
            if isinstance(r[k2], list) and k2 not in ('l2_prims', 'linf_prims', 'wall_mach'):
                r[k2] = np.array([v if v is not None else np.nan for v in r[k2]], dtype=float)

    # Erreurs aéro vs GT
    gt_CL = results['HR']['CL']
    gt_CD = results['HR']['CD']
    gt_wall_mach = results['HR']['wall_mach']
    for nm in method_names:
        r = results[nm]
        r['CL_err'] = np.abs(r['CL'] - gt_CL)
        r['CD_err'] = np.abs(r['CD'] - gt_CD)
        wall_mach_l2 = []
        for wm_p, wm_r in zip(r['wall_mach'], gt_wall_mach):
            if len(wm_p) > 0 and len(wm_r) > 0:
                wall_mach_l2.append(float(
                    np.linalg.norm(wm_p - wm_r) / (np.linalg.norm(wm_r) + 1e-8)))
            else:
                wall_mach_l2.append(np.nan)
        r['wall_mach_l2'] = np.array(wall_mach_l2, dtype=float)

    results['_cases'] = cases
    results['_name_map'] = {'LR IDW': 'LR IDW'}
    results['_name_map'].update({ts.display_name(m): m.name for m in models})
    _print_phys_table(results, method_names, _has_mesh)
    return results

def _plot_ref(results: dict, ts: TestSet, out: Path):
    from utils.viz.eval import plot_reference_grid, plot_wall_profiles_eval
    from eval.core import save_metrics_csv

    print("\n── Plots : cas de référence ────────────────────────────────")
    plot_reference_grid(results, ts.triang_hr, out / 'reference_grid.png')
    plot_wall_profiles_eval(results, ts.wc, out)
    save_metrics_csv(results, out / 'metrics_ref.csv')


def _plot_sweep(results: dict, cases: list[dict], ts: TestSet, out: Path):
    from utils.viz.eval import (
        plot_global_errors, plot_distributions, plot_aero_distributions,
        plot_cl_cd_distributions, plot_error_kde, plot_error_scatter,
        plot_error_heatmap, plot_regime_bars, plot_summary_table,
    )
    print("\n── Plots : sweep ───────────────────────────────────────────")
    plot_global_errors(results, out)
    plot_distributions(results, out)
    plot_aero_distributions(results, out)
    plot_cl_cd_distributions(results, out)
    plot_error_kde(results, out)
    plot_error_scatter(results, cases, out)
    plot_error_scatter(results, cases, out, metric_key='sw2_mach',
             metric_label=r'$SW_2$', fname='error_vs_mach_sw2.png')
    plot_error_heatmap(results, cases, out)
    plot_regime_bars(results, cases, out)
    plot_summary_table({}, results, out)

def _print_header(ts: TestSet, models: list[ModelEntry]):
    sep = '─' * 58
    print(f"\n┌{sep}┐")
    print(f"│  TestSet : {ts.label:<44}│")
    print(f"│  LR h={ts.lr_res:<6g}  HR h={ts.hr_res:<6g}"
          f"  {len(ts.ref_cases)} cas ref  {len(ts.test_cases)} cas test  │")
    print(f"│  Modèles : {', '.join(m.name for m in models):<44}│")
    print(f"└{sep}┘")


def _print_phys_table(results: dict, method_names: list[str], has_mesh: bool):
    _w = 16
    _cols = []
    if has_mesh:
        _cols.append(('Euler FVM (RMS)', 'euler_fvm', 'euler_fvm_hr'))
    _cols += [('Enthalpie (RMS)', 'enthalpy', 'enthalpy_hr'),
              ('Entropie (RMS)', 'entropy', 'entropy_hr')]
    print("\n  ── Résidus physiques (test set) " + "─" * 28)
    hdr = f"  {'Méthode':<22}" + "".join(f"  {c[0]:>{_w}}" for c in _cols)
    print(hdr); print("  " + "─" * (len(hdr) - 2))

    def _row(label, vals):
        parts = []
        for v in vals:
            arr = np.asarray(v, dtype=float)
            m = float(np.nanmean(arr))
            parts.append(f"{m:>{_w}.3e}" if np.isfinite(m) else f"{'N/A':>{_w}}")
        print(f"  {label:<22}" + "".join(f"  {p}" for p in parts))

    _row('HR GT', [results['HR'][c[2]] for c in _cols])
    for nm in method_names:
        _row(nm, [results[nm][c[1]] for c in _cols])


# ── Figures combinées multi-TestSet ───────────────────────────────────────────

def _merge_sweep_results(sweep_list: list[dict]) -> dict:
    """Fusionne les sweep results de plusieurs TestSets sous les noms de base.

    Le `_name_map` de chaque sweep permet de normaliser les clés :
    'FAM (OOD géométrique)' et 'FAM' deviennent tous les deux 'FAM'.
    """
    if not sweep_list:
        return {}

    # Ordre stable des méthodes de base (union dans l'ordre de première apparition)
    base_names: list[str] = []
    seen: set[str] = set()
    for r in sweep_list:
        name_map = r.get('_name_map', {})
        for disp in r:
            if disp.startswith('_') or disp == 'HR':
                continue
            base = name_map.get(disp, disp)
            if base not in seen:
                base_names.append(base)
                seen.add(base)

    _LIST_KEYS = {'wall_mach', 'l2_prims', 'linf_prims'}

    def _gather(key: str, sweep_dicts: list[dict]) -> list:
        """Collecte les valeurs d'une clé à travers tous les sweeps pour une méthode."""
        out = []
        for r in sweep_dicts:
            v = r.get(key)
            if v is None:
                continue
            if key in _LIST_KEYS:
                out.extend(v)
            elif isinstance(v, np.ndarray):
                out.append(v)
        return out

    merged: dict = {}

    # GT HR
    hr_dicts = [r['HR'] for r in sweep_list if 'HR' in r]
    merged['HR'] = {}
    for key in next(iter(hr_dicts), {}):
        if key in _LIST_KEYS:
            merged['HR'][key] = []
            for r in hr_dicts:
                merged['HR'][key].extend(r.get(key, []))
        else:
            arrs = [r[key] for r in hr_dicts
                    if isinstance(r.get(key), np.ndarray)]
            if arrs:
                merged['HR'][key] = np.concatenate(arrs)

    # Méthodes
    for base in base_names:
        # Sweep dicts qui contiennent ce nom de base (sous n'importe quel display name)
        sub: list[dict] = []
        for r in sweep_list:
            nm_map = r.get('_name_map', {})
            disp = next((d for d, b in nm_map.items() if b == base), None)
            if disp and disp in r:
                sub.append(r[disp])

        if not sub:
            continue

        merged[base] = {}
        for key in sub[0]:
            vals = _gather(key, sub)
            if not vals:
                continue
            if key in _LIST_KEYS:
                merged[base][key] = vals
            else:
                try:
                    merged[base][key] = np.concatenate(vals)
                except (ValueError, TypeError):
                    pass

    merged['_cases'] = []
    for r in sweep_list:
        merged['_cases'].extend(r.get('_cases', []))

    return merged


def plot_combined(ts_results: dict[str, dict], out_dir: Path) -> None:
    """Figures globales agrégées sur tous les TestSets évalués.

    Appelé automatiquement quand il y a plusieurs TestSets avec un sweep complet.
    Les noms de modèles sont normalisés (nom de base, sans suffixe OOD)
    """
    from utils.viz.eval import (plot_global_errors, plot_distributions,
                                 plot_error_kde, plot_cl_cd_distributions,
                                 plot_aero_distributions)

    sweep_list = [v['sweep'] for v in ts_results.values()
                  if v.get('sweep') is not None]
    if len(sweep_list) < 2:
        return

    tags = [tag for tag, v in ts_results.items() if v.get('sweep') is not None]
    print(f"\n── Figures combinées ({' + '.join(tags)}) "
          f"{'─' * max(0, 46 - len(' + '.join(tags)))}")

    merged = _merge_sweep_results(sweep_list)
    out = Path(out_dir) / 'combined'
    out.mkdir(exist_ok=True)

    plot_global_errors(merged, out)
    plot_distributions(merged, out)
    plot_aero_distributions(merged, out)
    plot_cl_cd_distributions(merged, out)
    plot_error_kde(merged, out)

    print(f"  -> {out}/")

def _save_summary_csv(ref_results: dict | None, sweep_results: dict | None,
                      path: Path):
    from utils.viz.eval import save_summary_csv
    save_summary_csv(ref_results or {}, sweep_results, path)
