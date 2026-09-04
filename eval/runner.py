"""evaluate() unifié : cas de référence + sweep complet pour n'importe quel TestSet."""
from __future__ import annotations
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from eval.loader import ModelEntry
from eval.testset import TestSet, _find_val_cases
from eval.core import (mesh_geom_from_case, build_hr_feat, build_lr_feat, build_wall_feat,
                       build_features, predict_det, predict_ensemble, predict_wall_baseline,
                       make_batch_predict, run_batched, _BATCH_SIZE, _resolve_mach_norm,
                       idw_baseline_name)
from utils.refs import to_mach
from utils.metrics import (compute_field_errors, l2_rel, aero_metrics,
                            enthalpy_rms, entropy_violation, fvm_euler_rms,
                            fvm_euler_step_rms, idw_weights)
from utils.aero import aero_coeffs
from utils.gt_cache import load_or_build_gt_cache
from utils.layout import load_sample

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


def _stratified_sample(cases: list[dict], n: int) -> list[dict]:
    """Sous-échantillon stratifié sur (aoa, mach), trié numériquement (pas
    lexicographiquement -- 'aoa-5.00' < 'aoa0.00' < 'aoa10.00' < 'aoa2.00' en
    tri de chaînes !). Même recette que _branch_res_std à l'entraînement
    (np.linspace sur les indices triés) : couvre tout l'intervalle du pool au
    lieu de se concentrer sur les premiers cas triés par nom de fichier.
    """
    if len(cases) <= n:
        return list(cases)
    ordered = sorted(cases, key=lambda c: (c['aoa_in'], c['mach_in']))
    idx = np.unique(np.linspace(0, len(ordered) - 1, n).astype(int))
    return [ordered[i] for i in idx]


def _maybe_recalibrate(models: list[ModelEntry], ts: TestSet, knn_map: dict[str, dict],
                       n_calib: int = 64) -> set[str]:
    """Recalibre res_scale des modèles FAM sur CE TestSet, systématiquement.

    Le buffer res_scale du checkpoint n'est qu'une valeur unique choisie parmi
    les branches d'entraînement (la plus proche de la résolution nominale, en
    cas d'égalité la première géométrie du yaml -- typiquement 'diamond' en
    multi-géométrie). Il ne reflète donc PAS le res_scale propre à chaque
    géométrie, y compris en indistribution (même géométrie/résolution vue à
    l'entraînement mais différente de celle qui a rempli le buffer). On
    recalibre donc pour tous les TestSets, indistrib compris, à partir d'un
    échantillon de cas -- plus fiable que le buffer.

    Pool de calibration, par ordre de préférence :
    - split 'val' de CETTE géométrie/résolution (layout.proc_dir()/val), s'il
      existe : totalement disjoint du sweep de test, aucune circularité.
    - à défaut (géométries OOD préprocessées en test_only=True, cf
      preprocessing/preprocess.py : pas de val/train), un sous-échantillon
      stratifié de ts.test_cases. Les cas ainsi utilisés sont retournés pour
      être exclus du sweep par evaluate() -- sinon on calibre res_scale sur
      des cas qu'on ensuite note, ce qui biaise favorablement leurs métriques.

    Utilise les stats d'entraînement du modèle (mu_train/sig_train) pour normaliser les features et les targets.
    """
    # Seuls les FAM à res_scale buffer sont concernés : avec learned_res_scale,
    # FAM._res_scale ignore le buffer qu'on recalibrerait ici, donc la calibration
    # serait sans effet tout en consommant des cas du sweep.
    needs_calib = [m for m in models
                   if type(m.model).__name__ == 'FAM' and not m.model._learned_res_scale]
    if not needs_calib:
        return set()

    val_cases = _find_val_cases(ts.layout)
    pool = val_cases or ts.test_cases or ts.ref_cases
    if not pool:
        return set()
    if val_cases:
        calib_cases = _stratified_sample(pool, n_calib)
        excluded = set()
    elif len(pool) <= 1:
        # Pool trop petit pour separer calibration et sweep : on accepte de calibrer
        # et de noter sur le meme cas plutot que de vider le sweep.
        calib_cases = pool
        excluded = set()
    else:
        # Pool = les cas de sweep eux-memes (geometrie ou resolution OOD sans split
        # 'val') : les exclure du sweep apres calibration. Plafonne a len(pool)-1 pour
        # qu'un petit pool ne soit pas entierement mange par la calibration.
        n_calib_capped = min(n_calib, len(pool) - 1)
        calib_cases = _stratified_sample(pool, n_calib_capped)
        excluded = {c['path'] for c in calib_cases}

    pool_aoa = [c['aoa_in'] for c in pool]
    calib_aoa = [c['aoa_in'] for c in calib_cases]
    print(f"  Calibration : {len(calib_cases)} cas, source="
          f"{'val (disjoint du sweep)' if val_cases else 'test (holdout stratifié, exclu du sweep)'}"
          f"  AoA calib=[{min(calib_aoa):.2f}, {max(calib_aoa):.2f}]"
          f"  vs pool=[{min(pool_aoa):.2f}, {max(pool_aoa):.2f}]")

    for entry in needs_calib:
        ood_kind = ts.ood_kind(entry)
        mu, sig = _train_stats(entry, ts.stats)
        trained_lr = (entry.cfg or {}).get('resolution', {}).get('lr', 0.1)
        coord_norm = (entry.cfg or {}).get('architecture', {}).get('coord_norm', 'object')
        print(f"\n── Recalibration res_scale [{entry.name}]"
              f"  ({ood_kind})  LR train={trained_lr} → eval={ts.lr_res} ──")
        lr_feats, tg_list = [], []
        for c in calib_cases:
            d = load_sample(c['path'], c.get('raw_hr_path'))
            if 'hr_primitives' not in d:
                continue
            geom = mesh_geom_from_case(d, coord_norm, ts.hr_mesh_meta)
            lr_feats.append(build_lr_feat(geom, d, mu, sig))
            tg_list.append((d['hr_primitives'].astype(np.float32) - mu) / sig)
        if lr_feats:
            entry.model.recalibrate_res_scale(lr_feats, tg_list, knn_map[entry.name])
    return excluded


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

    calib_excluded = _maybe_recalibrate(models, ts, knn_map)

    # KNN IDW dédié : toujours reconstruit sur la géométrie/résolution du TestSet.
    # wall=True (tous les modèles évalués sont bord-seul) : baseline pertinente =
    # IDW(bord) blend freestream, pas l'IDW volumique (pas la même info disponible).
    is_wall_eval = bool(models) and all(hasattr(m.model, 'wall_encoder') for m in models)
    print(f"  [{'baseline bord' if is_wall_eval else 'IDW'}]  construction kNN simple...")
    idw_knn = ts.build_idw_knn(k=6, wall=is_wall_eval)

    ref_results = None
    sweep_results = None

    ts_label = ts.ts_title(models)

    if ts.ref_cases:
        print("\n── Cas de référence ────────────────────────────────────────")
        ref_results = _run_ref_cases(models, ts, knn_map, idw_knn)
        ref_results['_ts_label'] = ts_label
        _plot_ref(ref_results, ts, out)

    if ts.test_cases:
        # Exclut les cas éventuellement consommés par _maybe_recalibrate (repli
        # sans split 'val') : jamais calibrer et noter sur les mêmes cas.
        test_cases = [c for c in ts.test_cases if c['path'] not in calib_excluded]
        if calib_excluded:
            print(f"  [calibration] {len(calib_excluded)} cas exclus du sweep "
                  f"(utilisés pour recalibrer res_scale)")
        cases = test_cases[:n_eval] if n_eval > 0 else test_cases
        print(f"\n── Sweep test set ({len(cases)} cas) ──"
              f"───────────────────────────────")
        if not cases:
            # Rien à sweeper : on saute plutôt que de planter sur all_data[0].
            print("  (0 cas, sweep sauté)")
        else:
            mesh_hr = np.load(ts.layout.mesh_path(ts.hr_res), allow_pickle=True).item()
            mu_gt = ts.stats['mu'].astype(np.float64)
            gt_cache = load_or_build_gt_cache(ts.layout, cases, ts.wc,
                                              mesh=mesh_hr, mu=mu_gt)
            sweep_results = _run_sweep(models, ts, knn_map, cases, gt_cache, batch_size,
                                       idw_knn, mesh_hr=mesh_hr)
            sweep_results['_ts_label'] = ts_label
            _plot_sweep(sweep_results, cases, ts, out)
            _save_summary_csv(ref_results, sweep_results, out / 'summary.csv')

    print(f"\nDone — {out}/")
    return {'ref': ref_results, 'sweep': sweep_results,
            'triang': getattr(ts, 'triang_hr', None), 'tag': ts.tag,
            'geometry': ts.geometry}


def _run_ref_cases(models: list[ModelEntry], ts: TestSet, knn_map: dict[str, dict], idw_knn: dict) -> dict:
    """Inférence + métriques sur les cas de référence."""
    wc = ts.wc

    dxy_edges = np.linalg.norm(
        wc.bary[wc.cell_adj_edges[:, 0]] - wc.bary[wc.cell_adj_edges[:, 1]], axis=1
    ).astype(np.float32)

    # Pré-chargement des cas
    case_data = []
    for c in ts.ref_cases:
        d = load_sample(c['path'], c.get('raw_hr_path'))
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
            _coord_norm = (entry.cfg or {}).get('architecture', {}).get('coord_norm', 'object')
            _is_wall = hasattr(entry.model, 'wall_encoder')
            _hf, _lf, *_ = build_features(_d0, _c0['mach_in'], _c0['aoa_in'],
                                           {'mu': _mu, 'sig': _sig}, _gid,
                                           _coord_norm, ts.hr_mesh_meta, is_wall=_is_wall,
                                           mach_norm=_resolve_mach_norm(entry.cfg))
            print(f"  [{entry.name}] warmup JIT ({ts.tag})...")
            jax.block_until_ready(entry.model.predict(_hf, _lf, knn_map[entry.name]))

    def _field_errs(prim, cd):
        if cd['hr_prim'] is None:
            return None, None, None
        er = compute_field_errors(prim, cd['hr_prim'],
                                  wc.cell_adj_edges, wc.bary, dxy_edges)
        return er['linf_mach'], er['l2w_mach'], er['w2_mach']

    def _aero(prim, cd):
        if cd['hr_prim'] is None:
            return None, None
        am = aero_metrics(prim, cd['hr_prim'], wc, cd['meta']['mach_in'])
        return am['wall_mach_pred'], am

    # Baseline "sans réseau" : IDW volumique (LR complet) ou baseline bord
    # (idw_knn['mode']=='wall'), selon le testset -- cf. TestSet.build_idw_knn.
    is_wall_bl = idw_knn is not None and idw_knn.get('mode') == 'wall'
    # Nom de la baseline dépendant du mode : 'LR IDW' (volumique) et 'Extrap. Bord'
    # (wall) sont de nature différente, cf. eval.core.idw_baseline_name.
    # utils/viz/eval.py et combined.py matchent sur IDW_BASELINE_NAMES.
    idw_row = {'name': idw_baseline_name(is_wall_bl), 'prim_preds': [], 'mach_preds': [],
                'l2': [], 'linf': [], 'l2w': [], 'w2': [], 'time_ms': [],
                'wall_mach': [], 'aero': []}
    if idw_knn is not None and (is_wall_bl or 'idx' in idw_knn):
        if not is_wall_bl:
            idx_idw = np.asarray(idw_knn['idx'])[:, :6].astype(np.int32)
            w_idw = idw_weights(np.asarray(idw_knn['dist'])[:, :6]).astype(np.float32)
        for cd in case_data:
            c = cd['meta']
            if is_wall_bl:
                prim, t = predict_wall_baseline(cd['d'], c['mach_in'], c['aoa_in'], idw_knn['wd_exp'])
            else:
                lr_p = cd['d']['lr_primitives'].astype(np.float32)
                t0 = time.perf_counter()
                prim = (w_idw[:, :, None] * lr_p[idx_idw]).sum(axis=1)
                t = (time.perf_counter() - t0) * 1e3
            mach = to_mach(prim)
            li, lw, w2v = _field_errs(prim, cd)
            wm, am = _aero(prim, cd)
            idw_row['prim_preds'].append(prim)
            idw_row['mach_preds'].append(mach)
            idw_row['l2'].append(l2_rel(mach, cd['mach_ref'])
                                 if cd['mach_ref'] is not None else None)
            idw_row['linf'].append(li); idw_row['l2w'].append(lw); idw_row['w2'].append(w2v)
            idw_row['time_ms'].append(t)
            idw_row['wall_mach'].append(wm); idw_row['aero'].append(am)

    # Modèles
    model_rows = []
    for entry in models:
        gid = ts.geom_id_for(entry)
        _mu_e, _sig_e = _train_stats(entry, ts.stats)
        _stats_e = {'mu': _mu_e, 'sig': _sig_e}
        # Modele stochastique (interpolant SDE) : ensemble -> moyenne + incertitude
        is_sde = getattr(entry.model, 'sampler', 'ode') == 'sde'
        row = {'name': ts.display_name(entry), 'prim_preds': [], 'mach_preds': [],
               'l2': [], 'linf': [], 'l2w': [], 'w2': [], 'time_ms': [],
               'wall_mach': [], 'aero': [], 'prim_std': []}
        for cd in case_data:
            c = cd['meta']
            if is_sde:
                prim, prim_std, t = predict_ensemble(
                    entry, cd['d'], c['mach_in'], c['aoa_in'],
                    _stats_e, knn=knn_map[entry.name], geom_id=gid,
                    mesh_meta=ts.hr_mesh_meta)
            else:
                prim, t = predict_det(entry, cd['d'], c['mach_in'], c['aoa_in'],
                                       _stats_e, knn=knn_map[entry.name], geom_id=gid,
                                       mesh_meta=ts.hr_mesh_meta)
                prim_std = None
            mach = to_mach(prim)
            li, lw, w2v = _field_errs(prim, cd)
            wm, am = _aero(prim, cd)
            row['prim_preds'].append(prim)
            row['prim_std'].append(prim_std)
            row['mach_preds'].append(mach)
            row['l2'].append(l2_rel(mach, cd['mach_ref'])
                             if cd['mach_ref'] is not None else None)
            row['linf'].append(li); row['l2w'].append(lw); row['w2'].append(w2v)
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
        'idw': idw_row if (idw_knn is not None and (is_wall_bl or 'idx' in idw_knn)) else None,
    }


def _run_sweep(models: list[ModelEntry], ts: TestSet,
               knn_map: dict[str, dict], cases: list[dict],
               gt_cache: dict | None, batch_size: int,
               idw_knn: dict | None = None, mesh_hr=None) -> dict:
    """Inférence batch + métriques sur le test set complet."""
    wc = ts.wc
    mu = ts.stats['mu'].astype(np.float32)
    sig = ts.stats['sig'].astype(np.float32)
    n = len(cases)

    det_models = [m for m in models if m.kind == 'det']
    is_wall_bl = idw_knn is not None and idw_knn.get('mode') == 'wall'
    _bl_name = idw_baseline_name(is_wall_bl)  # cf. remarque _run_ref_cases
    method_names = ([_bl_name] if idw_knn is not None else []) +                   [ts.display_name(m) for m in models]

    results: dict[str, dict] = {
        nm: {'times': [], 'CL': [], 'CD': [], 'grad_p_max': [],
             'mach_max': [], 'mach_min': [], 'l2_mach': [], 'linf_mach': [],
             'l2w_mach': [], 'w2_mach': [], 'l2_prims': [], 'linf_prims': [],
             'wall_mach': [], 'wall_cp': [], 'has_ref': [], 'enthalpy': [], 'entropy': [],
             'euler_fvm': [], 'euler_step': []}
        for nm in method_names
    }
    results['HR'] = {'CL': [], 'CD': [], 'grad_p_max': [],
                     'mach_max': [], 'mach_min': [], 'wall_mach': [], 'wall_cp': [],
                     'enthalpy_hr': [], 'entropy_hr': [], 'euler_fvm_hr': [],
                     'euler_step_hr': []}

    # Mesh HR pour le résidu Euler FVM exact (opérateur du solveur). Fourni par
    # evaluate() ; sans lui le résidu PDE n'est pas calculable (NaN).
    _mesh_obj = mesh_hr
    _has_mesh = mesh_hr is not None

    print(f"  Chargement parallèle de {n} fichiers...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        all_data: list[dict] = list(ex.map(lambda c: load_sample(c['path'], c.get('raw_hr_path')), cases))

    print("  Pré-calcul des features...")
    # 'domain' explicite (pas le defaut 'object') : ce geom partage sert a la baseline
    # IDW et aux modeles coord_norm='domain'. 'object' echouerait ici faute de
    # mesh_meta, et les modeles 'object' recalculent de toute facon leur propre geom_e.
    geom = mesh_geom_from_case(all_data[0], 'domain')
    # lr_feats avec ts.stats (pour IDW et modèles in-distrib), les modèles
    # OOD géométrique reconstruisent leurs propres features avec leurs training stats.
    lr_feats = [build_lr_feat(geom, d, mu, sig) for d in all_data]

    dxy_edges = np.linalg.norm(
        wc.bary[wc.cell_adj_edges[:, 0]] - wc.bary[wc.cell_adj_edges[:, 1]], axis=1
    ).astype(np.float32)

    # Baseline "sans réseau" (IDW volumique ou baseline bord, cf. plus haut)
    idw_preds: list[np.ndarray] = []
    idw_t_ms_per_case: float = 0.0
    if idw_knn is not None and is_wall_bl:
        t0 = time.perf_counter()
        for d, c in zip(all_data, cases):
            prim, _ = predict_wall_baseline(d, c['mach_in'], c['aoa_in'], idw_knn['wd_exp'])
            idw_preds.append(prim)
        idw_t_ms_per_case = (time.perf_counter() - t0) * 1e3 / max(n, 1)
    elif idw_knn is not None:
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
        coord_norm_e = (entry.cfg or {}).get('architecture', {}).get('coord_norm', 'object')
        mach_norm_e = _resolve_mach_norm(entry.cfg)
        geom_e = geom if coord_norm_e == 'domain' else mesh_geom_from_case(
            all_data[0], coord_norm_e, ts.hr_mesh_meta)

        # Temps de construction des features
        is_wall_e = hasattr(entry.model, 'wall_encoder')
        t0_feat = time.perf_counter()
        if is_wall_e:
            # FAMWall/DAMWall : observations de bord (fusionnées dans d par
            # utils.layout.load_sample depuis le store _wall compagnon), pas de
            # champ LR volumique -- jamais le chemin lr_feats partagé ci-dessus.
            lrf = [build_wall_feat(d, mu_e, sig_e, ts.hr_mesh_meta) for d in all_data]
            feat_ms_per_case = (time.perf_counter() - t0_feat) * 1e3 / n
        elif coord_norm_e == 'domain' and np.allclose(mu_e, mu) and np.allclose(sig_e, sig):
            lrf = lr_feats
            t0_lr = time.perf_counter()
            _ = build_lr_feat(geom_e, all_data[0], mu_e, sig_e)
            feat_ms_per_case = (time.perf_counter() - t0_lr) * 1e3
        else:
            lrf = [build_lr_feat(geom_e, d, mu_e, sig_e) for d in all_data]
            feat_ms_per_case = (time.perf_counter() - t0_feat) * 1e3 / n
        t0_hr = time.perf_counter()
        hr_feats = [build_hr_feat(geom_e, c['mach_in'], c['aoa_in'], gid, mach_norm=mach_norm_e)
                   for c in cases]
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
        r['wall_cp'].append(ac['wall_cp'])
        r['enthalpy'].append(enthalpy_rms(prim, mach_in))
        r['entropy'].append(entropy_violation(prim))
        r['euler_fvm'].append(fvm_euler_rms(prim, _mesh_obj, mach_in, aoa_in, mu)
                              if _has_mesh else np.nan)
        r['euler_step'].append(fvm_euler_step_rms(prim, _mesh_obj, mach_in, aoa_in)
                               if _has_mesh else np.nan)
        if has_ref:
            er = compute_field_errors(prim, hr_prim, wc.cell_adj_edges, wc.bary, dxy_edges)
            r['l2_mach'].append(er['l2_mach'])
            r['linf_mach'].append(er['linf_mach'])
            r['l2w_mach'].append(er['l2w_mach'])
            r['w2_mach'].append(er['w2_mach'])
            r['l2_prims'].append([er[f'l2_{p2}'] for p2 in _PRIM_NAMES])
            r['linf_prims'].append([er[f'linf_{p2}'] for p2 in _PRIM_NAMES])
        else:
            for k2 in ('l2_mach', 'linf_mach', 'l2w_mach', 'w2_mach'):
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
                if 'wall_cp' in gc:
                    gt['wall_cp'].append(gc['wall_cp'][i])
                else:
                    # cache GT généré avant l'ajout de wall_cp : repli ponctuel
                    # sur un calcul direct (à régénérer via utils/gt_cache.py).
                    gt['wall_cp'].append(aero_coeffs(hr_prim, wc, c['mach_in'])['wall_cp'])
                gt['enthalpy_hr'].append(float(gc.get('enthalpy_hr', [np.nan] * n)[i]))
                gt['entropy_hr'].append(float(gc.get('entropy_hr', [np.nan] * n)[i]))
            else:
                ac = aero_coeffs(hr_prim, wc, c['mach_in'])
                for k2 in ('CL', 'CD', 'grad_p_max', 'mach_max', 'mach_min'):
                    gt[k2].append(ac[k2])
                gt['wall_mach'].append(ac['wall_mach'])
                gt['wall_cp'].append(ac['wall_cp'])
                gt['enthalpy_hr'].append(enthalpy_rms(hr_prim, c['mach_in']))
                gt['entropy_hr'].append(entropy_violation(hr_prim))
            # Résidu Euler FVM du HR : priorité au cache (fvm_residual_hr), sinon live
            fr = gc.get('fvm_residual_hr') if gc is not None else None
            if fr is not None and np.isfinite(fr[i]):
                gt['euler_fvm_hr'].append(float(fr[i]))
            elif _has_mesh:
                gt['euler_fvm_hr'].append(
                    fvm_euler_rms(hr_prim, _mesh_obj, c['mach_in'], aoa_in, mu))
            else:
                gt['euler_fvm_hr'].append(np.nan)
            # Résidu de stationnarité (incrément relatif d'un pas) : toujours live.
            gt['euler_step_hr'].append(
                fvm_euler_step_rms(hr_prim, _mesh_obj, c['mach_in'], aoa_in)
                if _has_mesh else np.nan)
        else:
            for k2 in ('CL', 'CD', 'grad_p_max', 'mach_max', 'mach_min'):
                gt[k2].append(None)
            gt['wall_mach'].append(np.array([], dtype=np.float32))
            gt['wall_cp'].append(np.array([], dtype=np.float32))
            gt['enthalpy_hr'].append(np.nan)
            gt['entropy_hr'].append(np.nan)
            gt['euler_fvm_hr'].append(np.nan)
            gt['euler_step_hr'].append(np.nan)

        if idw_knn is not None:
            _fill(_bl_name, idw_preds[i], idw_t_ms_per_case,
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
            if isinstance(r[k2], list) and k2 not in ('l2_prims', 'linf_prims', 'wall_mach', 'wall_cp'):
                r[k2] = np.array([v if v is not None else np.nan for v in r[k2]], dtype=float)

    # Erreurs aéro vs GT
    gt_CL = results['HR']['CL']
    gt_CD = results['HR']['CD']
    gt_wall_mach = results['HR']['wall_mach']
    gt_wall_cp = results['HR']['wall_cp']
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
        wall_cp_l2 = []
        for cp_p, cp_r in zip(r['wall_cp'], gt_wall_cp):
            if len(cp_p) > 0 and len(cp_r) > 0:
                wall_cp_l2.append(float(
                    np.linalg.norm(cp_p - cp_r) / (np.linalg.norm(cp_r) + 1e-8)))
            else:
                wall_cp_l2.append(np.nan)
        r['wall_cp_l2'] = np.array(wall_cp_l2, dtype=float)

    results['_cases'] = cases
    results['_name_map'] = {_bl_name: _bl_name}
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
        plot_global_errors, plot_distributions,
        plot_cl_cd_distributions, plot_error_kde,
        plot_error_heatmap, plot_regime_heatmap, plot_summary_table,
    )
    print("\n── Plots : sweep ───────────────────────────────────────────")
    plot_global_errors(results, out)
    plot_distributions(results, out)
    plot_cl_cd_distributions(results, out)
    plot_error_kde(results, out)
    plot_error_heatmap(results, cases, out)
    plot_regime_heatmap(results, cases, out)
    plot_summary_table({}, results, out, hr_res=ts.hr_res, lr_res=ts.lr_res, geometry=ts.geometry)

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
        _cols.append(('Résidu Δt (rel)', 'euler_step', 'euler_step_hr'))
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

    _LIST_KEYS = {'wall_mach', 'wall_cp', 'l2_prims', 'linf_prims'}

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

    merged['_name_map'] = {base: base for base in base_names if base in merged}

    return merged


def _group_by_geometry(ts_results: dict[str, dict]) -> dict[str, dict]:
    """Regroupe les sweeps de plusieurs TestSets par géométrie (résolutions LR fusionnées).

    Ex. 'diamond_lr0.1' + 'diamond_lr0.2' -> une seule entrée 'diamond' dont les
    métriques sont la concaténation des deux résolutions (via _merge_sweep_results).
    """
    from collections import defaultdict
    from eval.testset import _geom_label

    by_geom: dict[str, list[dict]] = defaultdict(list)
    for tag, v in ts_results.items():
        sweep = v.get('sweep')
        if not sweep:
            continue
        geom = v.get('geometry', tag)
        by_geom[geom].append(sweep)

    out: dict[str, dict] = {}
    for geom, sweeps in by_geom.items():
        merged = _merge_sweep_results(sweeps)
        merged['_ts_label'] = _geom_label(geom)
        out[geom] = {'sweep': merged}
    return out


def plot_combined(ts_results: dict[str, dict], out_dir: Path) -> None:
    """Figures globales agrégées sur tous les TestSets évalués.

    Appelé automatiquement quand il y a plusieurs TestSets avec un sweep complet.
    Contrairement à l'ancienne approche (pool fusionné), la sortie garde la
    ventilation *par testset* : tableau global modèle×test, barres d'erreur par
    test, distributions facettées, et panel de champs (1 cas ref / testset).

    Produit en plus une ventilation *par géométrie seule* (résolutions LR
    fusionnées) dans combined/by_geometry/ : plus lisible quand plusieurs
    résolutions par géométrie diluent la comparaison en colonnes séparées.
    """
    from utils.viz.combined import plot_all_combined, plot_by_geometry

    tags = [tag for tag, v in ts_results.items() if v.get('sweep') is not None]
    if len(tags) < 2:
        return

    print(f"\n── Figures combinées ({' + '.join(tags)}) "
          f"{'─' * max(0, 46 - len(' + '.join(tags)))}")

    out = Path(out_dir) / 'combined'
    out.mkdir(exist_ok=True)

    plot_all_combined(ts_results, out)

    by_geom = _group_by_geometry(ts_results)
    if by_geom:
        out_geom = out / 'by_geometry'
        out_geom.mkdir(exist_ok=True)
        plot_by_geometry(by_geom, out_geom)
        print(f"  -> {out_geom}/")

    print(f"  -> {out}/")

def _save_summary_csv(ref_results: dict | None, sweep_results: dict | None,
                      path: Path):
    from utils.viz.eval import save_summary_csv
    save_summary_csv(ref_results or {}, sweep_results, path)
