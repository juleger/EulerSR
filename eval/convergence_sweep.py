"""Campagne de mesure de convergence FVM en warm-start (cf. eval.convergence_case
pour la mesure sur un cas unique) sur un sous-ensemble du testset : moyenne du
nb d'itérations par modèle (IDW/FAM/DAM/...) et cartographie Mach/AoA des zones
les plus difficiles à reconstruire au sens dynamique.

Coûteux (un run FVM complet par champ initial ET par cas), donc : sous-
échantillonnage par stride Mach/AoA, garde-fou --max_cases, et chaque cas qui
échoue est loggé et sauté plutôt que de faire échouer toute la campagne.
"""
from __future__ import annotations
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from eval.single_case import run_single_case
from eval.convergence_case import run_convergence_case
from eval.live_fvm import _DEFAULT_PATIENCE, _DEFAULT_ENGINEERING_TOL
from utils.layout import load_sample


def select_sweep_cases(test_cases: list[dict], stride_mach: int = 4, stride_aoa: int = 4,
                        max_cases: int | None = 60) -> list[dict]:
    """Sous-échantillonne test_cases sur une grille Mach/AoA plus grossière (stride_mach, stride_aoa)"""
    machs = sorted({c['mach_in'] for c in test_cases})
    aoas = sorted({c['aoa_in'] for c in test_cases})
    keep_mach = set(machs[::max(1, stride_mach)])
    keep_aoa = set(aoas[::max(1, stride_aoa)])
    selected = [c for c in test_cases if c['mach_in'] in keep_mach and c['aoa_in'] in keep_aoa]
    if max_cases is not None and len(selected) > max_cases:
        raise ValueError(
            f"{len(selected)} cas sélectionnés (stride_mach={stride_mach}, "
            f"stride_aoa={stride_aoa}) > max_cases={max_cases}. Augmente les "
            f"strides, ou passe --max_cases plus grand / --max_cases 0 pour "
            f"désactiver le garde-fou (campagne longue, prévois un job slurm).")
    return selected


def run_convergence_sweep(ts, models, idw_knn, knn_map, cases: list[dict], res_h: float,
                           tf: float | None = None,
                           check_every: int | None = None,
                           patience: int = _DEFAULT_PATIENCE,
                           cd_tol: float = _DEFAULT_ENGINEERING_TOL,
                           cl_tol: float = _DEFAULT_ENGINEERING_TOL,
                           include_idw: bool = True,
                           lr_solve_time_s: float | None = None) -> list[dict]:
    """Lance eval.convergence_case.run_convergence_case sur chaque cas de 'cases'."""
    from eval.live_fvm import _DEFAULT_CHECK_EVERY
    check_every = _DEFAULT_CHECK_EVERY if check_every is None else check_every

    all_rows = []
    n = len(cases)
    for i, case in enumerate(cases):
        label = case.get('label', f"M{case['mach_in']:.2f}_A{case['aoa_in']:+.1f}")
        print(f"\n=== Cas {i + 1}/{n} : {label} (M={case['mach_in']:.2f}, "
              f"AoA={case['aoa_in']:+.1f}°) ===")
        try:
            d = load_sample(case['path'], case.get('raw_hr_path'))
            results = run_single_case(models, ts, case, d, idw_knn if include_idw else None, knn_map,
                                       n_repeat=1)
            summaries = run_convergence_case(
                results, ts.layout, case['mach_in'], case['aoa_in'], res_h,
                tf=tf, check_every=check_every, patience=patience, cd_tol=cd_tol, cl_tol=cl_tol,
                include_idw=include_idw, lr_solve_time_s=lr_solve_time_s)
        except Exception as e:
            print(f"  [ERREUR] cas {label} sauté : {e}")
            continue

        for s in summaries:
            s['case_label'] = label
        all_rows.extend(summaries)

    return all_rows


def _fmt(value, spec: str | None = None) -> str:
    if value is None:
        return 'N/A'
    return format(value, spec) if spec else str(value)


def print_sweep_summary(rows: list[dict]) -> None:
    """Deux blocs, comme print_convergence_table : le résidu puis l'ingénieur. Les
    deux ratios de speedup (itérations et temps de résolution) sont côte à côte dans
    le bloc résidu ; ils divergent surtout pour les warm-starts rapides."""
    names = list(dict.fromkeys(r['name'] for r in rows))  # ordre d'apparition, dédupliqué
    header = (f"{'Champ initial':<28}{'n cas':>8}{'converged':>12}"
              f"{'itér. moy.':>14}{'itér. médiane':>16}{'accél. itér.':>14}{'accél. temps':>14}"
              f"{'accél. pipeline':>17}")
    print('\n=== Critère résidu (arrêt réel du solveur) ===')
    print(header)
    print('-' * len(header))
    for name in names:
        sub = [r for r in rows if r['name'] == name]
        steps = [r['stopping_step'] for r in sub if r.get('stopping_step') is not None]
        speedups_it = [r['speedup_iterations'] for r in sub if r.get('speedup_iterations') is not None]
        speedups_t = [r['speedup_solve_time'] for r in sub if r.get('speedup_solve_time') is not None]
        speedups_p = [r['speedup_pipeline'] for r in sub if r.get('speedup_pipeline') is not None]
        n_conv = sum(1 for r in sub if r.get('converged'))
        print(
            f"{name:<28}{len(sub):>8}{f'{n_conv}/{len(sub)}':>12}"
            f"{_fmt(np.mean(steps) if steps else None, '.1f'):>14}"
            f"{_fmt(np.median(steps) if steps else None, '.1f'):>16}"
            f"{_fmt(np.mean(speedups_it) if speedups_it else None, '.2f'):>14}"
            f"{_fmt(np.mean(speedups_t) if speedups_t else None, '.2f'):>14}"
            f"{_fmt(np.mean(speedups_p) if speedups_p else None, '.2f'):>17}"
        )

    if any(r.get('engineering_target_cd') is not None for r in rows):
        header2 = (f"{'Champ initial':<28}{'n cas':>8}{'converged':>12}"
                   f"{'itér. moy.':>14}{'itér. médiane':>16}{'accél. moy.':>14}")
        print('\n=== Critère ingénieur (Cd/Cl vs cible HR) ===')
        print(header2)
        print('-' * len(header2))
        for name in names:
            sub = [r for r in rows if r['name'] == name]
            steps = [r['engineering_stopping_step'] for r in sub if r.get('engineering_stopping_step') is not None]
            speedups = [r['speedup_engineering'] for r in sub if r.get('speedup_engineering') is not None]
            n_conv = sum(1 for r in sub if r.get('engineering_converged'))
            print(
                f"{name:<28}{len(sub):>8}{f'{n_conv}/{len(sub)}':>12}"
                f"{_fmt(np.mean(steps) if steps else None, '.1f'):>14}"
                f"{_fmt(np.median(steps) if steps else None, '.1f'):>16}"
                f"{_fmt(np.mean(speedups) if speedups else None, '.2f'):>14}"
            )


def save_sweep_csv(rows: list[dict], path: Path) -> None:
    """Détail brut, une ligne par (cas, champ initial). Pour un rapport, plutôt
    save_sweep_summary_csv."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['case_label', 'mach', 'aoa', 'name', 'stopping_step', 'converged',
              'stationarity_rel', 'threshold_used', 'speedup_iterations', 'speedup_solve_time',
              'speedup_pipeline', 'engineering_stopping_step', 'engineering_converged',
              'speedup_engineering', 'engineering_target_cd', 'engineering_target_cl',
              'wall_time_s', 'compile_time_s', 'solve_time_s', 'infer_time_s',
              'total_pipeline_time_s', 'h', 'n_cells']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def save_sweep_summary_csv(rows: list[dict], path: Path) -> None:
    """Synthèse pour rapport : une ligne par champ initial, stats globales du sweep
    seulement. Les deux critères restent dans des colonnes préfixées distinctes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(dict.fromkeys(r['name'] for r in rows))

    def _stats(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return (None, None, None)
        return (float(np.mean(vals)), float(np.median(vals)), float(np.std(vals)))

    fields = ['name', 'n_cases',
              'residual_n_converged', 'residual_convergence_rate',
              'residual_iterations_mean', 'residual_iterations_median', 'residual_iterations_std',
              'residual_speedup_iterations_mean', 'residual_speedup_iterations_median',
              'residual_speedup_solve_time_mean', 'residual_speedup_solve_time_median',
              'residual_speedup_pipeline_mean', 'residual_speedup_pipeline_median',
              'pipeline_time_s_mean', 'pipeline_time_s_median',
              'engineering_n_converged', 'engineering_convergence_rate',
              'engineering_iterations_mean', 'engineering_iterations_median',
              'engineering_speedup_mean', 'engineering_speedup_median']

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name in names:
            sub = [r for r in rows if r['name'] == name]
            n = len(sub)
            it_mean, it_med, it_std = _stats([r.get('stopping_step') for r in sub])
            sp_it_mean, sp_it_med, _ = _stats([r.get('speedup_iterations') for r in sub])
            sp_t_mean, sp_t_med, _ = _stats([r.get('speedup_solve_time') for r in sub])
            sp_pipe_mean, sp_pipe_med, _ = _stats([r.get('speedup_pipeline') for r in sub])
            pipe_mean, pipe_med, _ = _stats([r.get('total_pipeline_time_s') for r in sub])
            eng_it_mean, eng_it_med, _ = _stats([r.get('engineering_stopping_step') for r in sub])
            sp_eng_mean, sp_eng_med, _ = _stats([r.get('speedup_engineering') for r in sub])
            n_conv = sum(1 for r in sub if r.get('converged'))
            n_eng_conv = sum(1 for r in sub if r.get('engineering_converged'))
            writer.writerow({
                'name': name, 'n_cases': n,
                'residual_n_converged': n_conv,
                'residual_convergence_rate': n_conv / n if n else None,
                'residual_iterations_mean': it_mean, 'residual_iterations_median': it_med,
                'residual_iterations_std': it_std,
                'residual_speedup_iterations_mean': sp_it_mean,
                'residual_speedup_iterations_median': sp_it_med,
                'residual_speedup_solve_time_mean': sp_t_mean,
                'residual_speedup_solve_time_median': sp_t_med,
                'residual_speedup_pipeline_mean': sp_pipe_mean,
                'residual_speedup_pipeline_median': sp_pipe_med,
                'pipeline_time_s_mean': pipe_mean, 'pipeline_time_s_median': pipe_med,
                'engineering_n_converged': n_eng_conv,
                'engineering_convergence_rate': n_eng_conv / n if n else None,
                'engineering_iterations_mean': eng_it_mean,
                'engineering_iterations_median': eng_it_med,
                'engineering_speedup_mean': sp_eng_mean,
                'engineering_speedup_median': sp_eng_med,
            })


def _build_grid(rows: list[dict], name: str, value_key: str):
    sub = [r for r in rows if r['name'] == name and r.get(value_key) is not None]
    machs = sorted({r['mach'] for r in sub})
    aoas = sorted({r['aoa'] for r in sub})
    grid = np.full((len(aoas), len(machs)), np.nan)
    for r in sub:
        i, j = aoas.index(r['aoa']), machs.index(r['mach'])
        grid[i, j] = r[value_key]
    return np.array(machs), np.array(aoas), grid


_TRIVIAL_SPEEDUP_LABELS = {'cold-start (freestream)', 'HR (référence)'}


def plot_convergence_heatmaps(rows: list[dict], out_dir: Path,
                               value_key: str = 'stopping_step') -> Path | None:
    """Heatmap Mach/AoA du nb d'itérations pour chaque champ initial (name) de rows, sauvegardée dans out_dir.

    Pour les value_key 'speedup_*', le cold-start et le HR de référence sont exclus :
    leur accélération vis-à-vis d'eux-mêmes est triviale et ne montre rien."""
    names = list(dict.fromkeys(r['name'] for r in rows))
    if value_key.startswith('speedup_'):
        names = [n for n in names if n not in _TRIVIAL_SPEEDUP_LABELS]
    grids = {name: _build_grid(rows, name, value_key) for name in names}
    grids = {k: v for k, v in grids.items() if v[0].size >= 2 and v[1].size >= 2}
    if not grids:
        print(f"  [WARN] pas assez de points pour une heatmap ({value_key}) -- "
              f"élargis le sweep (moins de stride) pour au moins une grille 2x2.")
        return None

    all_vals = np.concatenate([g[2][~np.isnan(g[2])] for g in grids.values()])
    all_vals = all_vals[all_vals > 0]
    vmin, vmax = (all_vals.min(), all_vals.max()) if all_vals.size else (1, 1)

    n = len(grids)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    from matplotlib.colors import LogNorm
    norm = LogNorm(vmin=max(vmin, 1e-12), vmax=max(vmax, vmin * 1.01))
    mesh = None
    for idx, (name, (machs, aoas, grid)) in enumerate(grids.items()):
        ax = axes[idx // ncols][idx % ncols]
        mesh = ax.pcolormesh(machs, aoas, grid, shading='nearest', cmap='viridis', norm=norm)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel('Mach')
        ax.set_ylabel('AoA (°)')
    for idx in range(len(grids), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis('off')

    if mesh is not None:
        fig.colorbar(mesh, ax=axes, shrink=0.8, label=value_key)
    fig.suptitle(f"Convergence FVM en warm-start — {value_key} (plan Mach/AoA)")

    out_path = Path(out_dir) / f'convergence_heatmap_{value_key}.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out_path
