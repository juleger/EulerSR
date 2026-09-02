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
                           check_every: int | None = None) -> list[dict]:
    """Lance eval.convergence_case.run_convergence_case sur chaque cas de 'cases' """
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
            results = run_single_case(models, ts, case, d, idw_knn, knn_map, n_repeat=1)
            summaries = run_convergence_case(
                results, ts.layout, case['mach_in'], case['aoa_in'], res_h,
                tf=tf, check_every=check_every)
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
    names = list(dict.fromkeys(r['name'] for r in rows))  # ordre d'apparition, dédupliqué
    header = (f"{'Champ initial':<28}{'n cas':>8}{'converged':>12}"
              f"{'itér. moy.':>14}{'itér. médiane':>16}{'accél. moy.':>14}")
    print('\n' + header)
    print('-' * len(header))
    for name in names:
        sub = [r for r in rows if r['name'] == name]
        steps = [r['stopping_step'] for r in sub if r.get('stopping_step') is not None]
        speedups = [r['speedup_vs_coldstart'] for r in sub if r.get('speedup_vs_coldstart') is not None]
        n_conv = sum(1 for r in sub if r.get('converged'))
        print(
            f"{name:<28}{len(sub):>8}{f'{n_conv}/{len(sub)}':>12}"
            f"{_fmt(np.mean(steps) if steps else None, '.1f'):>14}"
            f"{_fmt(np.median(steps) if steps else None, '.1f'):>16}"
            f"{_fmt(np.mean(speedups) if speedups else None, '.2f'):>14}"
        )


def save_sweep_csv(rows: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['case_label', 'mach', 'aoa', 'name', 'stopping_step', 'converged',
              'stationarity_rel', 'threshold_used', 'speedup_vs_coldstart',
              'wall_time_s', 'h', 'n_cells']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _build_grid(rows: list[dict], name: str, value_key: str):
    sub = [r for r in rows if r['name'] == name and r.get(value_key) is not None]
    machs = sorted({r['mach'] for r in sub})
    aoas = sorted({r['aoa'] for r in sub})
    grid = np.full((len(aoas), len(machs)), np.nan)
    for r in sub:
        i, j = aoas.index(r['aoa']), machs.index(r['mach'])
        grid[i, j] = r[value_key]
    return np.array(machs), np.array(aoas), grid


def plot_convergence_heatmaps(rows: list[dict], out_dir: Path,
                               value_key: str = 'stopping_step') -> Path | None:
    """Heatmap Mach/AoA du nb d'itérations pour chaque champ initial (name) de rows, sauvegardée dans out_dir."""
    names = list(dict.fromkeys(r['name'] for r in rows))
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
