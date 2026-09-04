"""utils/viz/eval.py — visualisations d'évaluation SR-CFD."""
from __future__ import annotations
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as _mgs
from matplotlib.colors import LogNorm, Normalize
from scipy.stats import gaussian_kde
from scipy.interpolate import griddata

from utils.viz._style import _DPI, CMAP_FIELD as _CMAP_FIELD, CMAP_ERR as _CMAP_ERR
from eval.core import IDW_BASELINE_NAMES

# Okabe-Ito — palette colorblind-safe, standard publications scientifiques
_METHOD_COLORS = [
    '#0072B2',  # bleu
    '#D55E00',  # vermillon
    '#009E73',  # vert
    '#E69F00',  # orange
    '#CC79A7',  # rose/mauve
    '#F0E442',  # jaune
    '#56B4E9',  # bleu ciel
]

_HR_COLOR = '#222222'
_HR_LW = 1.3
_HR_LS = '--'
_HR_ALPHA = 0.85


def _ts_suptitle(base: str, results: dict) -> str:
    """Préfixe le titre de base avec le label du test set (géométrie + OOD)."""
    label = results.get('_ts_label', '') if isinstance(results, dict) else ''
    return f"{label} — {base}" if label else base


def _display_name(nm: str) -> str:
    if nm in ('GT', 'HR'):
        return 'HR'
    if nm == 'IDW':
        return 'LR IDW'
    return nm


def _build_palette(method_names: list[str]) -> dict[str, str]:
    palette: dict[str, str] = {}
    ci = 0
    for nm in method_names:
        if nm in ('GT', 'HR'):
            palette[nm] = _HR_COLOR
        else:
            palette[nm] = _METHOD_COLORS[ci % len(_METHOD_COLORS)]
            ci += 1
    return palette


def _kde_or_hist(ax, vals: np.ndarray, color: str, label: str, lw: float = 1.4,
            ls: str = '-', fill_alpha: float = 0.06, zorder: int = 2, alpha: float = 1.0):
    """KDE avec fallback histogramme, rognage aux percentiles 1–99."""
    vals = vals[np.isfinite(vals)]
    if len(vals) < 10:
        return
    lo = float(np.percentile(vals, 1))
    hi = float(np.percentile(vals, 99))
    v = np.clip(vals, lo, hi)
    if len(v) > 60_000:
        v = np.random.default_rng(0).choice(v, 60_000, replace=False)
    try:
        kde = gaussian_kde(v, bw_method='scott')
        xs = np.linspace(lo, hi, 600)
        ax.plot(xs, kde(xs), ls, lw=lw, label=label, color=color,
                zorder=zorder, alpha=alpha)
        if fill_alpha > 0:
            ax.fill_between(xs, kde(xs), alpha=fill_alpha * alpha, color=color,
                            zorder=zorder - 1)
    except Exception:
        ax.hist(v, bins=80, density=True, histtype='step',
                lw=lw, label=label, color=color, linestyle=ls,
                zorder=zorder, alpha=alpha)


def plot_reference_grid(results: dict, triang, out_path: Path):
    """comparison_fields.png + comparison_errors.png."""
    out_path = Path(out_path)
    out_fields = out_path.parent / 'comparison_fields.png'
    out_errors = out_path.parent / 'comparison_errors.png'

    cases = results['cases']
    n_cases = len(cases)
    has_ref = [c['mach_ref'] is not None for c in cases]

    col_defs: list[dict] = []
    ref_mach = [c['mach_ref'] for c in cases]
    hr_col = ({'name': 'HR GT', 'mach_preds': ref_mach, 'l2': [None] * n_cases}
                if any(m is not None for m in ref_mach) else None)
    if results['idw']:
        col_defs.append(results['idw'])
    for row in results['rows']:
        col_defs.append(row)
    if hr_col is not None:
        col_defs.append(hr_col)  # HR toujours à droite comme référence

    n_cols = len(col_defs)
    if n_cols and n_cases:
        ar = max(float(triang.y.max() - triang.y.min()) /
                 max(float(triang.x.max() - triang.x.min()), 1e-6), 0.1)
        pw, cb_w = 2.3, 0.20
        ph = pw * ar
        fig = plt.figure(figsize=(n_cols * pw + cb_w, n_cases * ph), dpi=_DPI)
        gs = _mgs.GridSpec(n_cases, n_cols + 1, figure=fig,
                            width_ratios=[*([pw] * n_cols), cb_w],
                            hspace=0.01, wspace=0.01)
        axes = np.array([[fig.add_subplot(gs[ri, ci]) for ci in range(n_cols)]
                           for ri in range(n_cases)])
        cb_axs = [fig.add_subplot(gs[ri, n_cols]) for ri in range(n_cases)]
        fig.suptitle(_ts_suptitle('Champs Mach', results),
                     fontsize=11, fontweight='bold')

        for ci, cd in enumerate(col_defs):
            axes[0, ci].set_title(cd['name'], fontsize=8, fontweight='bold', pad=3)

        for ri, case in enumerate(cases):
            mach_ref = case['mach_ref']
            va = float(mach_ref.min()) if mach_ref is not None else None
            vb = float(mach_ref.max()) if mach_ref is not None else None
            sc_row = None

            for ci, cd in enumerate(col_defs):
                ax = axes[ri, ci]
                mach = cd['mach_preds'][ri]
                if mach is None:
                    ax.axis('off'); continue
                vmin = va if va is not None else float(mach.min())
                vmax = vb if vb is not None else float(mach.max())
                sc = ax.tripcolor(triang, facecolors=np.clip(mach, vmin, vmax),
                                  cmap=_CMAP_FIELD, vmin=vmin, vmax=vmax)
                ax.set_aspect('equal'); ax.axis('off')
                if sc_row is None:
                    sc_row = sc

            axes[ri, 0].text(-0.03, 0.5,
                             f"M={case['mach_in']:.2f}  AoA={case['aoa_in']:+.0f}°",
                             transform=axes[ri, 0].transAxes,
                             va='center', ha='center', fontsize=7.5, fontweight='bold',
                             rotation=90)

            if sc_row is not None:
                _p = cb_axs[ri].get_position()
                _h = _p.height * 0.62
                cb_axs[ri].set_position([_p.x0, _p.y0 + (_p.height - _h) / 2, _p.width, _h])
                fig.colorbar(sc_row, cax=cb_axs[ri], label='Mach')

        plt.savefig(out_fields, dpi=_DPI, bbox_inches='tight')
        plt.close(fig)
        print(f"  > {out_fields.name}")

    err_cols: list[dict] = []
    if any(has_ref):
        if results['idw']:
            err_cols.append({'name': results['idw']['name'], 'mach_preds': results['idw']['mach_preds'],
                             'w2': results['idw'].get('w2', [])})
        for row in results['rows']:
            err_cols.append({'name': row['name'], 'mach_preds': row['mach_preds'],
                             'w2': row.get('w2', [])})

    if not err_cols:
        return

    all_err_vals: list[float] = []
    for ri, case in enumerate(cases):
        mach_ref = case['mach_ref']
        if mach_ref is None:
            continue
        for cd in err_cols:
            mach = cd['mach_preds'][ri]
            if mach is not None:
                all_err_vals.extend(
                    (np.abs(mach - mach_ref) / (np.abs(mach_ref) + 1e-6)).tolist())
    err_vmax = float(np.percentile(all_err_vals, 95)) if all_err_vals else 0.1

    n_ecols = len(err_cols)
    ar = max(float(triang.y.max() - triang.y.min()) /
             max(float(triang.x.max() - triang.x.min()), 1e-6), 0.1)
    pw, ph = 2.3, 2.3 * ar
    fig = plt.figure(figsize=(n_ecols * pw + 0.25, n_cases * ph), dpi=_DPI)
    gs = _mgs.GridSpec(n_cases, n_ecols + 1, figure=fig,
                        width_ratios=[*([pw] * n_ecols), 0.20],
                        hspace=0.01, wspace=0.01)
    axes = np.array([[fig.add_subplot(gs[ri, ci]) for ci in range(n_ecols)]
                       for ri in range(n_cases)])
    cb_col = fig.add_subplot(gs[:, n_ecols])
    fig.suptitle(_ts_suptitle('Erreur relative Mach', results),
                 fontsize=11, fontweight='bold')

    for ci, cd in enumerate(err_cols):
        axes[0, ci].set_title(cd['name'], fontsize=8, fontweight='bold', pad=3)

    sc_ref = None
    for ri, case in enumerate(cases):
        mach_ref = case['mach_ref']
        axes[ri, 0].text(-0.03, 0.5,
                         f"M={case['mach_in']:.2f}  AoA={case['aoa_in']:+.0f}°",
                         transform=axes[ri, 0].transAxes,
                         va='center', ha='center', fontsize=7.5, fontweight='bold',
                         rotation=90)
        for ci, cd in enumerate(err_cols):
            ax = axes[ri, ci]
            mach = cd['mach_preds'][ri]
            if mach is None or mach_ref is None:
                ax.axis('off'); continue
            err = np.abs(mach - mach_ref) / (np.abs(mach_ref) + 1e-6)
            sc = ax.tripcolor(triang, facecolors=err,
                               cmap=_CMAP_ERR, vmin=0, vmax=err_vmax)
            ax.set_aspect('equal'); ax.axis('off')
            w2_v = (cd.get('w2', []) or []); w2_v = w2_v[ri] if ri < len(w2_v) else None
            if w2_v is not None:
                ax.text(0.5, 0.03, f'W2={w2_v:.4f}',
                        transform=ax.transAxes, ha='center', va='bottom',
                        fontsize=6.5, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.88,
                                  ec='#888888', lw=0.5))
            if sc_ref is None:
                sc_ref = sc

    if sc_ref is not None:
        _p = cb_col.get_position()
        _h = _p.height * 0.80
        cb_col.set_position([_p.x0, _p.y0 + (_p.height - _h) / 2, _p.width, _h])
        fig.colorbar(sc_ref, cax=cb_col, label='Erreur relative Mach')

    plt.savefig(out_errors, dpi=_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  > {out_errors.name}")


def _wall_profile_eval(vals_at_cells, wc, n_pts=150):
    """Profil scalaire lisse sur la face supérieure de la paroi (y > centre)."""
    from scipy.interpolate import interp1d as _interp1d
    bw = wc.bary[wc.unique_wall_cell_ids]
    v = np.asarray(vals_at_cells)[wc.unique_wall_cell_ids]
    x, y = bw[:, 0], bw[:, 1]
    y_mid = 0.5 * (float(y.min()) + float(y.max()))
    mask = y >= y_mid
    xi, vi = x[mask], v[mask]
    if len(xi) < 3:
        return xi, vi
    ord_ = np.argsort(xi)
    xi, vi = xi[ord_], vi[ord_]
    _, uniq = np.unique(np.round(xi * 1e5).astype(int), return_index=True)
    xi, vi = xi[uniq], vi[uniq]
    if len(xi) < 2:
        return xi, vi
    f = _interp1d(xi, vi, kind='linear', bounds_error=False,
                    fill_value=(vi[0], vi[-1]))
    x_r = np.linspace(xi.min(), xi.max(), n_pts)
    return x_r, f(x_r)


def plot_wall_profiles_eval(results: dict, wc, out_dir):
    from utils.aero import cp_field as _cp_field

    cases = results['cases']
    n_cases = len(cases)
    all_rows = []
    if results.get('idw') and results['idw'].get('prim_preds'):
        all_rows.append(results['idw'])
    for row in results.get('rows', []):
        if row.get('prim_preds'):
            all_rows.append(row)

    if not all_rows:
        print('  [SKIP] plot_wall_profiles_eval : pas de prim_preds dans results')
        return

    fig, axes = plt.subplots(n_cases, 1, figsize=(7, 3.5 * n_cases),
                             dpi=_DPI, constrained_layout=True)
    if n_cases == 1:
        axes = [axes]

    for ri, c in enumerate(cases):
        ax = axes[ri]
        mach_in = c['mach_in']
        hp = c.get('hr_prim')
        if hp is not None:
            xu_m, cpu = _wall_profile_eval(_cp_field(hp, mach_in), wc)
            ax.plot(xu_m, cpu, _HR_LS, lw=_HR_LW, color=_HR_COLOR, label='HR',
                    alpha=_HR_ALPHA, zorder=1)

        for ci, row in enumerate(all_rows):
            pp = row.get('prim_preds', [])
            if ri >= len(pp) or pp[ri] is None:
                continue
            xu_m, cpu = _wall_profile_eval(_cp_field(pp[ri], mach_in), wc)
            ax.plot(xu_m, cpu, '-', lw=1.8, zorder=2,
                    color=_METHOD_COLORS[ci % len(_METHOD_COLORS)],
                    label=_display_name(row['name']))

        ax.axhline(0.0, color='gray', lw=0.8, ls=':', alpha=0.6)
        ax.invert_yaxis()  # convention aéro : -Cp vers le haut
        ax.set_ylabel(r'$C_p$'); ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9, loc='best', framealpha=0.9)
        ax.set_title(f"M = {c['mach_in']:.2f}    AoA = {c['aoa_in']:+.0f}°",
                     fontsize=10, fontweight='bold')

    axes[-1].set_xlabel('x')
    fig.suptitle(_ts_suptitle('Coefficient de pression $C_p$ en paroi — face supérieure', results),
                 fontsize=13, fontweight='bold')
    out_path = Path(out_dir) / 'wall_profiles.png'
    plt.savefig(out_path, dpi=_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  > {out_path.name}')


def plot_cp_profile_eval(results: dict, wc, out_dir):
    """Profil de Cp en paroi (face supérieure) — pendant de plot_wall_profiles_eval."""
    from utils.aero import cp_field as _cp_field

    cases = results['cases']
    n_cases = len(cases)
    all_rows = []
    if results.get('idw') and results['idw'].get('prim_preds'):
        all_rows.append(results['idw'])
    for row in results.get('rows', []):
        if row.get('prim_preds'):
            all_rows.append(row)

    if not all_rows:
        print('  [SKIP] plot_cp_profile_eval : pas de prim_preds dans results')
        return

    fig, axes = plt.subplots(n_cases, 1, figsize=(7, 3.5 * n_cases),
                             dpi=_DPI, constrained_layout=True)
    if n_cases == 1:
        axes = [axes]

    for ri, c in enumerate(cases):
        ax = axes[ri]
        hp = c.get('hr_prim')
        if hp is not None:
            xu_c, cpu = _wall_profile_eval(_cp_field(hp, c['mach_in']), wc)
            ax.plot(xu_c, cpu, _HR_LS, lw=_HR_LW, color=_HR_COLOR, label='HR',
                    alpha=_HR_ALPHA, zorder=1)

        for ci, row in enumerate(all_rows):
            pp = row.get('prim_preds', [])
            if ri >= len(pp) or pp[ri] is None:
                continue
            xu_c, cpu = _wall_profile_eval(_cp_field(pp[ri], c['mach_in']), wc)
            ax.plot(xu_c, cpu, '-', lw=1.8, zorder=2,
                    color=_METHOD_COLORS[ci % len(_METHOD_COLORS)],
                    label=_display_name(row['name']))

        ax.invert_yaxis()  # convention aéro : Cp décroissant vers le haut
        ax.set_ylabel(r'$C_p$'); ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9, loc='best', framealpha=0.9)
        ax.set_title(f"M = {c['mach_in']:.2f}    AoA = {c['aoa_in']:+.0f}°",
                     fontsize=10, fontweight='bold')

    axes[-1].set_xlabel('x')
    fig.suptitle(_ts_suptitle(r'Coefficient de pression $C_p$ en paroi — face supérieure', results),
                 fontsize=13, fontweight='bold')
    out_path = Path(out_dir) / 'cp_profile.png'
    plt.savefig(out_path, dpi=_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  > {out_path.name}')


def plot_distributions(dataset_results: dict, out_dir: Path):
    method_names = [nm for nm in dataset_results if not nm.startswith('_')]
    palette = _build_palette(method_names)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=_DPI, constrained_layout=True)
    fig.suptitle(_ts_suptitle('Distributions physiques', dataset_results),
                 fontsize=12, fontweight='bold')

    for ax, key, xlabel, title in zip(
        axes,
        ['mach_max', 'grad_p_max'],
        [r'$M_{max}$', r'$\|\nabla p\|_{max}$'],
        ['Mach maximum (champ)', 'Gradient de pression max'],
    ):
        for nm in method_names:
            r = dataset_results.get(nm, {})
            vals = np.array(r.get(key, []), dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) < 5:
                continue
            is_hr = nm in ('GT', 'HR')
            _kde_or_hist(ax, vals, palette[nm], _display_name(nm),
                         lw=_HR_LW if is_hr else 1.4,
                         ls=_HR_LS if is_hr else '-',
                         fill_alpha=0.0 if is_hr else 0.10,
                         zorder=3 if is_hr else 2,
                         alpha=_HR_ALPHA if is_hr else 1.0)
        ax.set_xlabel(xlabel); ax.set_ylabel('Densité')
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.25, linestyle='--')

    out_path = out_dir / 'dist_mach_grad.png'
    plt.savefig(out_path, dpi=_DPI, bbox_inches='tight'); plt.close(fig)
    print(f"  > {out_path.name}")


def plot_cl_cd_distributions(dataset_results: dict, out_dir: Path):
    method_names = [nm for nm in dataset_results if not nm.startswith('_')]
    palette = _build_palette(method_names)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=_DPI, constrained_layout=True)
    fig.suptitle(_ts_suptitle(r'Distributions $C_L$ et $C_D$', dataset_results),
                 fontsize=13, fontweight='bold')

    for ax, key, xlabel, title in zip(
        axes,
        ['CL', 'CD'],
        [r'$C_L$', r'$C_D$'],
        [r'Portance $C_L$', r'Traînée $C_D$'],
    ):
        for nm in method_names:
            r = dataset_results.get(nm, {})
            vals = np.array(r.get(key, []), dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) < 5:
                continue
            is_hr = nm in ('GT', 'HR')
            _kde_or_hist(ax, vals, palette.get(nm, 'gray'), _display_name(nm),
                         lw=_HR_LW if is_hr else 1.4,
                         ls=_HR_LS if is_hr else '-',
                         fill_alpha=0.0 if is_hr else 0.10,
                         zorder=3 if is_hr else 2,
                         alpha=_HR_ALPHA if is_hr else 1.0)
        ax.set_xlabel(xlabel); ax.set_ylabel('Densité')
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(True, alpha=0.25, linestyle='--')

    out_path = out_dir / 'cl_cd_distributions.png'
    plt.savefig(out_path, dpi=_DPI, bbox_inches='tight'); plt.close(fig)
    print(f'  > {out_path.name}')


def plot_error_kde(dataset_results: dict, out_dir: Path):
    methods = [nm for nm in dataset_results
               if not nm.startswith('_') and nm not in ('GT', 'HR')
               and any(k in dataset_results.get(nm, {})
                       for k in ('l2_mach', 'l2w_mach', 'w2_mach'))]
    if not methods:
        return

    palette = _build_palette(methods)
    metrics = [('l2w_mach', r'$L_{2w}$ Mach'), ('w2_mach', r'$W_2$ Mach')]
    valid_metrics = [(k, lb) for k, lb in metrics
                     if any(len(dataset_results.get(nm, {}).get(k, [])) > 5
                            for nm in methods)]
    if not valid_metrics:
        return

    n_m = len(valid_metrics)
    fig, axes = plt.subplots(1, n_m, figsize=(5.5 * n_m, 4.5),
                             dpi=_DPI, constrained_layout=True)
    if n_m == 1:
        axes = [axes]
    fig.suptitle(_ts_suptitle('Distribution des erreurs par cas', dataset_results),
                 fontsize=11, fontweight='bold')

    for ax, (key, label) in zip(axes, valid_metrics):
        for nm in methods:
            vals = np.array(dataset_results[nm].get(key, []), dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) < 5:
                continue
            _kde_or_hist(ax, vals, palette[nm], nm, lw=1.8, ls='-',
                         fill_alpha=0.10, zorder=2)
        ax.set_xlabel(label); ax.set_ylabel('Densité')
        ax.set_title(f'Distribution — {label}', fontweight='bold')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.25, linestyle='--')
        ax.set_xlim(left=0)

    out_path = Path(out_dir) / 'error_kde.png'
    plt.savefig(out_path, dpi=_DPI, bbox_inches='tight'); plt.close(fig)
    print(f"  > {out_path.name}")


def _heatmap_metric(dataset_results: dict, all_cases: list[dict], out_dir: Path,
                    metric_key: str, metric_label: str, fname: str):
    methods = [nm for nm in dataset_results
               if not nm.startswith('_') and nm not in ({'GT', 'HR'} | IDW_BASELINE_NAMES)
               and metric_key in dataset_results.get(nm, {})]
    if not methods:
        return

    machs = np.array([c['mach_in'] for c in all_cases])
    aoas = np.array([c['aoa_in'] for c in all_cases])
    # La triangulation de Delaunay exige au moins 4 points non alignes dans le plan
    # (Mach, AoA) : sur un sweep tronque, les cas partagent souvent le meme AoA et
    # font planter tout le run sur un QhullError.
    if len(np.unique(machs)) < 2 or len(np.unique(aoas)) < 2 or len(machs) < 4:
        print(f"  [saut] heatmap {metric_label} : "
              f"{len(machs)} cas sur {len(np.unique(machs))} Mach x "
              f"{len(np.unique(aoas))} AoA, triangulation impossible")
        return
    m_grid = np.linspace(machs.min(), machs.max(), 60)
    a_grid = np.linspace(aoas.min(), aoas.max(), 40)
    MM, AA = np.meshgrid(m_grid, a_grid)

    n_m = len(methods)
    fig, axes = plt.subplots(1, n_m, figsize=(4.5 * n_m, 4.0),
                             dpi=_DPI, constrained_layout=True)
    if n_m == 1:
        axes = [axes]
    fig.suptitle(_ts_suptitle(f'Heatmap {metric_label} (Mach, AoA)', dataset_results),
                 fontsize=11, fontweight='bold')

    all_vals = [dataset_results[nm][metric_key] for nm in methods]
    vmax = float(np.nanpercentile(np.concatenate(all_vals), 95))

    last_pc = None
    for ax, nm in zip(axes, methods):
        vals = np.array(dataset_results[nm][metric_key], dtype=float)
        valid = np.isfinite(vals)
        Z = griddata((machs[valid], aoas[valid]), vals[valid],
                     (MM, AA), method='linear')
        last_pc = ax.pcolormesh(MM, AA, Z, cmap='YlOrRd', vmin=0, vmax=vmax, shading='auto')
        ax.scatter(machs[valid], aoas[valid], c='k', s=4, alpha=0.3, linewidths=0)
        ax.set_xlabel('Mach')
        ax.set_ylabel('AoA (°)')
        ax.set_title(nm, fontweight='bold')

    if last_pc is not None:
        fig.colorbar(last_pc, ax=axes, label=metric_label, location='bottom',
                     shrink=0.6, pad=0.12)

    out_path = Path(out_dir) / fname
    plt.savefig(out_path, dpi=_DPI, bbox_inches='tight'); plt.close(fig)
    print(f"  > {out_path.name}")


def plot_error_heatmap(dataset_results: dict, all_cases: list[dict], out_dir: Path):
    _heatmap_metric(dataset_results, all_cases, out_dir,
                    'l2w_mach', r'$L_{2w}$ Mach', 'error_heatmap_l2w.png')
    _heatmap_metric(dataset_results, all_cases, out_dir,
                    'w2_mach', r'$W_2$ Mach', 'error_heatmap_w2.png')


# 4 métriques les plus lues (champ, aéro, paroi) pour plot_regime_heatmap. Cp paroi
# plutôt que Mach paroi, cf. has_cp dans _summary_methods_rows.
_REGIME_METRICS = [
    ('w2_mach', r'$W_2$'),
    ('CL_err', r'$\Delta C_L$'),
    ('CD_err', r'$\Delta C_D$'),
    ('wall_cp_l2', r'$C_p$ paroi L2'),
]


def plot_regime_heatmap(dataset_results: dict, all_cases: list[dict], out_dir: Path,
                        n_bins: int = 6):
    """error_heatmap_{l2w,w2}.png croisent Mach × AoA pour une seule métrique. Ici
    l'inverse : on agrège l'AoA par régime de Mach pour caser 4 métriques × toutes
    les méthodes sur une image, et répondre à « quel régime est le plus dur, et pour
    quelle métrique ? ».

    Bornes de régime par quantiles plutôt qu'à largeur fixe, les cas d'un sweep
    n'étant pas échantillonnés uniformément en Mach. Échelle de couleur normalisée
    par ligne : les 4 métriques n'ont ni la même unité ni le même ordre de grandeur."""
    methods = [nm for nm in dataset_results
               if not nm.startswith('_') and nm not in ({'GT', 'HR'} | IDW_BASELINE_NAMES)
               and any(k in dataset_results.get(nm, {}) for k, _ in _REGIME_METRICS)]
    if not methods:
        return

    machs = np.array([c['mach_in'] for c in all_cases], dtype=float)
    if machs.size == 0 or not np.isfinite(machs).any():
        return
    edges = np.unique(np.nanquantile(machs, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return  # pas assez de valeurs de Mach distinctes pour binner utilement
    nb = len(edges) - 1
    bin_idx = np.clip(np.digitize(machs, edges[1:-1], right=True), 0, nb - 1)
    bin_labels = [f'{edges[i]:.2g}–{edges[i + 1]:.2g}' for i in range(nb)]

    n_met, n_meth = len(_REGIME_METRICS), len(methods)
    fig, axes = plt.subplots(n_met, n_meth,
                             figsize=(max(1.15 * nb, 1.6) * n_meth + 1.0, 1.35 * n_met + 0.8),
                             dpi=_DPI, constrained_layout=True, squeeze=False)
    fig.suptitle(_ts_suptitle('Erreur par régime de Mach (AoA agrégé)', dataset_results),
                 fontsize=11, fontweight='bold')

    for ri, (key, mlabel) in enumerate(_REGIME_METRICS):
        row_all = np.concatenate([
            np.array(dataset_results[nm].get(key, []), dtype=float) for nm in methods
        ]) if methods else np.array([])
        row_all = row_all[np.isfinite(row_all)]
        vmax = float(np.nanpercentile(row_all, 95)) if row_all.size else 1.0
        vmax = vmax if vmax > 0 else 1.0

        last_pc = None
        for ci, nm in enumerate(methods):
            ax = axes[ri][ci]
            vals = np.asarray(dataset_results[nm].get(key, []), dtype=float)
            if vals.shape[0] != bin_idx.shape[0]:
                vals = np.full(bin_idx.shape[0], np.nan)
            cell = np.full(nb, np.nan)
            for b in range(nb):
                sel = vals[bin_idx == b]
                sel = sel[np.isfinite(sel)]
                if sel.size:
                    cell[b] = float(np.mean(sel))

            last_pc = ax.pcolormesh(np.arange(nb + 1), [0, 1], cell[None, :],
                                    cmap='YlOrRd', vmin=0, vmax=vmax, shading='flat')
            for b in range(nb):
                if np.isfinite(cell[b]):
                    ax.text(b + 0.5, 0.5, f'{cell[b]:.3g}', ha='center', va='center',
                            fontsize=7, color='white' if cell[b] > 0.6 * vmax else 'black')
            ax.set_xticks(np.arange(nb) + 0.5)
            ax.set_yticks([])
            ax.set_xticklabels(bin_labels if ri == n_met - 1 else [],
                               rotation=40, ha='right', fontsize=7)
            if ri == 0:
                ax.set_title(nm, fontsize=9, fontweight='bold')
            if ci == 0:
                ax.set_ylabel(mlabel, fontsize=9, fontweight='bold',
                             rotation=0, ha='right', va='center')

        if last_pc is not None:
            fig.colorbar(last_pc, ax=list(axes[ri]), shrink=0.75, pad=0.015, aspect=14)

    if n_met > 0:
        axes[-1][0].set_xlabel('Mach', fontsize=8)

    out_path = Path(out_dir) / 'regime_heatmap_mach.png'
    plt.savefig(out_path, dpi=_DPI, bbox_inches='tight'); plt.close(fig)
    print(f"  > {out_path.name}")


def _fmt_metric(v: float) -> str:
    if not np.isfinite(v):
        return 'N/A'
    return f'{v:.3e}'


def _fmt_time(v: float) -> str:
    if not np.isfinite(v) or v <= 0:
        return 'N/A'
    if v >= 1000:
        return f'{v/1e3:.1f}s'
    if v >= 1:
        return f'{v:.1f}ms'
    return f'{v*1e3:.1f}µs'


def _summary_methods_rows(results_ref: dict, dataset_results: dict | None):
    """Retourne (methods, col_labels, rows, raw). `raw` a la même forme que `rows`
    hors colonne 'Méthode' : valeurs numériques (nan si N/A), pour déterminer le
    meilleur/pire modèle par colonne sans reparser les chaînes formatées."""
    methods = []
    if results_ref.get('idw'):
        methods.append(results_ref['idw']['name'])
    for row in results_ref.get('rows', []):
        methods.append(row['name'])
    if not methods and dataset_results:
        methods = [nm for nm in dataset_results
                   if not nm.startswith('_') and nm not in ('GT', 'HR')]
    if not methods:
        return methods, [], [], []

    def _raw(nm, key):
        if dataset_results and nm in dataset_results:
            arr = np.array(dataset_results[nm].get(key, [np.nan]), dtype=float)
            return float(np.nanmean(arr))
        return float('nan')

    def _raw_time(nm):
        if dataset_results and nm in dataset_results:
            t = np.array(dataset_results[nm].get('times', [np.nan]), dtype=float)
            v = float(np.nanmean(t))
            if np.isfinite(v) and v > 0:
                return v
        src = results_ref.get('idw') if nm in IDW_BASELINE_NAMES else next(
            (r for r in results_ref.get('rows', []) if r['name'] == nm), None)
        if src:
            return float(np.nanmean(src.get('time_ms', [np.nan])))
        return float('nan')

    has_aero = (dataset_results is not None and
                 any('CL_err' in dataset_results.get(nm, {}) for nm in methods))
    # Cp plutôt que Mach paroi : c'est la grandeur directement comparée aux mesures
    # et à la littérature aéro, même convention que combined._GLOBAL_SUMMARY_METRICS.
    has_cp = (dataset_results is not None and
                 any('wall_cp_l2' in dataset_results.get(nm, {}) for nm in methods))
    has_euler = (dataset_results is not None and
                 any(np.isfinite(np.asarray(dataset_results.get(nm, {}).get('euler_fvm', [np.nan]),
                                            dtype=float)).any() for nm in methods))

    col_labels = ['Méthode', r'$W_2$']
    if has_aero:
        col_labels += [r'$\Delta C_L$', r'$\Delta C_D$']
    if has_cp:
        col_labels += [r'$C_p$ paroi L2']
    if has_euler:
        col_labels += [r'Résidu Euler']
    col_labels += ['Temps/cas']

    rows, raw = [], []
    for nm in methods:
        raw_vals = [_raw(nm, 'w2_mach')]
        if has_aero:
            raw_vals += [_raw(nm, 'CL_err'), _raw(nm, 'CD_err')]
        if has_cp:
            raw_vals += [_raw(nm, 'wall_cp_l2')]
        if has_euler:
            raw_vals += [_raw(nm, 'euler_fvm')]
        raw_vals += [_raw_time(nm)]

        row = [nm] + [_fmt_metric(v) for v in raw_vals[:-1]] + [_fmt_time(raw_vals[-1])]
        rows.append(row)
        raw.append(raw_vals)

    return methods, col_labels, rows, raw


def plot_global_errors(dataset_results: dict, out_dir: Path):
    methods = [nm for nm in dataset_results
               if not nm.startswith('_') and nm not in ('GT', 'HR')
               and 'l2w_mach' in dataset_results.get(nm, {})]
    if not methods:
        return
    palette = _build_palette(methods)
    colors = [palette[nm] for nm in methods]

    fig1, ax1 = plt.subplots(figsize=(6, 4.5), dpi=_DPI, constrained_layout=True)
    metrics = [('l2w_mach', r'$L_{2w}$'), ('w2_mach', r'$W_2$')]
    x = np.arange(len(metrics))
    bw = 0.8 / len(methods)
    for mi, (nm, color) in enumerate(zip(methods, colors)):
        vals = [float(np.nanmean(dataset_results[nm].get(k, [np.nan]))) for k, _ in metrics]
        offset = (mi - (len(methods) - 1) / 2) * bw
        bars = ax1.bar(x + offset, vals, width=bw * 0.9, color=color,
                         label=nm, edgecolor='white')
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.0005,
                         f'{v:.4f}', ha='center', va='bottom', fontsize=7)
    ax1.set_xticks(x); ax1.set_xticklabels([lb for _, lb in metrics])
    ax1.set_ylabel('Erreur (moyenne dataset)')
    ax1.set_title('Erreurs sur le champ de Mach', fontweight='bold')
    ax1.legend(); ax1.grid(True, axis='y', alpha=0.3)
    out1 = out_dir / 'global_errors_mach.png'
    plt.savefig(out1, dpi=_DPI, bbox_inches='tight'); plt.close(fig1)
    print(f"  > {out1.name}")

    prim_labels = [r'$\rho$', r'$u$', r'$v$', r'$p$']
    fig2, ax2 = plt.subplots(figsize=(7, 4.5), dpi=_DPI, constrained_layout=True)
    x = np.arange(4)
    bw = 0.8 / len(methods)
    for mi, (nm, color) in enumerate(zip(methods, colors)):
        data = dataset_results[nm].get('l2_prims', [])
        vals_all = np.array([row for row in data if row is not None], dtype=float)
        if len(vals_all) == 0:
            continue
        means = np.nanmean(vals_all, axis=0)
        offset = (mi - (len(methods) - 1) / 2) * bw
        bars = ax2.bar(x + offset, means, width=bw * 0.9, color=color,
                         label=nm, edgecolor='white')
        for bar, v in zip(bars, means):
            ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.0005,
                     f'{v:.4f}', ha='center', va='bottom', fontsize=6)
    ax2.set_xticks(x); ax2.set_xticklabels(prim_labels)
    ax2.set_ylabel(r'$L_2$ relatif (moyenne dataset)')
    ax2.set_title('Erreurs $L_2$ sur les primitives', fontweight='bold')
    ax2.legend(fontsize=8); ax2.grid(True, axis='y', alpha=0.3)
    out2 = out_dir / 'global_errors_primitives.png'
    plt.savefig(out2, dpi=_DPI, bbox_inches='tight'); plt.close(fig2)
    print(f"  > {out2.name}")


_FVM_TIMES_PATH = Path('data/fvm_times.json')


def _load_fvm_times(path: Path = _FVM_TIMES_PATH) -> dict | None:
    if not path.exists():
        return None
    try:
        import json
        return json.load(open(path))
    except Exception:
        return None


def _fvm_time_at(fvm: dict, geometry: str, res: float) -> float | None:
    """Temps FVM moyen (s/cas) pour (géométrie, résolution) exacte, ou None si cette
    résolution n'a jamais été mesurée. Plus de repli sur la moyenne toutes résolutions
    confondues : sur diamond elles vont de ~1.5 s/cas (h=0.3) à ~200 s/cas (h=0.0125),
    donc cette moyenne n'approxime aucune résolution et affiche un chiffre trompeur."""
    geo = fvm.get('by_geometry_resolution', {}).get(geometry, {})
    key = next((k for k in geo if abs(float(k) - res) < 1e-6), None)
    return geo[key]['time_mean_s'] if key is not None else None


def _fvm_avg_time_text(geometry: str | None, hr_res: float | None,
                       lr_res: float | None = None,
                       fvm_times_path: Path = _FVM_TIMES_PATH) -> str | None:
    """Texte simple 'temps solveur FVM' (LR + HR) pour le jeu de test évalué,
    lu dans data/fvm_times.json (construit par utils/build_fvm_times.py à
    partir des logs bruts logs/euler/)."""
    if geometry is None:
        return None
    fvm = _load_fvm_times(fvm_times_path)
    if fvm is None:
        return None

    parts = []
    if lr_res is not None:
        t = _fvm_time_at(fvm, geometry, lr_res)
        if t is not None:
            parts.append(f"LR h={lr_res:g} : {t:.1f} s/cas")
    if hr_res is not None:
        t = _fvm_time_at(fvm, geometry, hr_res)
        if t is not None:
            parts.append(f"HR h={hr_res:g} : {t:.1f} s/cas")

    if not parts:
        return None
    return "Temps solveur FVM (GPU) — " + '   '.join(parts)


def _color_table_gradient(tbl, raw: list[list[float]], row_names: list[str],
                          n_metric_cols: int, exclude: frozenset[str] = IDW_BASELINE_NAMES):
    """Colore une table matplotlib (lignes 1..len(row_names), colonnes
    0..n_metric_cols) selon un gradient RdYlGn_r continu *par colonne*
    (vert = meilleur/plus bas, rouge = pire/plus haut), calculé parmi les
    lignes hors `exclude` -- toujours grisées. Ailleurs (colonne 'Méthode',
    valeurs manquantes) : fond zébré discret. Remplace l'ancien
    vert/rouge binaire (best/worst uniquement) par une teinte continue qui
    situe aussi les modèles intermédiaires -- cf. combined_global_table."""
    _GRAY = '#c9c9c9'
    cmap = plt.get_cmap('RdYlGn_r')
    model_idx = [i for i, nm in enumerate(row_names) if nm not in exclude]

    for i in range(len(row_names)):
        for j in range(n_metric_cols + 1):
            tbl[i + 1, j].set_facecolor('#eaf0f6' if i % 2 == 0 else 'white')

    for j in range(n_metric_cols):
        vals = [(i, raw[i][j]) for i in model_idx if np.isfinite(raw[i][j])]
        norm = None
        if len(vals) >= 2:
            vmin = min(v for _, v in vals); vmax = max(v for _, v in vals)
            if vmax > vmin:
                norm = (LogNorm(vmin=vmin, vmax=vmax) if (vmin > 0 and vmax / vmin > 5)
                        else Normalize(vmin=vmin, vmax=vmax))
        if norm is not None:
            for i, v in vals:
                tbl[i + 1, j + 1].set_facecolor(cmap(norm(v)))

    for i, nm in enumerate(row_names):
        if nm in exclude:
            for j in range(n_metric_cols + 1):
                tbl[i + 1, j].set_facecolor(_GRAY)


def plot_summary_table(results_ref: dict, dataset_results: dict | None, out_dir: Path,
                       hr_res: float | None = None, lr_res: float | None = None,
                       geometry: str | None = None):
    methods, col_labels, rows, raw = _summary_methods_rows(results_ref, dataset_results)
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(col_labels)), 0.5 + 0.45 * len(rows)),
                           dpi=_DPI)
    ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.5)

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#2c3e50')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    _color_table_gradient(tbl, raw, methods, len(col_labels) - 1)

    fvm_text = _fvm_avg_time_text(geometry, hr_res, lr_res)
    if fvm_text:
        fig.text(0.5, 0.01, fvm_text, ha='center', va='bottom',
                 fontsize=8, style='italic', color='#555555')

    out_path = Path(out_dir) / 'summary_table.png'
    plt.savefig(out_path, dpi=_DPI, bbox_inches='tight'); plt.close(fig)
    print(f"  > {out_path.name}")


def _fvm_live_time_text(fvm_live: dict | None) -> str | None:
    """Texte 'temps solveur FVM (mesure live, cas exact)' pour le pied de page
    du tableau single-case — distinct de _fvm_avg_time_text (agrégat
    data/fvm_times.json) pour ne jamais confondre un temps mesuré à l'instant
    sur CE cas précis avec une moyenne historique sur d'autres cas."""
    if not fvm_live:
        return None
    parts = []
    for key, label in (('lr', 'LR'), ('hr', 'HR')):
        entry = fvm_live.get(key)
        if entry is None:
            continue
        cached = ' (cache)' if entry.get('from_cache') else ''
        parts.append(f"{label} h={entry['h']:g} : {entry['wall_time_s']:.1f} s{cached}")
    if not parts:
        return None
    return "Temps solveur FVM (GPU, mesure live sur ce cas) — " + '   '.join(parts)


def build_single_case_table(results: dict):
    """Retourne (col_labels, rows, raw) pour le tableau récapitulatif d'un cas
    unique, à partir des métriques déjà calculées par
    eval.single_case.run_single_case sur results['idw']/results['rows']
    (listes par cas : 'l2', 'w2', 'time_ms', 'aero' — dict aero_metrics).

    Contrairement à _summary_methods_rows (qui exige un sweep complet pour
    afficher ΔCL/ΔCD/M_wall L2), ce tableau lit directement les métriques du
    (ou des) cas de référence passés — c'est le chaînon manquant pour un
    résumé sur un cas unique.
    """
    methods_rows = []
    if results.get('idw'):
        methods_rows.append(results['idw'])
    methods_rows.extend(results.get('rows', []))
    if not methods_rows:
        return [], [], []

    def _mean(vals):
        vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
        return float(np.mean(vals)) if vals else float('nan')

    has_w2 = any(row.get('w2') for row in methods_rows)
    has_aero = any(row.get('aero') and any(a is not None for a in row['aero'])
                   for row in methods_rows)

    col_labels = ['Méthode', 'L2 Mach']
    if has_w2:
        col_labels.append(r'$W_2$ Mach')
    if has_aero:
        col_labels += [r'$\Delta C_L$', r'$\Delta C_D$', r'$C_p$ paroi L2', r'$M_{wall}$ L2']
    col_labels.append('Temps/cas')

    rows, raw = [], []
    for row in methods_rows:
        raw_vals = [_mean(row.get('l2', []))]
        if has_w2:
            raw_vals.append(_mean(row.get('w2', [])))
        if has_aero:
            aeros = [a for a in row.get('aero', []) if a is not None]
            raw_vals += [
                _mean([a['CL_err_abs'] for a in aeros]),
                _mean([a['CD_err_abs'] for a in aeros]),
                _mean([a['wall_cp_l2'] for a in aeros]),
                _mean([a['wall_mach_l2'] for a in aeros]),
            ]
        raw_vals.append(_mean(row.get('time_ms', [])))

        label = _display_name(row['name'])
        if row.get('step_info'):
            label = f"{label} ({row['step_info']})"
        rows.append([label] + [_fmt_metric(v) for v in raw_vals[:-1]] + [_fmt_time(raw_vals[-1])])
        raw.append(raw_vals)

    return col_labels, rows, raw


def plot_single_case_table(results: dict, out_dir: Path, fvm_live: dict | None = None):
    col_labels, rows, raw = build_single_case_table(results)
    if not rows:
        return

    methods = ([results['idw']['name']] if results.get('idw') else []) + \
              [row['name'] for row in results.get('rows', [])]

    # La 1ère colonne (méthode) peut porter le détail d'échantillonnage
    # (ex. "FAM_dia (16 pas ODE, 4 éch.)") : élargir la figure en conséquence
    # pour ne pas la voir tronquée hors du tableau.
    max_label_len = max((len(r[0]) for r in rows), default=6)
    fig_w = max(6, 1.5 * len(col_labels), 0.13 * max_label_len + 1.2 * (len(col_labels) - 1))
    fig, ax = plt.subplots(figsize=(fig_w, 0.5 + 0.45 * len(rows)),
                           dpi=_DPI)
    ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.5)
    # Colonne 0 (méthode) potentiellement longue (step_info FAM/SIAM) : sans
    # ceci les largeurs par défaut (équiréparties) la tronquent.
    tbl.auto_set_column_width(col=list(range(len(col_labels))))

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#2c3e50')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    _GRAY, _GREEN, _RED = '#c9c9c9', '#a5d6a7', '#ef9a9a'
    n_metric_cols = len(col_labels) - 1
    model_idx = [i for i, nm in enumerate(methods) if nm not in IDW_BASELINE_NAMES]
    best_i: list[int | None] = [None] * n_metric_cols
    worst_i: list[int | None] = [None] * n_metric_cols
    for j in range(n_metric_cols):
        vals = [(i, raw[i][j]) for i in model_idx if np.isfinite(raw[i][j])]
        if vals:
            best_i[j] = min(vals, key=lambda t: t[1])[0]
            worst_i[j] = max(vals, key=lambda t: t[1])[0]

    for i in range(1, len(rows) + 1):
        ri = i - 1
        is_idw = methods[ri] in IDW_BASELINE_NAMES
        for j in range(len(col_labels)):
            cell = tbl[i, j]
            if is_idw:
                cell.set_facecolor(_GRAY)
            elif j > 0 and best_i[j - 1] is not None and best_i[j - 1] != worst_i[j - 1] and ri == best_i[j - 1]:
                cell.set_facecolor(_GREEN)
            elif j > 0 and worst_i[j - 1] is not None and best_i[j - 1] != worst_i[j - 1] and ri == worst_i[j - 1]:
                cell.set_facecolor(_RED)
            else:
                cell.set_facecolor('#eaf0f6' if i % 2 == 0 else 'white')

    footer = _fvm_live_time_text(fvm_live)
    if footer:
        fig.text(0.5, 0.01, footer, ha='center', va='bottom',
                 fontsize=8, style='italic', color='#555555')

    out_path = Path(out_dir) / 'summary_table.png'
    plt.savefig(out_path, dpi=_DPI, bbox_inches='tight'); plt.close(fig)
    print(f"  > {out_path.name}")


def save_single_case_csv(results: dict, out_path: Path):
    col_labels, rows, _ = build_single_case_table(results)
    if not rows:
        return
    clean = [c.replace('$', '').replace('\\', '').replace('{', '').replace('}', '')
             for c in col_labels]
    with open(out_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(clean)
        w.writerows(rows)
    _print_summary_table(clean, rows)
    print(f"  > {Path(out_path).name}")


def _print_summary_table(clean: list, rows: list):
    """Affiche la table récapitulative alignée dans le terminal."""
    widths = [max(len(str(clean[j])), *(len(str(r[j])) for r in rows))
              for j in range(len(clean))]
    sep = '─' * (sum(widths) + 3 * len(widths) + 1)
    fmt = ' │ '.join('{:<' + str(w) + '}' for w in widths)
    print('\n── Résumé des métriques ' + '─' * max(0, len(sep) - 24))
    print(fmt.format(*clean))
    print(sep)
    for r in rows:
        print(fmt.format(*[str(c) for c in r]))
    print()


def save_summary_csv(results_ref: dict, dataset_results: dict | None, out_path: Path):
    _, col_labels, rows, _ = _summary_methods_rows(results_ref, dataset_results)
    if not rows:
        return
    clean = [c.replace('$', '').replace('\\', '').replace('{', '').replace('}', '')
             for c in col_labels]
    with open(out_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(clean)
        w.writerows(rows)
    _print_summary_table(clean, rows)
    print(f"  > {Path(out_path).name}")


__all__ = [
    'plot_reference_grid',
    'plot_wall_profiles_eval',
    'plot_cp_profile_eval',
    'plot_distributions',
    'plot_cl_cd_distributions',
    'plot_error_kde',
    'plot_error_heatmap',
    'plot_regime_heatmap',
    'plot_global_errors',
    'plot_summary_table',
    'save_summary_csv',
    'build_single_case_table',
    'plot_single_case_table',
    'save_single_case_csv',
]
