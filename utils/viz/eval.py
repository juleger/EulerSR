"""utils/viz/eval.py — visualisations d'évaluation SR-CFD."""
from __future__ import annotations
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as _mgs
from scipy.stats import gaussian_kde
from scipy.interpolate import griddata

from utils.viz._style import _DPI, CMAP_FIELD as _CMAP_FIELD, CMAP_ERR as _CMAP_ERR

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
                _h = _p.height * 0.80
                cb_axs[ri].set_position([_p.x0, _p.y0 + (_p.height - _h) / 2, _p.width, _h])
                fig.colorbar(sc_row, cax=cb_axs[ri], label='Mach')

        plt.savefig(out_fields, dpi=_DPI, bbox_inches='tight')
        plt.close(fig)
        print(f"  > {out_fields.name}")

    err_cols: list[dict] = []
    if any(has_ref):
        if results['idw']:
            err_cols.append({'name': 'LR IDW', 'mach_preds': results['idw']['mach_preds'],
                             'l2w': results['idw'].get('l2w', [])})
        for row in results['rows']:
            err_cols.append({'name': row['name'], 'mach_preds': row['mach_preds'],
                             'l2w': row.get('l2w', [])})

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
    if n_cases == 1: axes = axes[np.newaxis, :]
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
            l2w_v = (cd.get('l2w', []) or []); l2w_v = l2w_v[ri] if ri < len(l2w_v) else None
            if l2w_v is not None:
                ax.text(0.5, 0.03, f'L2w={l2w_v:.3f}',
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


_REGIMES = [
    ('0.7 ≤ M ≤ 1.3', lambda m: (m >= 0.7) & (m <= 1.3)),  # Transsonique
    ('M > 1.3', lambda m: m > 1.3),  # Supersonique
    ('M < 0.7', lambda m: m < 0.7),  # Subsonique
]


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


def plot_aero_distributions(dataset_results: dict, out_dir: Path):
    method_names = [nm for nm in dataset_results if not nm.startswith('_')]
    palette = _build_palette(method_names)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=_DPI, constrained_layout=True)
    fig.suptitle(_ts_suptitle('Distributions aérodynamiques', dataset_results),
                 fontsize=12, fontweight='bold')

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
            _kde_or_hist(ax, vals, palette[nm], _display_name(nm),
                         lw=_HR_LW if is_hr else 1.4,
                         ls=_HR_LS if is_hr else '-',
                         fill_alpha=0.0 if is_hr else 0.10,
                         zorder=3 if is_hr else 2,
                         alpha=_HR_ALPHA if is_hr else 1.0)
        ax.set_xlabel(xlabel, fontsize=10); ax.set_ylabel('Densité', fontsize=9)
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.25, linestyle='--')

    out_path = out_dir / 'dist_aero.png'
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
                       for k in ('l2_mach', 'l2w_mach', 'sw2_mach'))]
    if not methods:
        return

    palette = _build_palette(methods)
    metrics = [('l2w_mach', r'$L_{2w}$ Mach'), ('sw2_mach', r'$SW_2$ Mach')]
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


def plot_error_scatter(dataset_results: dict, all_cases: list[dict], out_dir: Path,
                       metric_key: str = 'l2w_mach',
                       metric_label: str = r'$L_{2w}$',
                       fname: str = 'error_vs_mach.png'):
    methods = [nm for nm in dataset_results
               if not nm.startswith('_') and nm not in ('GT', 'HR')
               and metric_key in dataset_results.get(nm, {})]
    if not methods:
        return

    machs = np.array([c['mach_in'] for c in all_cases])
    aoas = np.array([c['aoa_in'] for c in all_cases])

    n_m = len(methods)
    fig, axes = plt.subplots(1, n_m, figsize=(4 * n_m, 4),
                             dpi=_DPI, constrained_layout=True, sharey=True)
    fig.set_constrained_layout_pads(h_pad=0.02, w_pad=0.02, hspace=0.02, wspace=0.03)
    if n_m == 1:
        axes = [axes]
    fig.suptitle(_ts_suptitle(f'Erreur {metric_label} Mach vs Mach (couleur = AoA)',
                              dataset_results),
                 fontsize=11, fontweight='bold')

    from scipy.ndimage import uniform_filter1d
    for ax, nm in zip(axes, methods):
        vals = np.array(dataset_results[nm][metric_key], dtype=float)
        valid = np.isfinite(vals)
        sc = ax.scatter(machs[valid], vals[valid], c=aoas[valid],
                        cmap='coolwarm', s=18, alpha=0.7,
                        vmin=aoas.min(), vmax=aoas.max(), linewidths=0)
        order = np.argsort(machs[valid])
        smooth = uniform_filter1d(vals[valid][order], size=max(1, valid.sum() // 30))
        ax.plot(machs[valid][order], smooth, 'k-', lw=1.5, alpha=0.6, label='tendance')
        ax.set_xlabel('Mach'); ax.set_title(nm, fontweight='bold')
        ax.grid(True, alpha=0.25, linestyle='--')
        plt.colorbar(sc, ax=ax, label='AoA (°)', shrink=0.8)

    axes[0].set_ylabel(f'{metric_label} — Mach')
    out_path = Path(out_dir) / fname
    plt.savefig(out_path, dpi=_DPI, bbox_inches='tight'); plt.close(fig)
    print(f"  > {out_path.name}")


def _heatmap_metric(dataset_results: dict, all_cases: list[dict], out_dir: Path,
                    metric_key: str, metric_label: str, fname: str):
    methods = [nm for nm in dataset_results
               if not nm.startswith('_') and nm not in ('GT', 'HR', 'LR IDW')
               and metric_key in dataset_results.get(nm, {})]
    if not methods:
        return

    machs = np.array([c['mach_in'] for c in all_cases])
    aoas = np.array([c['aoa_in'] for c in all_cases])
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
                    'sw2_mach', r'$SW_2$ Mach', 'error_heatmap_sw2.png')


def _regime_bars_metric(dataset_results: dict, all_cases: list[dict], out_dir: Path,
                        metric_key: str, ylabel: str, fname: str):
    methods = [nm for nm in dataset_results
               if not nm.startswith('_') and nm not in ('GT', 'HR')
               and metric_key in dataset_results.get(nm, {})]
    if not methods:
        return

    palette = _build_palette(methods)
    machs = np.array([c['mach_in'] for c in all_cases])

    data: dict[str, list[float]] = {nm: [] for nm in methods}
    counts: list[int] = []
    for _, mask_fn in _REGIMES:
        idx = np.where(mask_fn(machs))[0]
        counts.append(len(idx))
        for nm in methods:
            vals_arr = np.array(dataset_results[nm][metric_key], dtype=float)
            v_reg = vals_arr[idx]
            data[nm].append(float(np.nanmean(v_reg)) if np.any(np.isfinite(v_reg)) else np.nan)

    n_reg, n_met = len(_REGIMES), len(methods)
    x, bw = np.arange(n_reg), 0.75 / n_met

    fig, ax = plt.subplots(figsize=(max(6, n_reg * 2.5), 4.5),
                           dpi=_DPI, constrained_layout=True)
    for mi, nm in enumerate(methods):
        color = palette[nm]
        offset = (mi - (n_met - 1) / 2) * bw
        bars = ax.bar(x + offset, data[nm], width=bw * 0.9,
                        color=color, label=nm, edgecolor='white')
        for bar, v in zip(bars, data[nm]):
            if np.isfinite(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.0005,
                        f'{v:.4f}', ha='center', va='bottom', fontsize=7)

    tick_labels = [f'{lbl}\n(N={c})' for (lbl, _), c in zip(_REGIMES, counts)]
    ax.set_xticks(x); ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(f'Erreur {ylabel} par régime d\'écoulement', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.3, linestyle='--')

    out_path = Path(out_dir) / fname
    plt.savefig(out_path, dpi=_DPI, bbox_inches='tight'); plt.close(fig)
    print(f"  > {out_path.name}")


def plot_regime_bars(dataset_results: dict, all_cases: list[dict], out_dir: Path):
    _regime_bars_metric(dataset_results, all_cases, out_dir,
                        'l2w_mach', r'$L_{2w}$ Mach (moyenne)', 'error_by_regime_l2w.png')
    _regime_bars_metric(dataset_results, all_cases, out_dir,
                        'sw2_mach', r'$SW_2$ Mach (moyenne)', 'error_by_regime_sw2.png')

def _fmt_metric(v: float) -> str:
    if not np.isfinite(v):
        return 'N/A'
    return f'{v:.2e}'


def _fmt_time(v: float) -> str:
    if not np.isfinite(v) or v <= 0:
        return 'N/A'
    if v >= 1000:
        return f'{v/1e3:.1f}s'
    if v >= 1:
        return f'{v:.1f}ms'
    return f'{v*1e3:.1f}µs'


def _summary_methods_rows(results_ref: dict, dataset_results: dict | None):
    methods = []
    if results_ref.get('idw'):
        methods.append('LR IDW')
    for row in results_ref.get('rows', []):
        methods.append(row['name'])
    if not methods and dataset_results:
        methods = [nm for nm in dataset_results
                   if not nm.startswith('_') and nm not in ('GT', 'HR')]
    if not methods:
        return methods, [], []

    def _get(nm, key):
        if dataset_results and nm in dataset_results:
            arr = np.array(dataset_results[nm].get(key, [np.nan]), dtype=float)
            return _fmt_metric(float(np.nanmean(arr)))
        return 'N/A'

    def _get_time(nm):
        if dataset_results and nm in dataset_results:
            t = np.array(dataset_results[nm].get('times', [np.nan]), dtype=float)
            v = float(np.nanmean(t))
            if np.isfinite(v) and v > 0:
                return _fmt_time(v)
        src = results_ref.get('idw') if nm == 'LR IDW' else next(
            (r for r in results_ref.get('rows', []) if r['name'] == nm), None)
        if src:
            return _fmt_time(float(np.nanmean(src.get('time_ms', [np.nan]))))
        return 'N/A'

    has_aero = (dataset_results is not None and
                 any('CL_err' in dataset_results.get(nm, {}) for nm in methods))
    has_cp = (dataset_results is not None and
                 any('wall_cp_l2' in dataset_results.get(nm, {}) for nm in methods))
    has_euler = (dataset_results is not None and
                 any(np.isfinite(np.asarray(dataset_results.get(nm, {}).get('euler_fvm', [np.nan]),
                                            dtype=float)).any() for nm in methods))

    col_labels = ['Méthode', r'$L_{2w}$', r'$SW_2$']
    if has_aero:
        col_labels += [r'$\Delta C_L$', r'$\Delta C_D$']
    if has_cp:
        col_labels += [r'$C_p$ L2']
    if has_euler:
        col_labels += [r'Résidu Euler']
    col_labels += ['Temps/cas']

    rows = []
    for nm in methods:
        row = [nm, _get(nm, 'l2w_mach'), _get(nm, 'sw2_mach')]
        if has_aero:
            row += [_get(nm, 'CL_err'), _get(nm, 'CD_err')]
        if has_cp:
            row += [_get(nm, 'wall_cp_l2')]
        if has_euler:
            row += [_get(nm, 'euler_fvm')]
        row += [_get_time(nm)]
        rows.append(row)

    return methods, col_labels, rows


def plot_timing(results_ref: dict, dataset_results: dict | None,
                euler_times: dict, out_path: Path):
    labels, values, colors = [], [], []

    if results_ref['idw']:
        nm_idw = 'LR IDW'
        t_idw = (float(np.nanmean(dataset_results[nm_idw]['times']))
                  if dataset_results and nm_idw in dataset_results
                  else float(np.nanmean(results_ref['idw']['time_ms'])))
        labels.append('LR IDW'); values.append(t_idw); colors.append(_METHOD_COLORS[0])

    for i, row in enumerate(results_ref['rows']):
        nm_sr = row['name']
        t_sr = (float(np.nanmean(dataset_results[nm_sr]['times']))
                 if dataset_results and nm_sr in dataset_results
                 else float(np.nanmean(row['time_ms'])))
        labels.append(nm_sr); values.append(t_sr)
        colors.append(_METHOD_COLORS[(i + 1) % len(_METHOD_COLORS)])

    if euler_times:
        t_fvm_ms = float(np.median(list(euler_times.values()))) * 1e3
        labels.append('FVM (solveur)'); values.append(t_fvm_ms)
        colors.append('dimgray')

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4), 4.5), dpi=_DPI)
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, width=0.55, edgecolor='white', linewidth=0.5)
    ax.set_yscale('log')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel("Temps d'inférence moyen (ms/cas, échelle log)")
    ax.set_title("Comparaison des temps de calcul : SR vs solveur FVM", fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')

    for bar, v in zip(bars, values):
        s = f'{v/1e3:.1f}s' if v >= 1e3 else (f'{v:.0f}ms' if v >= 1 else f'{v*1e3:.1f}µs')
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.3, s,
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    fig.tight_layout()
    plt.savefig(out_path, dpi=_DPI, bbox_inches='tight'); plt.close(fig)
    print(f"  > {out_path.name}")


def plot_global_errors(dataset_results: dict, out_dir: Path):
    methods = [nm for nm in dataset_results
               if not nm.startswith('_') and nm not in ('GT', 'HR')
               and 'l2w_mach' in dataset_results.get(nm, {})]
    if not methods:
        return
    palette = _build_palette(methods)
    colors = [palette[nm] for nm in methods]

    fig1, ax1 = plt.subplots(figsize=(6, 4.5), dpi=_DPI, constrained_layout=True)
    metrics = [('l2w_mach', r'$L_{2w}$'), ('sw2_mach', r'$SW_2$')]
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


def plot_summary_table(results_ref: dict, dataset_results: dict | None, out_dir: Path):
    _, col_labels, rows = _summary_methods_rows(results_ref, dataset_results)
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
    for i in range(1, len(rows) + 1):
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor('#eaf0f6' if i % 2 == 0 else 'white')

    out_path = Path(out_dir) / 'summary_table.png'
    plt.savefig(out_path, dpi=_DPI, bbox_inches='tight'); plt.close(fig)
    print(f"  > {out_path.name}")


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
    _, col_labels, rows = _summary_methods_rows(results_ref, dataset_results)
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
    'plot_distributions',
    'plot_aero_distributions',
    'plot_cl_cd_distributions',
    'plot_error_kde',
    'plot_error_scatter',
    'plot_error_heatmap',
    'plot_regime_bars',
    'plot_timing',
    'plot_global_errors',
    'plot_summary_table',
    'save_summary_csv',
]
