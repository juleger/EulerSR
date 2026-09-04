"""utils/viz/combined.py — figures d'évaluation combinées (plusieurs TestSets).

Contrairement à l'ancienne approche (pool fusionné de tous les cas), ces figures
gardent la ventilation par testset : on voit quel modèle est bon sur quel test.
Chaque figure porte en sous-titre la liste des tests évalués.
"""
from __future__ import annotations
from pathlib import Path

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as _mgs
from matplotlib.colors import LogNorm

from utils.viz._style import _DPI, CMAP_FIELD as _CMAP_FIELD
from utils.viz.eval import (_build_palette, _display_name, _kde_or_hist,
                            _color_table_gradient, _fmt_metric, _fmt_time)
from eval.core import IDW_BASELINE_NAMES

# Tableau global : les métriques essentielles (erreurs aéro/paroi + distributionnelle + coût)
_TABLE_METRICS = [
    ('w2_mach', r'$W_2$ Mach'),
    ('CL_err', r'$\Delta C_L$'),
    ('CD_err', r'$\Delta C_D$'),
    ('wall_mach_l2', r'$M_{wall}$ L2'),
    ('times', 't (ms/cas)'),
]
# Table récapitulative unique (une ligne par modèle, moyennée sur tous les
# testsets) : mêmes erreurs que _TABLE_METRICS mais Cp paroi plutôt que Mach
# paroi -- Cp est la grandeur directement comparée aux mesures/litté aéro.
_GLOBAL_SUMMARY_METRICS = [
    ('w2_mach', r'$W_2$'),
    ('CL_err', r'$\Delta C_L$'),
    ('CD_err', r'$\Delta C_D$'),
    ('wall_cp_l2', r'$C_p$ paroi L2'),
    ('times', 'Temps/cas'),
]
# Barres + distributions : erreurs de champ (L2w, W2) + erreur paroi
_ERR_METRICS = [
    ('l2w_mach', r'$L_{2w}$ Mach'),
    ('w2_mach', r'$W_2$ Mach'),
    ('wall_mach_l2', r'$M_{wall}$ L2'),
]
# Colonnes du CSV (superset, pour garder aussi L2w/Linf sous la main)
_CSV_KEYS = ['w2_mach', 'l2w_mach', 'linf_mach', 'CL_err', 'CD_err',
             'wall_mach_l2', 'wall_cp_l2', 'times', 'enthalpy', 'entropy']


def _mean(v) -> float:
    """Moyenne robuste (ignore None/NaN) d'une liste hétérogène de métriques."""
    if v is None:
        return np.nan
    a = np.array([x if x is not None else np.nan for x in v], dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else np.nan


def _vals(v) -> np.ndarray:
    if v is None:
        return np.array([])
    a = np.array([x if x is not None else np.nan for x in v], dtype=float)
    return a[np.isfinite(a)]


def collect(ts_results: dict) -> dict:
    """Normalise ts_results en une structure facile à parcourir par (test, modèle).

    Retour :
      tags    : ordre des testsets ayant un sweep
      labels  : {tag: label lisible}
      models  : liste ordonnée des noms de modèle (base), 'LR IDW' en tête
      get(tag, model, key) -> liste de valeurs (ou None)
    """
    tags, labels = [], {}
    per_tag: dict[str, dict[str, dict]] = {}
    models: list[str] = []
    seen = set()

    for tag, v in ts_results.items():
        sweep = v.get('sweep') if isinstance(v, dict) else None
        if not sweep:
            continue
        tags.append(tag)
        labels[tag] = sweep.get('_ts_label', tag)
        name_map = sweep.get('_name_map', {})
        per_tag[tag] = {}
        for disp, base in name_map.items():
            if disp in sweep:
                per_tag[tag][base] = sweep[disp]
                if base not in seen:
                    seen.add(base)
                    models.append(base)

    # Baseline sans réseau ('LR IDW' ou 'Extrap. Bord') toujours en tête si présente.
    _bl = next((m for m in models if m in IDW_BASELINE_NAMES), None)
    if _bl is not None:
        models.remove(_bl)
        models.insert(0, _bl)

    def get(tag, model, key):
        return per_tag.get(tag, {}).get(model, {}).get(key)

    return {'tags': tags, 'labels': labels, 'models': models, 'get': get}


def _subtitle(C: dict) -> str:
    """Sous-titre listant les tests évalués (petit, discret)."""
    return 'Tests : ' + '  ·  '.join(C['labels'][t] for t in C['tags'])


def _suptitle(fig, title: str, C: dict):
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.995)
    fig.text(0.5, 0.955, _subtitle(C), ha='center', va='top',
             fontsize=8, color='#555555', style='italic')


def combined_global_table(ts_results: dict, out_dir: Path, min_groups: int = 2):
    """Matrices modèle × testset (une par métrique) : le tableau global visuel."""
    C = collect(ts_results)
    tags, models = C['tags'], C['models']
    if len(tags) < min_groups or not models:
        return

    metrics = _TABLE_METRICS
    # Grille compacte : max 3 métriques par rangée (évite la bande trop large)
    ncols = min(len(metrics), 3)
    nrows = (len(metrics) + ncols - 1) // ncols
    fig = plt.figure(figsize=(ncols * (1.35 * len(tags) + 0.7) + 0.4,
                              nrows * (0.42 * len(models) + 1.3) + 0.6),
                     dpi=_DPI)
    gs = _mgs.GridSpec(nrows, ncols, figure=fig, wspace=0.45, hspace=0.75)

    disp_models = [_display_name(m) for m in models]
    for mi, (key, mlabel) in enumerate(metrics):
        gr, gc = divmod(mi, ncols)
        ax = fig.add_subplot(gs[gr, gc])
        M = np.full((len(models), len(tags)), np.nan)
        for i, m in enumerate(models):
            for j, t in enumerate(tags):
                M[i, j] = _mean(C['get'](t, m, key))

        finite = M[np.isfinite(M)]
        if finite.size:
            vmin = max(finite.min(), 1e-9)
            vmax = finite.max()
            norm = LogNorm(vmin=vmin, vmax=vmax) if vmax / vmin > 5 else None
        else:
            norm = None
        im = ax.imshow(M, cmap='RdYlGn_r', norm=norm, aspect='auto',
                       vmin=None if norm else np.nanmin(M),
                       vmax=None if norm else np.nanmax(M))

        ax.set_xticks(range(len(tags)))
        ax.set_xticklabels(tags, rotation=30, ha='right', fontsize=8)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(disp_models if gc == 0 else [''] * len(models),
                           fontsize=8)
        ax.set_title(mlabel, fontsize=10, fontweight='bold')
        # annotation valeurs
        for i in range(len(models)):
            for j in range(len(tags)):
                if not np.isfinite(M[i, j]):
                    continue
                txt = f'{M[i, j]:.2f}' if key == 'times' else f'{M[i, j]:.3f}'
                ax.text(j, i, txt, ha='center', va='center', fontsize=7,
                        color='black')
        ax.set_xticks(np.arange(-.5, len(tags), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(models), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=1.2)
        ax.tick_params(which='minor', length=0)

    _suptitle(fig, 'Tableau global — moyenne par test (plus bas = mieux)', C)
    out_path = Path(out_dir) / 'global_table.png'
    plt.savefig(out_path, dpi=_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  > {out_path.name}")


def combined_summary_table(ts_results: dict, out_dir: Path, min_groups: int = 2):
    """Table récapitulative unique (une ligne par modèle) façon summary_table
    mono-géométrie, mais moyennée sur tous les testsets évalués -- complète
    combined_global_table (qui garde la ventilation par testset, utile pour
    diagnostiquer, mais illisible d'un coup d'œil dès qu'il y a beaucoup de
    testsets) par un classement global immédiat. Colonnes : W2 Mach, ΔCL,
    ΔCD, Cp paroi L2 (Mwall exclu ici, cf. _GLOBAL_SUMMARY_METRICS), temps/cas.
    """
    C = collect(ts_results)
    tags, models = C['tags'], C['models']
    if len(tags) < min_groups or not models:
        return

    metrics = _GLOBAL_SUMMARY_METRICS
    col_labels = ['Méthode'] + [lbl for _, lbl in metrics]

    raw: list[list[float]] = []
    for m in models:
        row = []
        for key, _ in metrics:
            per_tag = [v for v in (_mean(C['get'](t, m, key)) for t in tags) if np.isfinite(v)]
            row.append(float(np.mean(per_tag)) if per_tag else float('nan'))
        raw.append(row)

    rows = [[_display_name(m)] +
            [(_fmt_time(v) if key == 'times' else _fmt_metric(v))
             for (key, _), v in zip(metrics, r)]
            for m, r in zip(models, raw)]

    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(col_labels)), 0.9 + 0.45 * len(rows)),
                           dpi=_DPI)
    ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.5)

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#2c3e50')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    _color_table_gradient(tbl, raw, models, len(col_labels) - 1)

    # Titre + sous-titre en offset points (pas en fraction de figure) : cette
    # figure est bien plus basse que les grilles GridSpec où _suptitle est
    # calibré (0.995/0.955), et ces deux lignes s'y chevaucheraient sinon.
    ax.annotate('Tableau récapitulatif global — moyenne sur tous les tests',
               xy=(0.5, 1.0), xycoords='axes fraction', xytext=(0, 42),
               textcoords='offset points', ha='center', va='bottom',
               fontsize=13, fontweight='bold')
    ax.annotate(_subtitle(C), xy=(0.5, 1.0), xycoords='axes fraction', xytext=(0, 26),
               textcoords='offset points', ha='center', va='bottom',
               fontsize=8, style='italic', color='#555555')

    out_path = Path(out_dir) / 'summary_table_global.png'
    plt.savefig(out_path, dpi=_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  > {out_path.name}")


def combined_metric_bars(ts_results: dict, out_dir: Path, min_groups: int = 2):
    """Barres groupées : par testset (groupes X), une barre par modèle. Erreurs."""
    C = collect(ts_results)
    tags, models = C['tags'], C['models']
    if len(tags) < min_groups or not models:
        return
    palette = _build_palette(models)

    fig, axes = plt.subplots(1, len(_ERR_METRICS),
                             figsize=(6.2 * len(_ERR_METRICS), 4.6),
                             dpi=_DPI)
    if len(_ERR_METRICS) == 1:
        axes = [axes]

    x = np.arange(len(tags))
    w = 0.8 / max(len(models), 1)
    for ax, (key, label) in zip(axes, _ERR_METRICS):
        for k, m in enumerate(models):
            vals = [_mean(C['get'](t, m, key)) for t in tags]
            ax.bar(x + (k - (len(models) - 1) / 2) * w, vals, w,
                   label=_display_name(m), color=palette[m],
                   edgecolor='white', linewidth=0.4)
        ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels(tags, rotation=20, ha='right', fontsize=9)
        ax.set_ylabel(f'{label}  (moyenne, log)', fontsize=10)
        ax.set_title(label, fontweight='bold', fontsize=11)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    axes[0].legend(fontsize=8, ncol=1, framealpha=0.9)

    _suptitle(fig, 'Erreur moyenne par test et par modèle', C)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path = Path(out_dir) / 'bars_errors.png'
    plt.savefig(out_path, dpi=_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  > {out_path.name}")


def combined_error_distributions(ts_results: dict, out_dir: Path, min_groups: int = 2):
    """Grille KDE : lignes = métriques d'erreur, colonnes = testsets."""
    C = collect(ts_results)
    tags, models = C['tags'], C['models']
    if len(tags) < min_groups or not models:
        return
    palette = _build_palette(models)

    nrow, ncol = len(_ERR_METRICS), len(tags)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.2 * nrow),
                             dpi=_DPI, squeeze=False)

    for ri, (key, label) in enumerate(_ERR_METRICS):
        for ci, t in enumerate(tags):
            ax = axes[ri][ci]
            for m in models:
                v = _vals(C['get'](t, m, key))
                if v.size >= 5:
                    _kde_or_hist(ax, v, palette[m], _display_name(m),
                                 fill_alpha=0.08)
            if ri == 0:
                ax.set_title(C['labels'][t], fontsize=9, fontweight='bold')
            if ci == 0:
                ax.set_ylabel(f'{label}\nDensité', fontsize=9)
            ax.grid(True, alpha=0.25, linestyle='--')
            ax.tick_params(labelsize=8)
    axes[0][0].legend(fontsize=7, framealpha=0.9)

    _suptitle(fig, "Distribution des erreurs par cas — ventilé par test", C)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path = Path(out_dir) / 'error_distributions.png'
    plt.savefig(out_path, dpi=_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  > {out_path.name}")

def combined_reference_fields(ts_results: dict, out_dir: Path):
    """Panel Mach : une ligne par testset (cas de référence), colonnes = méthodes."""
    C = collect(ts_results)
    tags = C['tags']
    model_cols = [m for m in C['models'] if m not in IDW_BASELINE_NAMES]

    # testsets exploitables : ref + triang + cas avec HR
    usable = []
    for t in tags:
        v = ts_results[t]
        ref = v.get('ref')
        tri = v.get('triang')
        if ref and tri is not None and ref.get('cases'):
            if any(c['mach_ref'] is not None for c in ref['cases']):
                usable.append(t)
    if len(usable) < 2:
        return

    # colonnes : baseline, modèles..., HR. Nom réel de la baseline lu dans les
    # données plutôt qu'un littéral fixe, cf. eval.core.idw_baseline_name.
    _idw_name = next((ts_results[t]['ref']['idw']['name'] for t in usable
                      if ts_results[t]['ref'].get('idw')), None)
    col_names = ([_idw_name] if _idw_name else []) + \
                [_display_name(m) for m in model_cols] + ['HR GT']
    ncol = len(col_names)

    # ratio d'aspect par testset (pour height_ratios)
    def _ar(tri):
        return max(float(tri.y.max() - tri.y.min()) /
                   max(float(tri.x.max() - tri.x.min()), 1e-6), 0.15)
    ars = [_ar(ts_results[t]['triang']) for t in usable]

    pw = 2.4
    fig = plt.figure(figsize=(ncol * pw + 0.4, sum(a * pw for a in ars) + 1.2),
                     dpi=_DPI)
    gs = _mgs.GridSpec(len(usable), ncol + 1, figure=fig,
                       width_ratios=[*([pw] * ncol), 0.22],
                       height_ratios=ars, hspace=0.06, wspace=0.02)

    for ri, t in enumerate(usable):
        ref = ts_results[t]['ref']
        tri = ts_results[t]['triang']
        # premier cas de référence avec HR
        ci_case = next(i for i, c in enumerate(ref['cases'])
                       if c['mach_ref'] is not None)
        case = ref['cases'][ci_case]
        mach_ref = case['mach_ref']
        vmin, vmax = float(mach_ref.min()), float(mach_ref.max())

        # colonnes de données alignées sur col_names
        cols = []
        if _idw_name is not None:
            cols.append(ref['idw']['mach_preds'][ci_case] if ref.get('idw') else None)
        for row in ref['rows']:
            cols.append(row['mach_preds'][ci_case])
        cols.append(mach_ref)  # HR

        sc = None
        for cj, mach in enumerate(cols):
            ax = fig.add_subplot(gs[ri, cj])
            if ri == 0:
                ax.set_title(col_names[cj], fontsize=8, fontweight='bold', pad=3)
            if mach is None:
                ax.axis('off')
                continue
            sc = ax.tripcolor(tri, facecolors=np.clip(mach, vmin, vmax),
                              cmap=_CMAP_FIELD, vmin=vmin, vmax=vmax)
            ax.set_aspect('equal'); ax.axis('off')
        # étiquette ligne = testset + régime
        ax0 = fig.add_subplot(gs[ri, 0])
        ax0.axis('off')
        fig.text(gs[ri, 0].get_position(fig).x0 - 0.005,
                 gs[ri, 0].get_position(fig).y0 +
                 gs[ri, 0].get_position(fig).height / 2,
                 f"{C['labels'][t]}\nM={case['mach_in']:.2f}  AoA={case['aoa_in']:+.0f}°",
                 rotation=90, va='center', ha='right', fontsize=7.5,
                 fontweight='bold')
        if sc is not None:
            cax = fig.add_subplot(gs[ri, ncol])
            fig.colorbar(sc, cax=cax, label='Mach')

    _suptitle(fig, 'Champs Mach — cas de référence par test', C)
    out_path = Path(out_dir) / 'reference_fields.png'
    plt.savefig(out_path, dpi=_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  > {out_path.name}")


def combined_summary_csv(ts_results: dict, out_dir: Path):
    C = collect(ts_results)
    tags, models = C['tags'], C['models']
    keys = _CSV_KEYS
    out_path = Path(out_dir) / 'summary_combined.csv'
    with open(out_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['testset', 'model'] + keys)
        for t in tags:
            for m in models:
                w.writerow([t, _display_name(m)] +
                           [f'{_mean(C["get"](t, m, k)):.6g}' for k in keys])
    print(f"  > {out_path.name}")


def plot_all_combined(ts_results: dict, out_dir: Path):
    combined_global_table(ts_results, out_dir)
    combined_summary_table(ts_results, out_dir)
    combined_metric_bars(ts_results, out_dir)
    combined_error_distributions(ts_results, out_dir)
    combined_reference_fields(ts_results, out_dir)
    combined_summary_csv(ts_results, out_dir)


def plot_by_geometry(by_geom: dict, out_dir: Path):
    """Mêmes figures essentielles que plot_all_combined, mais ventilées par
    géométrie seule"""
    
    combined_global_table(by_geom, out_dir, min_groups=1)
    combined_metric_bars(by_geom, out_dir, min_groups=1)
    combined_error_distributions(by_geom, out_dir, min_groups=1)
    combined_summary_csv(by_geom, out_dir)


__all__ = [
    'collect', 'combined_global_table', 'combined_summary_table',
    'combined_metric_bars', 'combined_error_distributions',
    'combined_reference_fields', 'combined_summary_csv',
    'plot_all_combined', 'plot_by_geometry',
]
