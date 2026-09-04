"""Script principal d'évaluation des modèles SR-CFD sur les test sets"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / 'euler') not in sys.path:
    sys.path.append(str(REPO_ROOT / 'euler'))

import euler.jax_fvm.src.mesh  # noqa: F401

from flax import nnx

from eval.loader import load_model, ModelEntry
from eval.testset import TestSet
from eval.runner import evaluate, plot_combined
from eval.core import _BATCH_SIZE
from utils.layout import DataLayout


def _parse_testset_spec(spec: str, data_root: Path,
                        default_lr: float, default_hr: float) -> DataLayout:
    """Parse 'geometry' ou 'geometry:lr_res' ou 'geometry:lr_res:hr_res'."""
    parts = spec.split(':')
    geometry = parts[0]
    lr_res = float(parts[1]) if len(parts) > 1 else default_lr
    hr_res = float(parts[2]) if len(parts) > 2 else default_hr
    return DataLayout.from_root(data_root, geometry, lr_res, hr_res)


def _run_paired(args, testsets: list[TestSet], out_dir: Path) -> None:
    """Mode --paired : model[i] évalué uniquement sur testsets[i], chaque modèle avec
    sa propre condition d'entrée (ex. FAMWall entraînés à des N_wall différents). Les
    inférences se calculent séparément par paire, car load_sample dérive le store
    d'observations de bord du chemin des test_cases, puis sont fusionnées en un seul
    jeu de résultats avant de tracer. Le testset canonique (mesh, triang et GT
    partagés) est testsets[0] : 'cases', HR et baseline viennent de cette première
    paire, seule la ligne de chaque modèle est injectée depuis la sienne."""
    from eval.runner import _run_ref_cases, _run_sweep, _plot_ref, _plot_sweep, \
        _save_summary_csv, _maybe_recalibrate
    from utils.gt_cache import load_or_build_gt_cache
    from utils.viz.eval import _summary_methods_rows
    import numpy as np

    t0 = time.perf_counter()
    n_eval = args.n_eval if args.full_eval else 0
    canonical_ts = testsets[0]
    out = out_dir / f'{canonical_ts.tag}__paired'
    out.mkdir(parents=True, exist_ok=True)

    merged_ref: dict | None = None
    merged_sweep: dict | None = None
    merged_cases = None
    entries = []

    print("\n── Mode apparié fusionné (chaque modèle sur son propre testset, "
          "tracé comme un seul) ──")
    for ckpt, ts in zip(args.models, testsets):
        entry = load_model(Path(ckpt), ts.layout)
        if args.n_steps is not None and hasattr(entry.model, 'n_steps'):
            entry.model.n_steps = args.n_steps
        if args.n_samples is not None and hasattr(entry.model, 'n_samples'):
            entry.model.n_samples = args.n_samples
        if args.cfg_scale is not None and hasattr(entry.model, 'cfg_scale'):
            entry.model.cfg_scale = args.cfg_scale
        if args.geom_cfg_scale is not None and hasattr(entry.model, 'geom_cfg_scale'):
            entry.model.geom_cfg_scale = args.geom_cfg_scale
        print(f"  {entry.name} [{entry.kind}]  <-> {ts.tag}")
        entries.append(entry)

        knn_map = {entry.name: ts.build_knn(entry)}
        calib_excluded = _maybe_recalibrate([entry], ts, knn_map)
        is_wall_eval = hasattr(entry.model, 'wall_encoder')
        idw_knn = ts.build_idw_knn(k=6, wall=is_wall_eval)

        if ts.ref_cases:
            ref_res = _run_ref_cases([entry], ts, knn_map, idw_knn)
            if merged_ref is None:
                merged_ref = {'cases': ref_res['cases'], 'rows': [], 'idw': ref_res['idw']}
            merged_ref['rows'].extend(ref_res['rows'])

        test_cases = [c for c in ts.test_cases if c['path'] not in calib_excluded]
        if not test_cases and ts.test_cases:
            print(f"  [sweep] {ts.tag} : {len(ts.test_cases)} cas de test, tous "
                  f"consommes par la calibration, sweep saute.")
        if test_cases:
            cases = test_cases[:n_eval] if n_eval > 0 else test_cases
            mesh_hr = np.load(ts.layout.mesh_path(ts.hr_res), allow_pickle=True).item()
            mu_gt = ts.stats['mu'].astype(np.float64)
            gt_cache = load_or_build_gt_cache(ts.layout, cases, ts.wc, mesh=mesh_hr, mu=mu_gt)
            sweep_res = _run_sweep([entry], ts, knn_map, cases, gt_cache,
                                   args.batch_size, idw_knn)
            model_key = ts.display_name(entry)
            if merged_sweep is None:
                merged_sweep = dict(sweep_res)  # HR + baseline du 1er pair, gardes tels quels
                merged_cases = cases
            else:
                merged_sweep[model_key] = sweep_res[model_key]

    ts_label = f"Comparaison appariée ({', '.join(e.name for e in entries)})"
    if merged_ref:
        merged_ref['_ts_label'] = ts_label
        _plot_ref(merged_ref, canonical_ts, out)
    if merged_sweep:
        merged_sweep['_ts_label'] = ts_label
        _plot_sweep(merged_sweep, merged_cases, canonical_ts, out)
        _save_summary_csv(merged_ref, merged_sweep, out / 'summary.csv')

    # Figure de sensibilité additionnelle, une courbe ou barre par modèle et jamais
    # la baseline. Pas de tableau ici : summary_table.png ci-dessus couvre déjà ce rôle.
    if merged_ref or merged_sweep:
        methods, col_labels, _rows, raw = _summary_methods_rows(merged_ref or {}, merged_sweep)
        model_names = methods[-len(entries):]
        model_raw = raw[-len(entries):]
        if model_raw:
            _plot_paired_sensitivity(col_labels, model_raw, model_names, out)

    elapsed = time.perf_counter() - t0
    print(f"\nTerminé en {elapsed:.1f}s (mode apparié fusionné), résultats dans {out}/")


def _plot_paired_sensitivity(col_labels: list, raw: list, names: list, out_dir: Path) -> None:
    """Courbes métrique vs N_wall quand un N numérique est extrait des noms de modèles
    (convention 'N<N>' ou 'Nw<N>'), sinon barres groupées sur un axe catégoriel. Même
    dégradé RdYlGn_r que summary_table.png."""
    import re
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, LogNorm

    metric_labels = col_labels[1:]  # sans 'Méthode'
    raw = np.asarray(raw, dtype=float)  # (n_models, n_metrics), memes colonnes que metric_labels
    # 'Temps/cas' n'a pas sa place dans un panel de qualité, exclu en amont pour que
    # la grille de subplots n'ait pas de case vide.
    metric_idx = [i for i, lbl in enumerate(metric_labels) if lbl != 'Temps/cas']
    metric_labels = [metric_labels[i] for i in metric_idx]
    n_metrics = len(metric_labels)
    if n_metrics == 0:
        return

    x_num = []
    for nm in names:
        m = re.search(r'Nw?(\d+)', nm)
        x_num.append(int(m.group(1)) if m else None)
    has_numeric_x = all(x is not None for x in x_num) and len(set(x_num)) > 1

    cmap = plt.get_cmap('RdYlGn_r')
    ncols = min(n_metrics, 4)
    nrows = -(-n_metrics // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), dpi=140, squeeze=False)
    axes_flat = axes.flatten()
    for k, (label, col_i) in enumerate(zip(metric_labels, metric_idx)):
        axp = axes_flat[k]
        col = raw[:, col_i]
        vmin, vmax = float(np.nanmin(col)), float(np.nanmax(col))
        norm = ((LogNorm(vmin=vmin, vmax=vmax) if (vmin > 0 and vmax / vmin > 5)
                 else Normalize(vmin=vmin, vmax=vmax)) if vmax > vmin else None)
        colors = [cmap(norm(v)) if norm is not None else '#2c3e50' for v in col]
        if has_numeric_x:
            order = np.argsort(x_num)
            xs = np.array(x_num)[order]
            ys = col[order]
            axp.plot(xs, ys, '-', color='#c9c9c9', zorder=1)
            axp.scatter(xs, ys, c=[colors[i] for i in order], s=90,
                       edgecolors='#2c3e50', linewidths=0.8, zorder=2)
            axp.set_xlabel('N_wall (capteurs)')
        else:
            axp.bar(range(len(names)), col, color=colors, edgecolor='#2c3e50', linewidth=0.8)
            axp.set_xticks(range(len(names))); axp.set_xticklabels(names, rotation=30, ha='right')
        clean_label = label.replace('$', '').replace('\\', '').replace('{', '').replace('}', '')
        axp.set_title(clean_label)
        axp.set_yscale('log')
        axp.grid(True, alpha=0.3)
    for k in range(n_metrics, len(axes_flat)):
        axes_flat[k].axis('off')
    fig.suptitle('Sensibilité par modèle (mode apparié)', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    sens_path = out_dir / 'paired_sensitivity.png'
    plt.savefig(sens_path, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f"  > {sens_path.name}")


def main():
    p = argparse.ArgumentParser(
        description='Évaluation SR-CFD',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            exemples :
            # in-distrib diamond
            %(prog)s --data data/ --models m.pkl --testsets diamond

            # diamond + NACA OOD géométrique
            %(prog)s --data data/ --models m.pkl --testsets diamond naca0012

            # diamond in-distrib + diamond OOD résolution h=0.2 + NACA OOD géo+résolution h=0.2
            %(prog)s --data data/ --models m.pkl --testsets diamond diamond:0.2 naca0012:0.2

            # syntaxe complète : geometry:lr_res:hr_res
            %(prog)s --data data/ --models m.pkl --testsets diamond:0.05:0.025

            # exclure les cas Mach>2 (tf solveur raccourci sur la plupart des géométries, cf. logs/euler/*)
            %(prog)s --data data/ --models m.pkl --testsets naca0012 --mach_max 2.0

            # mode apparie : modele[i] evalue uniquement sur testset[i], pas de produit
            # croise. Ex. plusieurs FAMWall entraines chacun sur son propre N_wall.
            %(prog)s --data data/ --paired \\
                --models m_84.pkl m_42.pkl m_21.pkl m_10.pkl \\
                --testsets diamond diamond_half diamond_quarter diamond_nw10
            """)
    p.add_argument('--data', required=True, help='Racine des données (ex: data/)')
    p.add_argument('--models', nargs='+', required=True, help='Chemins vers les checkpoints (.pkl)')
    p.add_argument('--testsets', nargs='+', required=True,
                   help='Géométries à évaluer, optionnellement suffixées par :lr_res ou '
                        ':lr_res:hr_res  (ex: diamond:0.2  naca0012:0.2:0.025).')
    p.add_argument('--out_dir', default='results/eval/')
    p.add_argument('--out_dirs', nargs='+', default=None,
                   help='Si plusieurs testsets, permet de spécifier un répertoire de sortie par testset '
                        '(doit avoir autant d\'entrées que --testsets).')
    p.add_argument('--lr_res', type=float, default=0.1, help='Résolution LR par défaut (défaut: 0.1)')
    p.add_argument('--hr_res', type=float, default=0.025, help='Résolution HR par défaut (défaut: 0.025)')
    p.add_argument('--ref_cases', nargs='+', default=None, metavar='MACH:AOA',
                   help="Cas de reference epingles (figures), ex: --ref_cases 0.8:3 1.5:0. "
                        "Par defaut : les cas de utils/refs.py propres a la geometrie.")
    p.add_argument('--mach_max', type=float, default=None,
                   help="Exclut du sweep (et des cas de référence) tout cas Mach > valeur")
    p.add_argument('--full_eval', action='store_true', help='Sweep complet du test set (timing, distributions, aéro)')
    p.add_argument('--n_eval', type=int, default=0, help='Limiter le nombre de cas du sweep (0 = tout)')
    p.add_argument('--batch_size', type=int, default=_BATCH_SIZE)
    p.add_argument('--n_steps', type=int, default=None, help='Override n_steps ODE (défaut: valeur du checkpoint)')
    p.add_argument('--n_samples', type=int, default=None, help='Override n_samples ensemble (défaut: valeur du checkpoint)')
    p.add_argument('--cfg_scale', type=float, default=None, help='Override cfg_scale guidance Mach/AoA (défaut: valeur du checkpoint)')
    p.add_argument('--geom_cfg_scale', type=float, default=None,
                   help="Override geom_cfg_scale (guidance CFG sur la geometrie, FAM/SIAM ; "
                        "1.0 = desactive, defaut: valeur du checkpoint)")
    p.add_argument('--n_steps_sweep', type=int, nargs='+', default=None,
                   help="Balaie plusieurs n_steps pour --models[0] : toutes les valeurs sont "
                        "évaluées ensemble et apparaissent sur les mêmes graphes. Les modèles "
                        "--models[1:] sont ajoutés tels quels, une seule fois, hors sweep.")
    p.add_argument('--n_samples_sweep', type=int, nargs='+', default=None,
                   help='Balaie plusieurs n_samples pour --models[0] (cf. --n_steps_sweep), '
                        'combiné en produit cartésien avec --n_steps_sweep si les deux sont donnés.')
    p.add_argument('--geom_cfg_scale_sweep', type=float, nargs='+', default=None,
                   help='Balaie plusieurs geom_cfg_scale pour --models[0] (cf. --n_steps_sweep), '
                        'combiné en produit cartésien avec --n_steps_sweep/--n_samples_sweep si donnés.')
    p.add_argument('--cfg_scale_sweep', type=float, nargs='+', default=None,
                   help='Balaie plusieurs cfg_scale (guidance Mach/AoA) pour --models[0] '
                        '(cf. --n_steps_sweep), combiné en produit cartésien avec les autres sweeps si donnés.')
    p.add_argument('--paired', action='store_true',
                   help="Chaque --models[i] evalue uniquement sur --testsets[i] (appariement "
                        "1:1 au lieu du produit croise habituel), pour comparer des modeles "
                        "ayant chacun leur propre distribution d'entree (ex. FAMWall entraines "
                        "a des N_wall differents). --models et --testsets doivent avoir la "
                        "meme longueur.")
    args = p.parse_args()

    data_root = Path(args.data)
    out_dir = Path(args.out_dir)

    ref_specs = None
    if args.ref_cases:
        ref_specs = []
        for spec in args.ref_cases:
            mach, aoa = (float(v) for v in spec.split(':'))
            ref_specs.append((mach, aoa, f"M{mach:.2f}".replace('.', '') + f"_AoA{aoa:+g}"))
        print(f"  Cas de reference surcharges : {[r[2] for r in ref_specs]}")

    print("\n── Construction des TestSets ───────────────────────────────")
    testsets: list[TestSet] = []
    for spec in args.testsets:
        layout = _parse_testset_spec(spec, data_root, args.lr_res, args.hr_res)
        ts = TestSet.from_dir(data_root, layout.geometry, layout.lr_res, layout.hr_res,
                              mach_max=args.mach_max, ref_specs=ref_specs)
        testsets.append(ts)

    primary_layout = testsets[0].layout

    if args.paired:
        if len(args.models) != len(testsets):
            p.error("--paired requiert autant de --models que de --testsets "
                    f"({len(args.models)} vs {len(testsets)})")
        return _run_paired(args, testsets, out_dir)

    print("\n── Chargement des modèles ──────────────────────────────────")
    model_entries = []
    if args.n_steps_sweep or args.n_samples_sweep or args.geom_cfg_scale_sweep or args.cfg_scale_sweep:
        # Le sweep s'applique au premier modele de --models. Les suivants (ex. un DAM de
        # reference) sont ajoutes tels quels, une seule fois, comme au chemin sans sweep.
        base_entry = load_model(Path(args.models[0]), primary_layout)
        graphdef, state = nnx.split(base_entry.model)
        steps_vals = args.n_steps_sweep or [args.n_steps]
        samples_vals = args.n_samples_sweep or [args.n_samples]
        geom_scale_vals = args.geom_cfg_scale_sweep or [args.geom_cfg_scale]
        cfg_scale_vals = args.cfg_scale_sweep or [args.cfg_scale]
        for n_steps in steps_vals:
            for n_samples in samples_vals:
                for geom_cfg_scale in geom_scale_vals:
                    for cfg_scale in cfg_scale_vals:
                        # Instance indépendante (mêmes poids) : chaque config a son propre
                        # n_steps/n_samples/geom_cfg_scale/cfg_scale
                        model_i = nnx.merge(graphdef, state)
                        tag = ''
                        if n_steps is not None and hasattr(model_i, 'n_steps'):
                            model_i.n_steps = n_steps
                            tag += f'_n{n_steps}'
                        if n_samples is not None and hasattr(model_i, 'n_samples'):
                            model_i.n_samples = n_samples
                            tag += f'_s{n_samples}'
                        if cfg_scale is not None and hasattr(model_i, 'cfg_scale'):
                            model_i.cfg_scale = cfg_scale
                            tag += f'_c{cfg_scale:g}'
                        if geom_cfg_scale is not None and hasattr(model_i, 'geom_cfg_scale'):
                            model_i.geom_cfg_scale = geom_cfg_scale
                            tag += f'_g{geom_cfg_scale:g}'
                        entry = ModelEntry(name=f'{base_entry.name}{tag}', kind=base_entry.kind,
                                           model=model_i, knn=base_entry.knn,
                                           cfg=base_entry.cfg, layout=base_entry.layout)
                        print(f"  {entry.name} [{entry.kind}]  OK")
                        model_entries.append(entry)
        for ckpt in args.models[1:]:
            entry = load_model(Path(ckpt), primary_layout)
            if args.n_steps is not None and hasattr(entry.model, 'n_steps'):
                entry.model.n_steps = args.n_steps
            if args.n_samples is not None and hasattr(entry.model, 'n_samples'):
                entry.model.n_samples = args.n_samples
            if args.cfg_scale is not None and hasattr(entry.model, 'cfg_scale'):
                entry.model.cfg_scale = args.cfg_scale
            if args.geom_cfg_scale is not None and hasattr(entry.model, 'geom_cfg_scale'):
                entry.model.geom_cfg_scale = args.geom_cfg_scale
            print(f"  {entry.name} [{entry.kind}]  OK  (référence fixe, hors sweep)")
            model_entries.append(entry)
    else:
        for ckpt in args.models:
            entry = load_model(Path(ckpt), primary_layout)
            if args.n_steps is not None and hasattr(entry.model, 'n_steps'):
                entry.model.n_steps = args.n_steps
            if args.n_samples is not None and hasattr(entry.model, 'n_samples'):
                entry.model.n_samples = args.n_samples
            if args.cfg_scale is not None and hasattr(entry.model, 'cfg_scale'):
                entry.model.cfg_scale = args.cfg_scale
            if args.geom_cfg_scale is not None and hasattr(entry.model, 'geom_cfg_scale'):
                entry.model.geom_cfg_scale = args.geom_cfg_scale
            print(f"  {entry.name} [{entry.kind}]  OK")
            model_entries.append(entry)

    n_eval = args.n_eval if args.full_eval else 0
    t0 = time.perf_counter()

    if args.out_dirs is not None:
        if len(args.out_dirs) != len(testsets):
            p.error('--out_dirs doit avoir autant d\'entrées que --testsets '
                    f'({len(args.out_dirs)} vs {len(testsets)})')
        for ts, od in zip(testsets, args.out_dirs):
            evaluate(model_entries, ts, Path(od), n_eval=n_eval, batch_size=args.batch_size)
        dirs_str = ', '.join(args.out_dirs)
    else:
        all_results = {}
        for ts in testsets:
            all_results[ts.tag] = evaluate(model_entries, ts, out_dir, n_eval=n_eval, batch_size=args.batch_size)
        plot_combined(all_results, out_dir)
        dirs_str = str(out_dir)

    elapsed = time.perf_counter() - t0
    print(f"\nTerminé en {elapsed:.1f}s, résultats dans {dirs_str}/")


if __name__ == '__main__':
    main()
