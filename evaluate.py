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
                   help='Balaie plusieurs n_steps pour UN SEUL modèle (--models à 1 checkpoint) : '
                        'toutes les valeurs sont évaluées ensemble et apparaissent sur les mêmes graphes.')
    p.add_argument('--n_samples_sweep', type=int, nargs='+', default=None,
                   help='Balaie plusieurs n_samples pour UN SEUL modèle (--models à 1 checkpoint), '
                        'combiné en produit cartésien avec --n_steps_sweep si les deux sont donnés.')
    p.add_argument('--geom_cfg_scale_sweep', type=float, nargs='+', default=None,
                   help='Balaie plusieurs geom_cfg_scale pour UN SEUL modèle (--models à 1 checkpoint), '
                        'combiné en produit cartésien avec --n_steps_sweep/--n_samples_sweep si donnés.')
    p.add_argument('--cfg_scale_sweep', type=float, nargs='+', default=None,
                   help='Balaie plusieurs cfg_scale (guidance Mach/AoA) pour UN SEUL modèle '
                        '(--models à 1 checkpoint), combiné en produit cartésien avec les autres sweeps si donnés.')
    args = p.parse_args()

    data_root = Path(args.data)
    out_dir = Path(args.out_dir)

    print("\n── Construction des TestSets ───────────────────────────────")
    testsets: list[TestSet] = []
    for spec in args.testsets:
        layout = _parse_testset_spec(spec, data_root, args.lr_res, args.hr_res)
        ts = TestSet.from_dir(data_root, layout.geometry, layout.lr_res, layout.hr_res,
                              mach_max=args.mach_max)
        testsets.append(ts)

    primary_layout = testsets[0].layout

    print("\n── Chargement des modèles ──────────────────────────────────")
    model_entries = []
    if args.n_steps_sweep or args.n_samples_sweep or args.geom_cfg_scale_sweep or args.cfg_scale_sweep:
        if len(args.models) != 1:
            p.error('--n_steps_sweep/--n_samples_sweep/--geom_cfg_scale_sweep/--cfg_scale_sweep '
                    'requiert exactement un modèle dans --models')
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
