"""Script principal d'évaluation des modèles SR-CFD sur les test sets

"""
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

from eval.loader import load_model
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
            """)
    p.add_argument('--data', required=True, help='Racine des données (ex: data/)')
    p.add_argument('--models', nargs='+', required=True, help='Chemins vers les checkpoints (.pkl)')
    p.add_argument('--testsets', nargs='+', required=True,
                   help='Géométries à évaluer, optionnellement suffixées par :lr_res ou '
                        ':lr_res:hr_res  (ex: diamond:0.2  naca0012:0.2:0.025).')
    p.add_argument('--out_dir', default='results/eval/')
    p.add_argument('--lr_res', type=float, default=0.1, help='Résolution LR par défaut (défaut: 0.1)')
    p.add_argument('--hr_res', type=float, default=0.025, help='Résolution HR par défaut (défaut: 0.025)')
    p.add_argument('--full_eval', action='store_true', help='Sweep complet du test set (timing, distributions, aéro)')
    p.add_argument('--n_eval', type=int, default=0, help='Limiter le nombre de cas du sweep (0 = tout)')
    p.add_argument('--batch_size', type=int, default=_BATCH_SIZE)
    p.add_argument('--n_steps', type=int, default=None, help='Override n_steps ODE (défaut: valeur du checkpoint)')
    p.add_argument('--n_samples', type=int, default=None, help='Override n_samples ensemble (défaut: valeur du checkpoint)')
    p.add_argument('--cfg_scale', type=float, default=None, help='Override cfg_scale guidance (défaut: valeur du checkpoint)')
    args = p.parse_args()

    data_root = Path(args.data)
    out_dir = Path(args.out_dir)

    print("\n── Construction des TestSets ───────────────────────────────")
    testsets: list[TestSet] = []
    for spec in args.testsets:
        layout = _parse_testset_spec(spec, data_root, args.lr_res, args.hr_res)
        ts = TestSet.from_dir(data_root, layout.geometry, layout.lr_res, layout.hr_res)
        testsets.append(ts)

    primary_layout = testsets[0].layout

    print("\n── Chargement des modèles ──────────────────────────────────")
    model_entries = []
    for ckpt in args.models:
        entry = load_model(Path(ckpt), primary_layout)
        if args.n_steps is not None and hasattr(entry.model, 'n_steps'):
            entry.model.n_steps = args.n_steps
        if args.n_samples is not None and hasattr(entry.model, 'n_samples'):
            entry.model.n_samples = args.n_samples
        if args.cfg_scale is not None and hasattr(entry.model, 'cfg_scale'):
            entry.model.cfg_scale = args.cfg_scale
        print(f"  {entry.name} [{entry.kind}]  OK")
        model_entries.append(entry)

    n_eval = args.n_eval if args.full_eval else 0
    t0 = time.perf_counter()
    all_results = {}
    for ts in testsets:
        all_results[ts.tag] = evaluate(model_entries, ts, out_dir, n_eval=n_eval, batch_size=args.batch_size)

    plot_combined(all_results, out_dir)

    elapsed = time.perf_counter() - t0
    print(f"\nTerminé en {elapsed:.1f}s, résultats dans {out_dir}/")


if __name__ == '__main__':
    main()
