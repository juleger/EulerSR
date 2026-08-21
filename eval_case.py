"""Évaluation SR-CFD sur UN cas de test unique : inférence de chaque modèle
sur le champ choisi (géométrie + Mach + AoA), avec temps de résolution FVM
LR/HR mesurés en direct (non biaisés) et images + tableau récapitulatif.

Usage :
    uv run python eval_case.py --data data/ --geometry diamond --mach 1.5 --aoa 3 \\
        --models results/checkpoints/dam_xxx/model.pkl results/checkpoints/fam_xxx/model.pkl \\
        --out_dir results/eval_case/
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

import euler.jax_fvm.src.mesh  # noqa: F401  requis pour unpickle des maillages

from eval.loader import load_model
from eval.testset import TestSet
from eval.single_case import run_single_case
from eval.live_fvm import solve_case_cached
from eval.convergence_case import run_convergence_case, print_convergence_table, save_convergence_csv
from utils.layout import load_sample
from utils.refs import parse_case
from utils.viz.eval import (plot_reference_grid, plot_wall_profiles_eval,
                            plot_cp_profile_eval, plot_single_case_table,
                            save_single_case_csv)


def _select_case(ts: TestSet, args) -> dict:
    """Résout le cas demandé en dict {'path','mach_in','aoa_in','label',...}."""
    if args.case_path:
        p = Path(args.case_path)
        if args.mach is not None and args.aoa is not None:
            mach_in, aoa_in = args.mach, args.aoa
        else:
            mach_in, aoa_in = parse_case(p)
        return {'path': str(p), 'mach_in': mach_in, 'aoa_in': aoa_in, 'label': p.stem}

    pool = list(ts.test_cases) + list(ts.ref_cases)
    if not pool:
        raise SystemExit(f"Aucun cas disponible pour {ts.tag} — vérifie --data/--geometry/--lr_res/--hr_res.")

    if args.case_label:
        for c in pool:
            if c.get('label') == args.case_label:
                return c
        raise SystemExit(f"Label '{args.case_label}' introuvable parmi {[c.get('label') for c in pool]}")

    if args.mach is None or args.aoa is None:
        raise SystemExit("Précise le cas à évaluer : --mach/--aoa, --case_label ou --case_path.")

    best, best_d = None, float('inf')
    for c in pool:
        dd = (c['mach_in'] - args.mach) ** 2 + (c['aoa_in'] - args.aoa) ** 2
        if dd < best_d:
            best_d, best = dd, c
    if args.tol is not None and best_d ** 0.5 > args.tol:
        print(f"  [WARN] cas le plus proche à distance {best_d ** 0.5:.3f} > tol={args.tol} "
              f"(M={best['mach_in']:.2f}, AoA={best['aoa_in']:+.1f})")
    return best


def main():
    p = argparse.ArgumentParser(
        description="Évaluation SR-CFD sur un cas unique (concret, visuel)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            exemples :
            %(prog)s --data data/ --geometry diamond --mach 1.5 --aoa 3 --models m1.pkl m2.pkl

            %(prog)s --data data/ --geometry naca0012 --case_label M080_AoA2 --models m.pkl --no_live_fvm

            %(prog)s --data data/ --geometry diamond --mach 1.5 --aoa 3 --models m1.pkl m2.pkl --convergence
            """)
    p.add_argument('--data', required=True, help='Racine des données (ex: data/)')
    p.add_argument('--geometry', required=True)
    p.add_argument('--lr_res', type=float, default=0.1)
    p.add_argument('--hr_res', type=float, default=0.025)

    p.add_argument('--mach', type=float, default=None)
    p.add_argument('--aoa', type=float, default=None)
    p.add_argument('--tol', type=float, default=None,
                   help='Distance max (Mach,AoA) tolérée avant avertissement sur le cas retenu')
    p.add_argument('--case_label', default=None, help="Label exact d'un cas de référence/test (ex: M09_AoA5)")
    p.add_argument('--case_path', default=None, help='Chemin direct vers un .npz prétraité')

    p.add_argument('--models', nargs='+', required=True, help='Chemins vers les checkpoints (.pkl)')
    p.add_argument('--n_steps', type=int, default=None)
    p.add_argument('--n_samples', type=int, default=None)
    p.add_argument('--cfg_scale', type=float, default=None)
    p.add_argument('--n_repeat', type=int, default=5,
                   help="Nombre d'appels d'inférence répétés par modèle (médiane retenue)")

    p.add_argument('--no_live_fvm', action='store_true',
                   help='Désactive la résolution FVM live LR/HR (pas de temps solveur dans le tableau)')
    p.add_argument('--force_resolve', action='store_true',
                   help='Ignore le cache de temps FVM live et relance la résolution')
    p.add_argument('--tf', type=float, default=None, help='Override tf du solveur FVM live')
    p.add_argument('--stationarity_threshold', type=float, default=1e-8)

    p.add_argument('--convergence', action='store_true',
                   help="Mesure le nb d'itérations FVM pour reconverger en repartant de chaque "
                        "champ reconstruit (warm start), vs cold start et warm start HR. "
                        "Coûteux : un run FVM complet par modèle + IDW + cold-start + HR.")
    p.add_argument('--convergence_check_every', type=int, default=None,
                   help='Override de stationarity_check_every pour --convergence (défaut : eval.live_fvm._DEFAULT_CHECK_EVERY)')

    p.add_argument('--out_dir', default='results/eval_case/')
    args = p.parse_args()

    data_root = Path(args.data)

    print("── Construction du TestSet ─────────────────────────────────")
    ts = TestSet.from_dir(data_root, args.geometry, args.lr_res, args.hr_res)

    case = _select_case(ts, args)
    print(f"  Cas retenu : M={case['mach_in']:.2f}  AoA={case['aoa_in']:+.1f}°  "
          f"({case.get('label', Path(case['path']).stem)})")

    print("\n── Chargement des modèles ──────────────────────────────────")
    model_entries = []
    for ckpt in args.models:
        entry = load_model(Path(ckpt), ts.layout)
        if args.n_steps is not None and hasattr(entry.model, 'n_steps'):
            entry.model.n_steps = args.n_steps
        if args.n_samples is not None and hasattr(entry.model, 'n_samples'):
            entry.model.n_samples = args.n_samples
        if args.cfg_scale is not None and hasattr(entry.model, 'cfg_scale'):
            entry.model.cfg_scale = args.cfg_scale
        print(f"  {entry.name} [{entry.kind}]  OK")
        model_entries.append(entry)

    print("\n── Construction des KNN ────────────────────────────────────")
    knn_map = {}
    for entry in model_entries:
        knn_map[entry.name] = ts.build_knn(entry)
        print(f"  [{entry.name}]  {ts.ood_label(entry)}")
    idw_knn = ts.build_idw_knn(k=6)

    d = load_sample(case['path'], case.get('raw_hr_path'))

    t0 = time.perf_counter()
    results = run_single_case(model_entries, ts, case, d, idw_knn, knn_map,
                              n_repeat=args.n_repeat)
    elapsed = time.perf_counter() - t0

    fvm_live = None
    if not args.no_live_fvm:
        print("\n── Résolution FVM live (LR puis HR, temps non biaisé) ──────")
        lr = solve_case_cached(ts.layout, case['mach_in'], case['aoa_in'], ts.lr_res,
                               tf=args.tf, threshold=args.stationarity_threshold,
                               force=args.force_resolve)
        print(f"  LR h={ts.lr_res:g} : {lr['wall_time_s']:.2f}s"
              f"{'  (cache)' if lr['from_cache'] else ''}")
        hr = solve_case_cached(ts.layout, case['mach_in'], case['aoa_in'], ts.hr_res,
                               tf=args.tf, threshold=args.stationarity_threshold,
                               force=args.force_resolve)
        print(f"  HR h={ts.hr_res:g} : {hr['wall_time_s']:.2f}s"
              f"{'  (cache)' if hr['from_cache'] else ''}")
        fvm_live = {'lr': lr, 'hr': hr}

    out_dir = Path(args.out_dir) / f'{ts.tag}_{case.get("label", "case")}'
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.convergence:
        print("\n── Convergence FVM à partir des champs reconstruits (warm start) ──")
        conv_kwargs = dict(tf=args.tf)
        if args.convergence_check_every is not None:
            conv_kwargs['check_every'] = args.convergence_check_every
        conv_summaries = run_convergence_case(
            results, ts.layout, case['mach_in'], case['aoa_in'], ts.hr_res,
            **conv_kwargs)
        print_convergence_table(conv_summaries)
        save_convergence_csv(conv_summaries, out_dir / 'convergence.csv')
        print(f"  -> {out_dir / 'convergence.csv'}")

    print("\n── Figures ──────────────────────────────────────────────────")
    plot_reference_grid(results, ts.triang_hr, out_dir / 'reference_grid.png')
    plot_wall_profiles_eval(results, ts.wc, out_dir)
    plot_cp_profile_eval(results, ts.wc, out_dir)
    plot_single_case_table(results, out_dir, fvm_live=fvm_live)
    save_single_case_csv(results, out_dir / 'summary.csv')

    print(f"\nTerminé en {elapsed:.1f}s (inférence) — résultats dans {out_dir}/")


if __name__ == '__main__':
    main()
