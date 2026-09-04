"""Campagne de convergence FVM en warm-start sur un sous-ensemble du testset :
moyenne du nb d'itérations par modèle (IDW/FAM/DAM/...) + heatmap Mach/AoA des
zones les plus difficiles à reconstruire au sens dynamique (cf. docstring de
eval/convergence_sweep.py pour la méthode et son coût).

Usage :
    uv run python eval_convergence_sweep.py --data data/ --geometry diamond \\
        --models results/checkpoints/dam_xxx/model.pkl results/checkpoints/fam_xxx/model.pkl \\
        --stride_mach 4 --stride_aoa 4 --out_dir results/convergence_sweep/
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
from eval.convergence_sweep import (select_sweep_cases, run_convergence_sweep,
                                    print_sweep_summary, save_sweep_csv,
                                    plot_convergence_heatmaps)


def main():
    p = argparse.ArgumentParser(
        description="Campagne de convergence FVM en warm-start sur le testset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            exemples :
            %(prog)s --data data/ --geometry diamond --models dam.pkl fam.pkl

            %(prog)s --data data/ --geometry diamond --models dam.pkl \\
                --stride_mach 2 --stride_aoa 2 --max_cases 200
            """)
    p.add_argument('--data', required=True, help='Racine des données (ex: data/)')
    p.add_argument('--geometry', required=True)
    p.add_argument('--lr_res', type=float, default=0.1)
    p.add_argument('--hr_res', type=float, default=0.025)

    p.add_argument('--models', nargs='+', required=True, help='Chemins vers les checkpoints (.pkl)')

    p.add_argument('--stride_mach', type=int, default=4,
                   help="Ne garde qu'1 valeur de Mach sur 'stride_mach' du testset")
    p.add_argument('--stride_aoa', type=int, default=4,
                   help="Ne garde qu'1 valeur d'AoA sur 'stride_aoa' du testset")
    p.add_argument('--max_cases', type=int, default=60,
                   help="Garde-fou : erreur si le sous-échantillon dépasse ce nombre de cas "
                        "(0 pour désactiver -- campagne longue, prévois un job slurm)")

    p.add_argument('--tf', type=float, default=None, help='Override tf du solveur FVM')
    p.add_argument('--convergence_check_every', type=int, default=None,
                   help='Override de stationarity_check_every (défaut eval.live_fvm._DEFAULT_CHECK_EVERY)')

    p.add_argument('--out_dir', default='results/convergence_sweep/')
    args = p.parse_args()

    data_root = Path(args.data)

    print("── Construction du TestSet ─────────────────────────────────")
    ts = TestSet.from_dir(data_root, args.geometry, args.lr_res, args.hr_res)

    max_cases = None if args.max_cases == 0 else args.max_cases
    cases = select_sweep_cases(ts.test_cases, stride_mach=args.stride_mach,
                                stride_aoa=args.stride_aoa, max_cases=max_cases)
    print(f"  {len(cases)} cas sélectionnés (stride_mach={args.stride_mach}, "
          f"stride_aoa={args.stride_aoa}) sur {len(ts.test_cases)} au total.")

    print("\n── Chargement des modèles ──────────────────────────────────")
    model_entries = []
    for ckpt in args.models:
        entry = load_model(Path(ckpt), ts.layout)
        print(f"  {entry.name} [{entry.kind}]  OK")
        model_entries.append(entry)

    print("\n── Construction des KNN ────────────────────────────────────")
    knn_map = {}
    for entry in model_entries:
        knn_map[entry.name] = ts.build_knn(entry)
        print(f"  [{entry.name}]  {ts.ood_label(entry)}")
    # Mode bord si TOUS les modèles chargés sont des modèles bord, même règle que
    # eval/runner.py. Change la nature de la baseline sans réseau : LR IDW n'a aucun
    # sens pour un modèle qui n'a jamais accès au champ LR.
    is_wall_eval = bool(model_entries) and all(
        hasattr(e.model, 'wall_encoder') for e in model_entries)
    idw_knn = ts.build_idw_knn(k=6, wall=is_wall_eval)

    print(f"\n── Campagne de convergence ({len(cases)} cas) ────────────────")
    t0 = time.perf_counter()
    rows = run_convergence_sweep(
        ts, model_entries, idw_knn, knn_map, cases, ts.hr_res, tf=args.tf,
        check_every=args.convergence_check_every)
    elapsed = time.perf_counter() - t0

    out_dir = Path(args.out_dir) / ts.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n── Résumé agrégé ────────────────────────────────────────────")
    print_sweep_summary(rows)
    save_sweep_csv(rows, out_dir / 'convergence_sweep.csv')
    print(f"\n  -> {out_dir / 'convergence_sweep.csv'}")

    print("\n── Heatmaps Mach/AoA ────────────────────────────────────────")
    heatmap_path = plot_convergence_heatmaps(rows, out_dir, value_key='stopping_step')
    if heatmap_path:
        print(f"  -> {heatmap_path}")
    speedup_path = plot_convergence_heatmaps(rows, out_dir, value_key='speedup_vs_coldstart')
    if speedup_path:
        print(f"  -> {speedup_path}")

    print(f"\nTerminé en {elapsed:.1f}s ({len(cases)} cas) — résultats dans {out_dir}/")


if __name__ == '__main__':
    main()
