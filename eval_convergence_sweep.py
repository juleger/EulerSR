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
from eval.live_fvm import _DEFAULT_PATIENCE, _DEFAULT_ENGINEERING_TOL
from eval.convergence_sweep import (select_sweep_cases, run_convergence_sweep,
                                    print_sweep_summary, save_sweep_csv,
                                    save_sweep_summary_csv, plot_convergence_heatmaps)


def _lookup_lr_solve_time(data_root: Path, geometry: str, lr_res: float) -> float | None:
    """Temps moyen (dataset complet, cf. data/fvm_times.json) de résolution FVM au
    maillage LR pour cette géométrie, estimateur du coût 'résoudre le LR' dans le
    temps de pipeline complet. Moyenne globale : fvm_times.json ne donne pas de
    détail par cas Mach/AoA."""
    import json
    path = data_root / 'fvm_times.json'
    if not path.exists():
        print(f"  [WARN] {path} introuvable, temps de pipeline non calculé (LR manquant).")
        return None
    d = json.loads(path.read_text())
    entry = d.get('by_geometry_resolution', {}).get(geometry, {}).get(f'{lr_res:g}')
    if entry is None:
        print(f"  [WARN] pas de temps LR pour {geometry} h={lr_res:g} dans {path}, "
              f"temps de pipeline non calculé.")
        return None
    return float(entry['time_mean_s'])


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
    p.add_argument('--n_steps', type=int, default=None,
                   help='Override n_steps ODE pour les modèles génératifs (FAM/SIAM), '
                        'défaut : valeur du checkpoint.')

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
    p.add_argument('--patience', type=int, default=None,
                   help='Nb de checks consécutifs sous le seuil résidu avant convergence '
                        '(hystérésis anti faux-positif ; défaut eval.live_fvm._DEFAULT_PATIENCE)')
    p.add_argument('--cd_tol', type=float, default=None,
                   help='Tolérance relative du critère ingénieur sur Cd (défaut 1%%)')
    p.add_argument('--cl_tol', type=float, default=None,
                   help='Tolérance relative du critère ingénieur sur Cl (défaut 1%%)')
    p.add_argument('--no_idw', action='store_true',
                   help="Exclut la baseline IDW du warm-start, ne garde que cold-start, "
                        "HR et les modèles.")
    p.add_argument('--no_pipeline_time', action='store_true',
                   help="Désactive le calcul du temps de pipeline complet (LR + inférence "
                        "+ warm-start HR), actif par défaut via data/fvm_times.json.")

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
        if args.n_steps is not None and hasattr(entry.model, 'n_steps'):
            entry.model.n_steps = args.n_steps
            print(f"  {entry.name} [{entry.kind}]  OK  (n_steps={args.n_steps})")
        else:
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

    patience = args.patience if args.patience is not None else _DEFAULT_PATIENCE
    cd_tol = args.cd_tol if args.cd_tol is not None else _DEFAULT_ENGINEERING_TOL
    cl_tol = args.cl_tol if args.cl_tol is not None else _DEFAULT_ENGINEERING_TOL
    include_idw = not args.no_idw

    lr_solve_time_s = None
    if not args.no_pipeline_time:
        lr_solve_time_s = _lookup_lr_solve_time(data_root, args.geometry, args.lr_res)
        if lr_solve_time_s is not None:
            print(f"  temps LR moyen (dataset, h={args.lr_res:g}) = {lr_solve_time_s:.2f}s "
                  f"-- utilisé pour le temps de pipeline complet")

    print(f"\n── Campagne de convergence ({len(cases)} cas, patience={patience}, "
          f"tol_ingénieur=±{cd_tol:.1%}/±{cl_tol:.1%}, idw={'oui' if include_idw else 'non'}) "
          f"────────────────")
    t0 = time.perf_counter()
    rows = run_convergence_sweep(
        ts, model_entries, idw_knn, knn_map, cases, ts.hr_res, tf=args.tf,
        check_every=args.convergence_check_every, patience=patience, cd_tol=cd_tol, cl_tol=cl_tol,
        include_idw=include_idw,
        lr_solve_time_s=lr_solve_time_s)
    elapsed = time.perf_counter() - t0

    out_dir = Path(args.out_dir) / ts.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n── Résumé agrégé ────────────────────────────────────────────")
    print_sweep_summary(rows)
    save_sweep_csv(rows, out_dir / 'convergence_sweep_detail.csv')
    save_sweep_summary_csv(rows, out_dir / 'convergence_sweep_summary.csv')
    print(f"\n  -> {out_dir / 'convergence_sweep_detail.csv'} (détail brut par cas)")
    print(f"  -> {out_dir / 'convergence_sweep_summary.csv'} (synthèse propre, une ligne par modèle)")

    print("\n── Heatmaps Mach/AoA ────────────────────────────────────────")
    # 'stopping_step'/'speedup_iterations' : critère résidu. 'speedup_solve_time' :
    # même critère en temps de résolution, compilation XLA exclue.
    # 'engineering_stopping_step' : critère Cd/Cl vs cible HR.
    value_keys = ['stopping_step', 'speedup_iterations', 'speedup_solve_time',
                  'speedup_pipeline', 'engineering_stopping_step']
    for value_key in value_keys:
        path = plot_convergence_heatmaps(rows, out_dir, value_key=value_key)
        if path:
            print(f"  -> {path}")

    print(f"\nTerminé en {elapsed:.1f}s ({len(cases)} cas) — résultats dans {out_dir}/")


if __name__ == '__main__':
    main()
