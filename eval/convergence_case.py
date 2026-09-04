"""Mesure le nombre d'itérations FVM pour reconverger à la stationnarité en
repartant d'un champ reconstruit par un modèle (warm start), comparé au cold
start (IC freestream) et au warm start depuis le champ HR de référence.
S'appuie sur eval.live_fvm.solve_convergence ; les champs viennent déjà
prédits de eval.single_case.run_single_case.

Deux critères sont suivis en parallèle, jamais confondus. Le critère résidu est
seul maître de l'arrêt du solveur, calé sur le résidu propre du champ HR. Le
critère ingénieur compte les itérations avant que Cd/Cl n'entrent durablement
dans une tolérance relative de leurs valeurs HR, sans jamais geler le solveur.
Les deux utilisent une hystérésis (cf. eval.live_fvm._DEFAULT_PATIENCE).

Le speedup est reporté en ratio d'itérations et en ratio de temps de résolution
effectif (solve_time_s, compilation XLA exclue) : le premier est trompeur pour
les warm-starts rapides, dont le poids relatif de la compilation est plus grand.
"""
from __future__ import annotations
import csv
from pathlib import Path

import numpy as np

from eval.live_fvm import (solve_convergence_cached, probe_residual, probe_forces,
                            _DEFAULT_CHECK_EVERY, _DEFAULT_PATIENCE, _DEFAULT_ENGINEERING_TOL)

_COLD_START_LABEL = 'cold-start (freestream)'
_HR_LABEL = 'HR (référence)'


def run_convergence_case(results: dict, layout, mach: float, aoa: float, res_h: float,
                          tf: float | None = None,
                          check_every: int = _DEFAULT_CHECK_EVERY,
                          patience: int = _DEFAULT_PATIENCE,
                          cd_tol: float = _DEFAULT_ENGINEERING_TOL,
                          cl_tol: float = _DEFAULT_ENGINEERING_TOL,
                          include_idw: bool = True,
                          lr_solve_time_s: float | None = None) -> list[dict]:
    """'results' : sortie de eval.single_case.run_single_case sur ce même cas.
    Lance un run FVM par champ initial (cold start, HR, IDW si include_idw,
    puis chaque modèle) et retourne la liste des résumés de convergence.

    Le seuil résidu est toujours le résidu du champ HR (HR = référence,
    convergé par définition) : ValueError si 'results' n'a pas de champ HR.
    Les cibles Cd/Cl du critère ingénieur sont celles du même champ HR.

    include_idw=False exclut la baseline sans réseau du warm-start, quand seule
    la comparaison modèles-vs-solveur-complet importe.
    lr_solve_time_s : temps moyen (dataset) de résolution FVM au maillage LR pour
    cette géométrie, utilisé pour reconstituer le temps de pipeline complet d'un
    modèle (LR + inférence + warm-start HR), cf. 'total_pipeline_time_s'."""
    hr_prim = results['cases'][0].get('hr_prim')
    if hr_prim is None:
        raise ValueError(
            "run_convergence_case requiert un champ HR de référence pour ce cas "
            "(threshold = résidu HR, aucun repli sur un seuil fixe).")

    # HR sert de référence au seuil lui-même : son warm-start reconverge par
    # construction dès le 1er point de contrôle, pas la peine de dépenser un
    # run FVM complet pour le confirmer.
    print("  [seuil = résidu HR] probe résidu du champ HR...")
    probe = probe_residual(layout, mach, aoa, res_h, hr_prim)
    # +0.1% de marge : le résidu du warm-start HR au 1er check et celui du
    # probe ne suivent pas exactement le même chemin de calcul XLA -- sans
    # marge, un écart flottant peut faire manquer le seuil par égalité stricte.
    threshold = probe['residual'] * 1.001
    print(f"    -> seuil résidu = {threshold:.3e} (marge +0.1% incluse), patience={patience}")

    print("  [cibles ingénieur = Cd/Cl HR] probe forces du champ HR...")
    forces = probe_forces(layout, mach, aoa, res_h, hr_prim)
    target_cd, target_cl = forces['cd'], forces['cl']
    print(f"    -> cibles Cd={target_cd:.4f}, Cl={target_cl:.4f} "
          f"(tol ±{cd_tol:.1%}/±{cl_tol:.1%}, patience={patience})")

    # HR n'est jamais relancé : trivialement convergé par construction du seuil
    # (c'est son propre résidu), le résumé est synthétisé directement.
    # infer_time_s : temps d'inférence (médiane des tirs) de la baseline ou du
    # modèle qui a produit ce champ initial, None pour le cold-start.
    to_run = [(_COLD_START_LABEL, None, None)]
    if include_idw and results.get('idw') is not None:
        idw = results['idw']
        to_run.append((idw['name'], idw['prim_preds'][0], float(np.median(idw['time_ms'])) / 1000.0))
    for row in results['rows']:
        to_run.append((row['name'], row['prim_preds'][0], float(np.median(row['time_ms'])) / 1000.0))

    summaries = [{
        'name': _HR_LABEL, 'stopping_step': check_every, 'converged': True,
        'stationarity_rel': probe['residual'], 'threshold_used': threshold,
        'wall_time_s': 0.0, 'compile_time_s': 0.0, 'solve_time_s': 0.0,
        'h': res_h, 'n_cells': None, 'mach': mach, 'aoa': aoa,
        'engineering_stopping_step': check_every, 'engineering_converged': True,
        'engineering_target_cd': target_cd, 'engineering_target_cl': target_cl,
        'not_measured': True,
    }]
    print(f"  [{_HR_LABEL}] non mesuré (trivial par construction du seuil) -> {check_every} itérations")

    for label, prim, infer_time_s in to_run:
        print(f"  [{label}] résolution FVM jusqu'à stationnarité...")
        summary = solve_convergence_cached(
            layout, mach, aoa, res_h, init_prim=prim, tf=tf, threshold=threshold,
            check_every=check_every, patience=patience,
            target_cd=target_cd, target_cl=target_cl, cd_tol=cd_tol, cl_tol=cl_tol,
            label=label)
        summary['name'] = label
        summary['threshold_used'] = threshold
        summary['infer_time_s'] = infer_time_s
        # Pipeline complet : résoudre le LR, inférer le modèle, puis résoudre le HR
        # en warm-start. Défini seulement si on dispose des deux temps amont.
        if lr_solve_time_s is not None and infer_time_s is not None:
            summary['total_pipeline_time_s'] = (
                lr_solve_time_s + infer_time_s + summary.get('solve_time_s', 0.0))
        step = summary.get('stopping_step')
        conv = summary.get('converged')
        eng_step = summary.get('engineering_stopping_step')
        cache_note = '  (cache)' if summary.get('from_cache') else ''
        print(f"    -> résidu : {step} itérations (converged={conv}) | "
              f"ingénieur : {eng_step} itérations{cache_note}")
        summaries.append(summary)

    cold = next((s for s in summaries if s['name'] == _COLD_START_LABEL), None)
    cold_steps = cold.get('stopping_step') if cold else None
    cold_solve_time = cold.get('solve_time_s') if cold else None
    cold_eng_steps = cold.get('engineering_stopping_step') if cold else None
    for summary in summaries:
        step = summary.get('stopping_step')
        solve_time = summary.get('solve_time_s')
        eng_step = summary.get('engineering_stopping_step')
        pipeline_time = summary.get('total_pipeline_time_s')
        summary['speedup_iterations'] = (
            cold_steps / step if (cold_steps and step) else None)
        # Temps cold-start (solveur seul) divisé par LR + inférence + warm-start HR :
        # en partant vraiment de rien, ce pipeline bat-il le solveur HR direct ?
        summary['speedup_pipeline'] = (
            cold_solve_time / pipeline_time if (cold_solve_time and pipeline_time) else None)
        # Ratio de temps de résolution effectif (compilation XLA exclue) plutôt que
        # de nombre d'itérations : la compilation, quasi fixe, pèse proportionnellement
        # plus lourd sur un warm-start rapide et gonfle le ratio d'itérations.
        summary['speedup_solve_time'] = (
            cold_solve_time / solve_time if (cold_solve_time and solve_time) else None)
        summary['speedup_engineering'] = (
            cold_eng_steps / eng_step if (cold_eng_steps and eng_step) else None)

    return summaries


def _fmt(value, spec: str | None = None) -> str:
    if value is None:
        return 'N/A'
    return format(value, spec) if spec else str(value)


def print_convergence_table(summaries: list[dict]) -> None:
    """Deux blocs séparés, le critère résidu puis le critère ingénieur : ils ne
    répondent pas à la même question."""
    if not summaries:
        return
    threshold = summaries[0].get('threshold_used')
    target_cd = summaries[0].get('engineering_target_cd')
    target_cl = summaries[0].get('engineering_target_cl')

    print(f"\n(seuil résidu = {threshold:.3e})" if threshold is not None else "")
    header = (f"{'Champ initial':<28}{'itér. (résidu)':>16}{'converged':>11}"
              f"{'résidu final':>16}{'accél. itér.':>14}{'accél. temps':>14}{'accél. pipeline':>17}")
    print('\n=== Critère résidu (arrêt réel du solveur) ' + '=' * max(0, 20))
    print(header)
    print('-' * len(header))
    for s in summaries:
        note = ' (non mesuré)' if s.get('not_measured') else ''
        print(
            f"{s['name']:<28}"
            f"{_fmt(s.get('stopping_step')):>16}"
            f"{_fmt(s.get('converged')):>11}"
            f"{_fmt(s.get('stationarity_rel'), '.3e'):>16}"
            f"{_fmt(s.get('speedup_iterations'), '.2f'):>14}"
            f"{_fmt(s.get('speedup_solve_time'), '.2f'):>14}"
            f"{_fmt(s.get('speedup_pipeline'), '.2f'):>17}{note}"
        )

    if target_cd is not None:
        print(f"\n=== Critère ingénieur (Cd_HR={target_cd:.4f}, Cl_HR={target_cl:.4f}) " + '=' * 6)
        header2 = f"{'Champ initial':<28}{'itér. (ingé.)':>16}{'converged':>11}{'accél. ingé.':>14}"
        print(header2)
        print('-' * len(header2))
        for s in summaries:
            note = ' (non mesuré)' if s.get('not_measured') else ''
            print(
                f"{s['name']:<28}"
                f"{_fmt(s.get('engineering_stopping_step')):>16}"
                f"{_fmt(s.get('engineering_converged')):>11}"
                f"{_fmt(s.get('speedup_engineering'), '.2f'):>14}{note}"
            )


def save_convergence_csv(summaries: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['name', 'stopping_step', 'converged', 'stationarity_rel', 'threshold_used',
              'speedup_iterations', 'speedup_solve_time', 'speedup_pipeline',
              'engineering_stopping_step', 'engineering_converged', 'speedup_engineering',
              'engineering_target_cd', 'engineering_target_cl',
              'wall_time_s', 'compile_time_s', 'solve_time_s', 'infer_time_s',
              'total_pipeline_time_s',
              'h', 'n_cells', 'mach', 'aoa', 'not_measured', 'from_cache']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for s in summaries:
            writer.writerow(s)
