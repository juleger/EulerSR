"""Mesure le nombre d'itérations FVM pour reconverger à la stationnarité en
repartant d'un champ reconstruit par un modèle (warm start), comparé au cold
start (IC freestream) et au warm start depuis le champ HR de référence.
S'appuie sur eval.live_fvm.solve_convergence ; les champs viennent déjà
prédits de eval.single_case.run_single_case.
"""
from __future__ import annotations
import csv
from pathlib import Path

from eval.live_fvm import solve_convergence_cached, probe_residual, _DEFAULT_CHECK_EVERY

_COLD_START_LABEL = 'cold-start (freestream)'
_HR_LABEL = 'HR (référence)'


def run_convergence_case(results: dict, layout, mach: float, aoa: float, res_h: float,
                          tf: float | None = None,
                          check_every: int = _DEFAULT_CHECK_EVERY) -> list[dict]:
    """'results' : sortie de eval.single_case.run_single_case sur ce même cas.
    Lance un run FVM par champ initial (cold start, HR, IDW, puis chaque
    modèle) et retourne la liste des résumés de convergence.

    Le seuil est toujours le résidu du champ HR (HR = référence, convergé par
    définition) : ValueError si 'results' n'a pas de champ HR."""
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
    print(f"    -> seuil = résidu HR = {threshold:.3e} (marge +0.1% incluse)")

    # HR n'est jamais relancé : trivialement convergé par construction du seuil
    # (c'est son propre résidu), le résumé est synthétisé directement.
    to_run = [(_COLD_START_LABEL, None)]
    if results.get('idw') is not None:
        to_run.append((results['idw']['name'], results['idw']['prim_preds'][0]))
    for row in results['rows']:
        to_run.append((row['name'], row['prim_preds'][0]))

    summaries = [{
        'name': _HR_LABEL, 'stopping_step': check_every, 'converged': True,
        'stationarity_rel': probe['residual'], 'threshold_used': threshold,
        'wall_time_s': 0.0, 'h': res_h, 'n_cells': None, 'mach': mach, 'aoa': aoa,
        'not_measured': True,
    }]
    print(f"  [{_HR_LABEL}] non mesuré (trivial par construction du seuil) -> {check_every} itérations")

    for label, prim in to_run:
        print(f"  [{label}] résolution FVM jusqu'à stationnarité...")
        summary = solve_convergence_cached(layout, mach, aoa, res_h, init_prim=prim,
                                            tf=tf, threshold=threshold, check_every=check_every,
                                            label=label)
        summary['name'] = label
        summary['threshold_used'] = threshold
        step = summary.get('stopping_step')
        conv = summary.get('converged')
        cache_note = '  (cache)' if summary.get('from_cache') else ''
        print(f"    -> {step} itérations (converged={conv}){cache_note}")
        summaries.append(summary)

    cold_steps = next((s.get('stopping_step') for s in summaries if s['name'] == _COLD_START_LABEL), None)
    for summary in summaries:
        step = summary.get('stopping_step')
        summary['speedup_vs_coldstart'] = (
            cold_steps / step if (cold_steps and step) else None)

    return summaries


def _fmt(value, spec: str | None = None) -> str:
    if value is None:
        return 'N/A'
    return format(value, spec) if spec else str(value)


def print_convergence_table(summaries: list[dict]) -> None:
    threshold = summaries[0].get('threshold_used') if summaries else None
    if threshold is not None:
        print(f"\n(seuil de stationnarité utilisé : {threshold:.3e})")
    header = (f"{'Champ initial':<28}{'itérations':>12}{'converged':>12}"
              f"{'résidu final':>16}{'accél. vs cold':>16}")
    print('\n' + header)
    print('-' * len(header))
    for s in summaries:
        note = ' (non mesuré)' if s.get('not_measured') else ''
        print(
            f"{s['name']:<28}"
            f"{_fmt(s.get('stopping_step')):>12}"
            f"{_fmt(s.get('converged')):>12}"
            f"{_fmt(s.get('stationarity_rel'), '.3e'):>16}"
            f"{_fmt(s.get('speedup_vs_coldstart'), '.2f'):>16}{note}"
        )


def save_convergence_csv(summaries: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['name', 'stopping_step', 'converged', 'stationarity_rel', 'threshold_used',
              'speedup_vs_coldstart', 'wall_time_s', 'h', 'n_cells', 'mach', 'aoa',
              'not_measured', 'from_cache']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for s in summaries:
            writer.writerow(s)
