#!/usr/bin/env python3
"""Worker subprocess pour eval.live_fvm.solve_case/solve_convergence, isolé
dans son propre process : les imports non packagés de euler/main.py
('from utils import ...') entrent en collision avec utils/ du repo, et
JAX_ENABLE_X64 doit être true ici mais false pour le reste de l'outil.

Par défaut, imprime juste le temps solveur. --emit-summary ajoute le résumé
de run en JSON. --init-field démarre en warm start depuis un champ fourni au
lieu de l'IC freestream. --probe-only ne calcule que le résidu d'un pas.
--forces-only ne calcule que C_D/C_L d'un champ fourni, sans évolution.
--stationarity-patience pilote l'hystérésis du critère résidu, et
--target-cd/--target-cl/--cd-tol/--cl-tol le critère ingénieur suivi en
parallèle (cf. euler/main.py::run).
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EULER_DIR = REPO_ROOT / 'euler'
# euler/ AVANT REPO_ROOT : sinon 'from utils import ...' dans euler/main.py
# résolvent vers utils/ du repo au lieu de euler/utils.py.
for _p in (REPO_ROOT, EULER_DIR):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True)
    ap.add_argument('--mesh-path', required=True)
    ap.add_argument('--mach', type=float, required=True)
    ap.add_argument('--aoa', type=float, required=True)
    ap.add_argument('--tf', type=float, required=True)
    ap.add_argument('--stationarity-threshold', type=float, required=True)
    ap.add_argument('--scratch-dir', required=True)
    ap.add_argument('--stationarity-check-every', type=int, default=None,
                     help='Override de stationarity_check_every (défaut config.toml : 10000)')
    ap.add_argument('--stationarity-patience', type=int, default=1,
                     help="Nb de checks consécutifs sous le seuil avant de déclarer la stationnarité "
                          "(hystérésis anti faux-positif ; 1 = comportement historique)")
    ap.add_argument('--target-cd', type=float, default=None,
                     help="Active le critère ingénieur : cible C_D (typiquement celle du champ HR)")
    ap.add_argument('--target-cl', type=float, default=None,
                     help="Active le critère ingénieur : cible C_L (typiquement celle du champ HR)")
    ap.add_argument('--cd-tol', type=float, default=0.001,
                     help="Tolérance RELATIVE sur C_D (fraction de |target-cd|, défaut 0.1%%)")
    ap.add_argument('--cl-tol', type=float, default=0.001,
                     help="Tolérance RELATIVE sur C_L (fraction de |target-cl|, défaut 0.1%%)")
    ap.add_argument('--engineering-patience', type=int, default=None,
                     help="Patience du critère ingénieur (défaut : même valeur que --stationarity-patience)")
    ap.add_argument('--init-field', default=None,
                     help='Chemin vers un .npy (N,4) de primitives pour démarrer en warm-start')
    ap.add_argument('--emit-summary', action='store_true',
                     help="Imprime le résumé de run en JSON ('FVM_SUMMARY_JSON=...')")
    ap.add_argument('--probe-only', action='store_true',
                     help="Avec --init-field : résidu d'UN pas seulement ('FVM_PROBE_JSON=...')")
    ap.add_argument('--forces-only', action='store_true',
                     help="Avec --init-field : C_D/C_L de ce champ seul, sans évolution ('FVM_FORCES_JSON=...')")
    args = ap.parse_args()

    from euler import main as euler_main
    from euler.config import load_default_config, load_mesh

    cfg = load_default_config()
    cfg.update(case=args.case, Mach=args.mach, aoa=args.aoa,
               mesh_path=args.mesh_path, tf=args.tf,
               stationarity_threshold=args.stationarity_threshold,
               verbose=False)
    if args.stationarity_check_every is not None:
        cfg['stationarity_check_every'] = args.stationarity_check_every
    cfg['stationarity_patience'] = max(1, int(args.stationarity_patience))
    if args.target_cd is not None and args.target_cl is not None:
        cfg['engineering_target_cd'] = float(args.target_cd)
        cfg['engineering_target_cl'] = float(args.target_cl)
        cfg['engineering_cd_tol'] = float(args.cd_tol)
        cfg['engineering_cl_tol'] = float(args.cl_tol)
        cfg['engineering_patience'] = (int(args.engineering_patience)
                                        if args.engineering_patience is not None
                                        else cfg['stationarity_patience'])
    # Aucune écriture de champ/figure sous data/raw/ : seul le résumé JSON
    # (scratch uniquement) est optionnellement exporté, via --emit-summary.
    cfg['export'] = dict(results=False, figures=[], graph=False,
                          summary=bool(args.emit_summary), bundle=False, n_snaps=1)

    mesh = load_mesh(cfg)

    scratch = Path(args.scratch_dir)
    out_dirs = {'fig': scratch / 'fig', 'res': scratch / 'res'}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    W, inlet = euler_main.initialize(mesh, cfg)
    if args.init_field:
        import numpy as np
        prim_init = np.load(args.init_field)
        if prim_init.shape != (mesh.area.shape[0], 4):
            raise ValueError(
                f"--init-field a la forme {prim_init.shape}, attendu "
                f"({mesh.area.shape[0]}, 4) pour ce maillage.")
        W = euler_main.helper.getConserved(
            euler_main.jnp.asarray(prim_init), gamma=cfg['gamma'], M=1.0)

    if args.forces_only:
        if not args.init_field:
            raise SystemExit('--forces-only nécessite --init-field.')
        rho_inf, p_inf, gamma = cfg['rho_inf'], cfg['p_inf'], cfg['gamma']
        U_inf = cfg['Mach'] * float((gamma * p_inf / rho_inf) ** 0.5)
        L_ref = mesh.metadata.get('obstacle_length')
        if L_ref is None:
            raise SystemExit(f"mesh.metadata ne contient pas 'obstacle_length' (case='{args.case}').")
        cd = float(euler_main.helper.get_drag_coefficient(W, mesh, rho_inf=rho_inf, U_inf=U_inf, L_ref=L_ref))
        cl = float(euler_main.helper.get_lift_coefficient(W, mesh, rho_inf=rho_inf, U_inf=U_inf, L_ref=L_ref))
        print('FVM_FORCES_JSON=' + json.dumps({'cd': cd, 'cl': cl}))
        return 0

    if args.probe_only:
        if not args.init_field:
            raise SystemExit('--probe-only nécessite --init-field.')
        # Même clamp CFL transsonique que euler_main.run(), pour que dt (et donc
        # le résidu) soit calculé exactement comme il le serait dans un vrai run.
        if 0.6 < cfg['Mach'] < 1.1 and cfg['CFL'] > 0.45:
            cfg['CFL'] = 0.45
        dt = euler_main.helper.get_dt(W, mesh, CFL=cfg['CFL'], gamma=cfg['gamma'], M=1.0)
        scheme = str(cfg['time_scheme']).upper()
        fn = {'EE': euler_main.euler.time_step_Euler, 'RK2': euler_main.euler.time_step_RK2,
              'RK4': euler_main.euler.time_step_RK4, 'SRK2': euler_main.euler.time_step_RK2_SSP,
              'SSP_RK2': euler_main.euler.time_step_RK2_SSP}[scheme]
        kw = dict(gamma=cfg['gamma'], M=1.0, reconstruction=cfg['reconstruction'],
                   flux=cfg['flux'], value=inlet, entropy=False)
        W_next = fn(W, mesh, dt, **kw)
        euler_main.jax.block_until_ready(W_next)
        residual = float(euler_main.jnp.linalg.norm(W_next - W)
                          / (euler_main.jnp.linalg.norm(W) + 1e-16))
        print('FVM_PROBE_JSON=' + json.dumps({'residual': residual, 'dt': float(dt)}))
        return 0

    if args.emit_summary:
        # Nettoyage préalable : un résumé résiduel d'un run précédent dans le
        # même scratch (label réutilisé) fausserait la lecture ci-dessous.
        for stale in out_dirs['res'].glob('summary_*.json'):
            stale.unlink()

    euler_main.run(W, mesh, inlet, cfg, out_dirs)

    if args.emit_summary:
        matches = sorted(out_dirs['res'].glob('summary_*.json'))
        if not matches:
            raise RuntimeError(f"--emit-summary demandé mais aucun résumé produit (case='{args.case}').")
        summary = json.loads(matches[-1].read_text())
        print('FVM_SUMMARY_JSON=' + json.dumps(summary))
    return 0


if __name__ == '__main__':
    sys.exit(main())
