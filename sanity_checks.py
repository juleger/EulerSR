"""
Script de vérification de cohérence des données (plots de distribution, erreurs de référence, etc.)
Usage :
    python sanity_checks.py --data data/ --geometry diamond

Le dataset doit avoir des cas de référence pertinent pour illustrer les données
Plots :
-  distribution des cas dans le dataset (train/val/test)
-  champs de référence (HR/LR) pour les cas de référence
-  erreurs de référence (HR vs LR) pour les cas de référence
-  gradient de pression pour les cas de référence
-  profils de Cp/Mach_wall pour les cas de référence
-  distribution des paramètres (Mach, AoA) dans le dataset
-  distribution des grandeurs aérodynamiques (Cl, Cd) entre LR et HR
-  balance du dataset Cl et Cd (train/val/test)
"""


import sys
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT / 'euler') not in sys.path:
    sys.path.append(str(REPO_ROOT / 'euler'))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import euler.jax_fvm.src.mesh  # noqa: F401
import numpy as np

from utils.refs import REFERENCE_CASES, NACA_REFERENCE_CASES, find_ref_file
from utils.layout import DataLayout
from utils.viz.sanity import (plot_dataset_distribution, plot_reference_fields,
    plot_primitive_fields, plot_reference_errors, plot_pressure_gradient, plot_cp_profiles,
    plot_param_distributions, plot_aero_distributions, plot_dataset_balance,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/', help='Racine des données (ex: data/)')
    parser.add_argument('--geometry', default='diamond', help='Géométrie (ex: diamond, naca0012)')
    parser.add_argument('--lr_res', type=float, default=0.1)
    parser.add_argument('--hr_res', type=float, default=0.025)
    parser.add_argument('--out_dir', default=None, help='Répertoire de sortie (défaut: results/stage0/<geom>)')
    args = parser.parse_args()

    layout = DataLayout.from_root(
        Path(args.data).resolve(), args.geometry, args.lr_res, args.hr_res)

    out_dir = Path(args.out_dir) if args.out_dir else              Path('results/stage0') / args.geometry
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_hr = np.load(layout.mesh_path(args.hr_res), allow_pickle=True).item()
    mesh_lr = np.load(layout.mesh_path(args.lr_res), allow_pickle=True).item()

    ref_pool = NACA_REFERENCE_CASES if 'naca' in args.geometry.lower() else REFERENCE_CASES

    all_splits = {}
    for split in ('test', 'val', 'train'):
        d = layout.proc_dir() / split
        all_splits[split] = sorted(d.glob('aoa*.npz')) if d.exists() else []

    ref_paths = []
    for mach_t, aoa_t, label in ref_pool:
        found = None
        for split in ('test', 'val', 'train'):
            p = find_ref_file(all_splits[split], mach_t, aoa_t)
            if p:
                found = p
                break
        if found is None:
            print(f"  Warning: aucun fichier trouvé pour M={mach_t} AoA={aoa_t}")
        else:
            ref_paths.append(found)
            print(f"  [{label}] -> {Path(found).stem}")

    plot_dataset_distribution(layout, out_dir)
    plot_reference_fields(ref_paths, mesh_hr, mesh_lr, out_dir)
    plot_primitive_fields(ref_paths, mesh_hr, out_dir)
    plot_reference_errors(ref_paths, mesh_hr, mesh_lr, out_dir)
    plot_pressure_gradient(ref_paths, mesh_hr, out_dir)
    plot_cp_profiles(ref_paths, mesh_hr, mesh_lr, out_dir)
    plot_param_distributions(layout, out_dir)
    plot_aero_distributions(layout, mesh_hr, mesh_lr, out_dir)
    plot_dataset_balance(layout, mesh_lr, out_dir)

    print(f"\nDone — plots sauvegardés dans {out_dir}/")


if __name__ == '__main__':
    main()
