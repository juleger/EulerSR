"""Prétraitement FAMWall : observations de bord, en dehors du store DAM/FAM.

WallSRDataset (preprocessing/dataset.py) doit joindre les deux stores par nom
de fichier à la lecture : hr_node_pos/hr_primitives/hr_grad_p depuis
{geometry}_hr (existant), wall_pos/wall_normal/wall_s/wall_value depuis
{geometry}_wall (ce script), aucune duplication du store HR (3 Go pour
diamond), juste quelques Ko de plus par cas.
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / 'euler') not in sys.path:
    sys.path.append(str(REPO_ROOT / 'euler'))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import euler.jax_fvm.src.mesh  # noqa: F401

from preprocessing.wall_obs import build_wall_layout, extract_wall_values
from utils.layout import DataLayout


def wall_proc_dir(layout: DataLayout) -> Path:
    """processed/{geometry}_wall/  jamais le même chemin que layout.hr_proc_dir."""
    return layout.root / 'processed' / f'{layout.geometry}_wall'


def build_wall_store(layout: DataLayout, n_wall: int, strategy: str = 'curvature',
                     force: bool = False) -> None:
    """Lit data/processed/{geometry}_hr/{split}/*.npz (lecture seule), écrit les
    observations de bord dans processed/{geometry}_wall/{split}/*.npz.
    """
    hr_dir = layout.hr_proc_dir
    if not hr_dir.exists():
        raise FileNotFoundError(
            f"{hr_dir} introuvable -- lance d'abord preprocessing/preprocess.py "
            "pour cette géométrie (store HR legacy, prérequis en lecture seule).")

    mesh_hr = np.load(layout.mesh_path(layout.hr_res), allow_pickle=True).item()
    wall_layout = build_wall_layout(mesh_hr, n_wall, strategy=strategy)
    print(f"  Layout de bord [{layout.geometry}] : n_wall={wall_layout['n_wall']} "
          f"(strategy={strategy})")

    out_root = wall_proc_dir(layout)
    total_written, total_skipped = 0, 0
    for split in ('train', 'val', 'test'):
        src_dir = hr_dir / split
        if not src_dir.exists():
            continue
        out_dir = out_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        n_written, n_skipped = 0, 0
        for src in sorted(src_dir.glob('aoa*.npz')):
            out = out_dir / src.name
            if out.exists() and not force:
                n_skipped += 1
                continue
            hr = np.load(src)
            wall_value = extract_wall_values(wall_layout, hr['hr_primitives'])
            np.savez(out, wall_pos=wall_layout['pos'], wall_normal=wall_layout['normal'],
                     wall_s=wall_layout['s'], wall_value=wall_value)
            n_written += 1
        print(f"  [{split}] {n_written} écrits, {n_skipped} déjà présents (sautés)")
        total_written += n_written
        total_skipped += n_skipped
    print(f"  Total : {total_written} écrits, {total_skipped} sautés -> {out_root}")


def main():
    parser = argparse.ArgumentParser(
        description="Prétraitement FAMWall : observations de bord (lecture seule du store HR legacy)")
    parser.add_argument('--data', default='data/', help='Racine des données (ex: data/)')
    parser.add_argument('--geometry', default='diamond', help='Géométrie (ex: diamond, naca0012)')
    parser.add_argument('--hr_res', type=float, default=0.025)
    parser.add_argument('--n_wall', type=int, required=True,
                        help="Nombre de points de bord observés à extraire.")
    parser.add_argument('--wall_strategy', default='curvature', choices=['curvature', 'uniform'])
    parser.add_argument('--force', action='store_true',
                        help="Régénère les fichiers wall_* déjà écrits (n'affecte jamais le store HR).")
    args = parser.parse_args()

    # lr_res non pertinent ici (FAMWall n'utilise aucun champ LR volumique)
    # DataLayout l'exige positionnellement, valeur arbitraire sans effet.
    layout = DataLayout.from_root(Path(args.data).resolve(), args.geometry, 0.1, args.hr_res)
    print(f"\nPrétraitement FAMWall  {layout.root}  géométrie={layout.geometry}  "
          f"(lecture seule de {layout.hr_proc_dir})")
    build_wall_store(layout, args.n_wall, strategy=args.wall_strategy, force=args.force)
    print("\nTerminé")


if __name__ == '__main__':
    main()
