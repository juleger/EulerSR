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

from utils.refs import REFERENCE_CASES, find_ref_file
from utils.viz.sanity import (
    plot_dataset_distribution,
    plot_reference_cases,
    plot_interp_error_panels,
    plot_interp_errors,
)

_MESH_HR = 'mesh_0.025.npy'
_MESH_LR = 'mesh_0.1.npy'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/diamond/')
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    train_files = sorted((data_dir / 'processed' / 'train').glob('aoa*.npz'))
    out_dir = Path('results/stage0')
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_hr = np.load(data_dir / _MESH_HR, allow_pickle=True).item()
    mesh_lr = np.load(data_dir / _MESH_LR, allow_pickle=True).item()

    ref_paths = []
    for mach_t, aoa_t, label in REFERENCE_CASES:
        p = find_ref_file(train_files, mach_t, aoa_t)
        if p is None:
            print(f"  Warning: no file found for M={mach_t} AoA={aoa_t}")
        else:
            ref_paths.append(p)
            print(f"  Ref case [{label}]: {Path(p).stem}")

    print("\nPlot 1 — distribution dataset...")
    plot_dataset_distribution(data_dir, out_dir)

    print("\nPlot 2 — cas de reference HR / LR / IDW...")
    plot_reference_cases(ref_paths, mesh_hr, mesh_lr, out_dir)

    print("\nPlot 3 — panels erreur IDW (champs)...")
    plot_interp_error_panels(ref_paths, mesh_hr, out_dir)

    print("\nPlot 4 — erreur L2 relative IDW...")
    plot_interp_errors(ref_paths, out_dir)

    print(f"\nDone — plots sauvegardes dans {out_dir}/")


if __name__ == '__main__':
    main()
