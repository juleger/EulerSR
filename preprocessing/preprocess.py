import sys
import re
import shutil
import argparse
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / 'euler') not in sys.path:
    sys.path.append(str(REPO_ROOT / 'euler'))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import euler.jax_fvm.src.mesh  # noqa: F401

from preprocessing.graph import build as build_graph
from preprocessing.knn import build as build_knn
from preprocessing.normalize import compute as compute_stats
from utils.refs import _res_tag
from utils.layout import DataLayout


_FNAME_RE = re.compile(r'AOA([+-]?[0-9.]+)_M([0-9.]+)_.*_t([0-9.]+)\.npz$')
_HR_FIELDS = ('node_pos', 'node_area', 'primitives', 'primitives_grad')
_LR_FIELDS = ('node_pos', 'node_area', 'primitives')

def _stratified_split(
    paired: dict,
    ref_cases: list,
    step: int = 10,
    val_offset: int = 5,
) -> dict:
    """
    Split systématique par AoA et Mach trié

    Pour chaque AoA, les cas sont triés par Mach croissant et assignés uniformément à train/val/test selon step=10
    val_offset=5 (step//2) place val à mi-chemin entre deux samples test. Permet d'avoir des sets physiquement représentatifs
    """
    from collections import defaultdict

    by_aoa: dict = defaultdict(list)
    for key in sorted(paired):
        by_aoa[key[0]].append(key)

    test_keys, val_keys, train_keys = [], [], []
    for aoa_s in sorted(by_aoa):
        for i, key in enumerate(sorted(by_aoa[aoa_s], key=lambda k: float(k[1]))):
            r = i % step
            if r == 0:
                test_keys.append(key)
            elif r == val_offset:
                val_keys.append(key)
            else:
                train_keys.append(key)

    # Épinglage des cas de référence dans test
    val_set   = set(map(tuple, val_keys))
    train_set = set(map(tuple, train_keys))
    for mach_t, aoa_t, _ in ref_cases:
        best_key, best_d = None, float('inf')
        for aoa_s, mach_s in paired:
            d = (float(mach_s) - mach_t) ** 2 + (float(aoa_s) - aoa_t) ** 2
            if d < best_d:
                best_d, best_key = d, (aoa_s, mach_s)
        if best_key is None or best_d > 1e-4:
            print(f"  Warning: aucun fichier exact pour ref ({mach_t}, {aoa_t})")
            continue
        if best_key in val_set:
            val_keys.remove(best_key)
            val_set.discard(best_key)
            test_keys.append(best_key)
        elif best_key in train_set:
            train_keys.remove(best_key)
            train_set.discard(best_key)
            test_keys.append(best_key)

    return {'test': test_keys, 'val': val_keys, 'train': train_keys}


def build_processed(layout: DataLayout, hr_res='h0.025', lr_res='h0.1', force=False):
    """Construction du dataset traité à partir des fichiers bruts
    Split déterministe 80/10/10 par AoA et Mach trié
    Les REFERENCE_CASES sont épinglés dans test
    """
    all_files: dict[str, dict] = {}
    for res in (hr_res, lr_res):
        all_files[res] = {}
        for aoa_dir in sorted((layout.raw_dir / res).iterdir()):
            if not aoa_dir.is_dir():
                continue
            for f in sorted(aoa_dir.iterdir()):
                m = _FNAME_RE.search(f.name)
                if not m:
                    continue
                key = (m.group(1), m.group(2))
                t = float(m.group(3))
                all_files[res].setdefault(key, {})[t] = f

    keys = set(all_files[hr_res]) & set(all_files[lr_res])
    paired: dict[tuple, tuple] = {}
    for key in sorted(keys):
        common_ts = set(all_files[hr_res][key]) & set(all_files[lr_res][key])
        if not common_ts:
            continue
        best_t = max(common_ts)
        paired[key] = (all_files[hr_res][key][best_t], all_files[lr_res][key][best_t])

    from utils.refs import REFERENCE_CASES
    splits = _stratified_split(paired, REFERENCE_CASES)

    proc_base = layout.proc_dir()
    for split_name in ('train', 'val', 'test'):
        (proc_base / split_name).mkdir(parents=True, exist_ok=True)

    fname_to_target = {
        f"aoa{float(aoa_s):+.2f}_m{float(mach_s):.2f}.npz": split_name
        for split_name, keys in splits.items()
        for (aoa_s, mach_s) in keys
    }
    for split_name in ('train', 'val', 'test'):
        for existing in list((proc_base / split_name).glob('aoa*.npz')):
            target = fname_to_target.get(existing.name)
            if target is None:
                existing.unlink()
            elif target != split_name:
                dest = proc_base / target / existing.name
                if dest.exists():
                    existing.unlink()
                else:
                    shutil.move(str(existing), dest)

    for split_name, split_keys in splits.items():
        split_dir = proc_base / split_name
        n_written = 0
        for (aoa_s, mach_s) in split_keys:
            out = split_dir / f"aoa{float(aoa_s):+.2f}_m{float(mach_s):.2f}.npz"
            if out.exists() and not force:
                continue
            hr_src, lr_src = paired[(aoa_s, mach_s)]
            hr = np.load(hr_src, allow_pickle=True)
            lr = np.load(lr_src, allow_pickle=True)
            extra = {}
            if 'primitives_grad' in lr.files:
                extra['lr_primitives_grad'] = lr['primitives_grad']
            np.savez(out,
                     hr_node_pos=hr['node_pos'],
                     hr_node_area=hr['node_area'],
                     hr_primitives=hr['primitives'],
                     hr_primitives_grad=hr['primitives_grad'],
                     lr_node_pos=lr['node_pos'],
                     lr_node_area=lr['node_area'],
                     lr_primitives=lr['primitives'],
                     **extra)
            n_written += 1
        print(f"{proc_base.name}/{split_name}: {len(split_keys)} samples, {n_written} écrits")


def load_mesh(path):
    return np.load(path, allow_pickle=True).item()


def main():
    parser = argparse.ArgumentParser(description='Prétraitement SR-CFD : processed + graphes + kNN + stats')
    parser.add_argument('--data',     default='data/', help='Racine des données (ex: data/)')
    parser.add_argument('--geometry', default='diamond', help='Géométrie (ex: diamond, naca0012)')
    parser.add_argument('--hr_res',   type=float, default=0.025)
    parser.add_argument('--lr_res',   type=float, default=0.1)
    parser.add_argument('--knn_k',    type=int, nargs='+', default=[6, 16])
    parser.add_argument('--force',    action='store_true',
                        help="Régénère les fichiers processed même s'ils existent")
    args = parser.parse_args()

    # Layout global pour la géométrie et les résolutions
    # Contient les chemins vers raw_dir, proc_dir, graphs_dir, knn_dir, stats_path
    layout = DataLayout.from_root(
        Path(args.data).resolve(), args.geometry, args.lr_res, args.hr_res)
    hr_str = f'h{args.hr_res}'
    lr_str = f'h{args.lr_res}'
    hr_tag = _res_tag(args.hr_res)
    lr_tag = _res_tag(args.lr_res)

    print(f"\nPrétraitement  {layout.root}  géométrie={layout.geometry}")
    print(f"  HR={hr_str}  LR={lr_str}\n")

    print("Construction du dataset traité...")
    build_processed(layout, hr_res=hr_str, lr_res=lr_str, force=args.force)

    mesh_hr = load_mesh(layout.mesh_path(args.hr_res))
    mesh_lr = load_mesh(layout.mesh_path(args.lr_res))

    graphs_dir = layout.graphs_dir
    graphs_dir.mkdir(parents=True, exist_ok=True)
    build_graph(mesh_hr, graphs_dir / f'graph_{hr_tag}.npz')
    build_graph(mesh_lr, graphs_dir / f'graph_{lr_tag}.npz')

    knn_dir = layout.knn_dir
    knn_dir.mkdir(parents=True, exist_ok=True)
    c_hr = np.asarray(mesh_hr.barycenter, dtype=np.float32)
    c_lr = np.asarray(mesh_lr.barycenter, dtype=np.float32)
    for k in args.knn_k:
        build_knn(c_hr, c_lr, k=k,
                  save_path=knn_dir / f'knn_{lr_tag}_to_{hr_tag}_k{k}.npz')

    # Stats calculées sur hr_primitives (toujours h=0.025 pour l'instant)
    stats_path = layout.stats_path
    if stats_path.exists():
        print(f"Stats existantes réutilisées ({stats_path}) : indépendantes de LR")
    else:
        compute_stats(layout, stats_path)

    print("\nTerminé")


if __name__ == '__main__':
    main()
