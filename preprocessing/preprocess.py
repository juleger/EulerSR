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
from preprocessing.dataset import compute_stats
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
    val_set = set(map(tuple, val_keys))
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


def _index_raw(res_dir: Path, mach_min: float = 0.70,
               mach_max: float = 3.00) -> dict[tuple, Path]:
    """Indexe les raw d'une résolution, restreint à la plage Mach [mach_min, mach_max].

    Pour un même (AoA, Mach), on garde le plus grand temps d'intégration disponible.
    """
    by_key: dict[tuple, dict[float, Path]] = {}
    for aoa_dir in sorted(res_dir.iterdir()):
        if not aoa_dir.is_dir():
            continue
        for f in sorted(aoa_dir.iterdir()):
            m = _FNAME_RE.search(f.name)
            if not m:
                continue
            if not (mach_min <= float(m.group(2)) <= mach_max):
                continue
            key = (m.group(1), m.group(2))
            by_key.setdefault(key, {})[float(m.group(3))] = f
    return {k: ts[max(ts)] for k, ts in by_key.items()}


def _case_name(aoa_s: str, mach_s: str) -> str:
    return f"aoa{float(aoa_s):+.2f}_m{float(mach_s):.2f}.npz"


def _write_store(store_dir: Path, files: dict[tuple, Path], key2split: dict[tuple, str],
                 save_fn, force: bool, kind: str) -> None:
    """Écrit/synchronise un store (HR ou LR) selon le split canonique key2split."""
    for split_name in ('train', 'val', 'test'):
        (store_dir / split_name).mkdir(parents=True, exist_ok=True)

    target = {_case_name(*k): key2split[k] for k in files if k in key2split}
    for split_name in ('train', 'val', 'test'):
        for existing in list((store_dir / split_name).glob('aoa*.npz')):
            tgt = target.get(existing.name)
            if tgt is None:
                existing.unlink()
            elif tgt != split_name:
                dest = store_dir / tgt / existing.name
                existing.unlink() if dest.exists() else shutil.move(str(existing), dest)

    counts = {'train': 0, 'val': 0, 'test': 0}
    n_written = 0
    for k, src in files.items():
        split_name = key2split.get(k)
        if split_name is None:
            continue
        counts[split_name] += 1
        out = store_dir / split_name / _case_name(*k)
        if out.exists() and not force:
            continue
        save_fn(out, src)
        n_written += 1
    print(f"{store_dir.name} [{kind}]: "
          f"train={counts['train']} val={counts['val']} test={counts['test']}  "
          f"({n_written} écrits)")


def _save_hr(out: Path, src: Path) -> None:
    # On ne garde que le gradient de pression hr_grad_p (= primitives_grad[:, 3, :]).
    hr = np.load(src, allow_pickle=True)
    np.savez(out,hr_node_pos=hr['node_pos'], hr_primitives=hr['primitives'], hr_grad_p=hr['primitives_grad'][:, 3, :])


def _save_lr(out: Path, src: Path) -> None:
    lr = np.load(src, allow_pickle=True)
    kw = dict(lr_node_pos=lr['node_pos'], lr_primitives=lr['primitives'])
    if 'primitives_grad' in lr.files:
        kw['lr_primitives_grad'] = lr['primitives_grad']
    np.savez(out, **kw)


def build_processed(layout: DataLayout, hr_res='h0.025', lr_res='h0.1', force=False,
                    test_only=False, mach_min=0.70, mach_max=3.00):
    """Construction des stores traités, HR et LR séparés

    - Store HR partagé  : processed/{geom}_hr/{split}/        (écrit une fois par géométrie)
    - Store LR résolu   : processed/{geom}_lr{tag}/{split}/   (un par résolution LR)

    Split déterministe 80/10/10 calculé sur l'ensemble des cas HR.
    test_only=True : tous les cas dans le split 'test' (géométrie OOD d'évaluation).
    Seuls les cas de Mach dans [mach_min, mach_max] sont retenus.
    """
    hr_files = _index_raw(layout.raw_dir / hr_res, mach_min, mach_max)
    lr_files = _index_raw(layout.raw_dir / lr_res, mach_min, mach_max)

    if test_only:
        key2split = {k: 'test' for k in hr_files}
    else:
        from utils.refs import REFERENCE_CASES
        # Split canonique sur les cas HR, indépendant du LR
        splits = _stratified_split({k: None for k in hr_files}, REFERENCE_CASES)
        key2split = {k: s for s, keys in splits.items() for k in keys}

    # Store HR partagé (tous les cas HR disponibles)
    _write_store(layout.hr_proc_dir, hr_files, key2split, _save_hr, force, kind='HR')

    # Store LR : uniquement les cas qui ont une HR correspondante
    paired = {k: lr_files[k] for k in hr_files if k in lr_files}
    _write_store(layout.proc_dir(), paired, key2split, _save_lr, force, kind='LR')


def build_hr_only(layout: DataLayout, hr_res='h0.025', force=False,
                  mach_min=0.70, mach_max=3.00):
    """Store HR seul, tous les cas dans le split 'test' (si aucun LR dispo)"""
    hr_files = _index_raw(layout.raw_dir / hr_res, mach_min, mach_max)
    key2split = {k: 'test' for k in hr_files}
    _write_store(layout.hr_proc_dir, hr_files, key2split, _save_hr, force,
                 kind='HR (test-only)')


def load_mesh(path):
    return np.load(path, allow_pickle=True).item()


def main():
    parser = argparse.ArgumentParser(description='Prétraitement SR-CFD : processed + graphes + kNN + stats')
    parser.add_argument('--data', default='data/', help='Racine des données (ex: data/)')
    parser.add_argument('--geometry', default='diamond', help='Géométrie (ex: diamond, naca0012)')
    parser.add_argument('--hr_res', type=float, default=0.025)
    parser.add_argument('--lr_res', type=float, default=0.1)
    parser.add_argument('--knn_k', type=int, nargs='+', default=[6, 16])
    parser.add_argument('--force', action='store_true', help="Régénère les fichiers processed même s'ils existent")
    parser.add_argument('--hr_only', action='store_true',
                        help="Géométrie sans LR : construit uniquement le store HR (split test).")
    parser.add_argument('--test_only', action='store_true',
                        help="Géométrie OOD : HR+LR avec tous les cas dans le split test (pas de train/val).")
    parser.add_argument('--mach_min', type=float, default=0.70,
                        help="Borne inférieure de la plage Mach retenue (défaut 0.70).")
    parser.add_argument('--mach_max', type=float, default=3.00,
                        help="Borne supérieure de la plage Mach retenue (défaut 3.00).")
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

    if args.hr_only:
        print(f"  HR={hr_str}  (mode HR-only, pas de LR)\n")
        print("Construction du store HR (test-only)...")
        build_hr_only(layout, hr_res=hr_str, force=args.force,
                      mach_min=args.mach_min, mach_max=args.mach_max)
        # Graphe HR seul (utile pour viz / recompute de gradients)
        mesh_hr = load_mesh(layout.mesh_path(args.hr_res))
        layout.graphs_dir.mkdir(parents=True, exist_ok=True)
        build_graph(mesh_hr, layout.graphs_dir / f'graph_{hr_tag}.npz')
        stats_path = layout.stats_path
        if stats_path.exists():
            print(f"Stats existantes réutilisées ({stats_path})")
        else:
            compute_stats(layout, stats_path, split='test')
        print("\nTerminé (HR-only)")
        return

    mode = "  (test-only OOD)" if args.test_only else ""
    print(f"  HR={hr_str}  LR={lr_str}{mode}\n")

    print(f"  Plage Mach retenue : [{args.mach_min}, {args.mach_max}]")
    print("Construction du dataset traité...")
    build_processed(layout, hr_res=hr_str, lr_res=lr_str, force=args.force,
                    test_only=args.test_only,
                    mach_min=args.mach_min, mach_max=args.mach_max)

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
        compute_stats(layout, stats_path, split='test' if args.test_only else 'train')

    print("\nTerminé")


if __name__ == '__main__':
    main()
