import sys
import re
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

"""
Script global de prétraitement des données : construction du dataset traité à partir des fichiers bruts, chargement des maillages, construction des graphes de voisinage pour le GNN, construction des voisinages kNN, et calcul des statistiques pour la normalisation. Les données traitées sont sauvegardées dans le dossier processed/, prêtes à être utilisées pour l'entraînement des modèles via SRDataset.
"""


_FNAME_RE = re.compile(r'AOA([+-]?[0-9.]+)_M([0-9.]+)_.*_t([0-9.]+)\.npz$')
_HR_FIELDS = ('node_pos', 'node_area', 'primitives', 'primitives_grad')
_LR_FIELDS = ('node_pos', 'node_area', 'primitives')


def _res_tag(r: float) -> str:
    return 'h' + f'{r}'.replace('0.', '')


def build_processed(data_dir: Path, hr_res='h0.025', lr_res='h0.1', force=False):
    """
    Construction du dataset traité à partir des fichiers bruts : appariement des fichiers HR et LR par Mach/AoA, extraction des champs nécessaires, et sauvegarde dans processed/.

    Split train/val/test (90/10/10) de manière harmonieuse sur toutes les plages Mach/AoA
    """
    all_files: dict[str, dict] = {}
    for res in (hr_res, lr_res):
        all_files[res] = {}
        for aoa_dir in sorted((data_dir / 'raw' / res).iterdir()):
            if not aoa_dir.is_dir():
                continue
            for f in sorted(aoa_dir.iterdir()):
                m = _FNAME_RE.search(f.name)
                if not m:
                    continue
                key = (m.group(1), m.group(2))
                t = float(m.group(3))
                all_files[res].setdefault(key, {})[t] = f

    # Extraction des paires HR/LR correspondantes, en s'assurant que les tf matchent pour un même Mach/AoA.
    keys = set(all_files[hr_res]) & set(all_files[lr_res])
    paired: dict[tuple, tuple] = {}
    for key in sorted(keys):
        common_ts = set(all_files[hr_res][key]) & set(all_files[lr_res][key])
        if not common_ts:
            continue
        best_t = max(common_ts)
        paired[key] = (all_files[hr_res][key][best_t], all_files[lr_res][key][best_t])

    keys_list = sorted(paired.keys())
    rng = np.random.default_rng(42)
    rng.shuffle(keys_list)
    n = len(keys_list)
    n_test = max(1, int(0.1 * n))
    n_val = max(1, int(0.1 * n))

    # Split 90/10/10
    splits = {
        'test': keys_list[:n_test],
        'val': keys_list[n_test:n_test + n_val],
        'train': keys_list[n_test + n_val:],
    }

    proc_base = data_dir / 'processed'

    for split_name, split_keys in splits.items():
        split_dir = proc_base / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        n_written = 0
        for (aoa_s, mach_s) in split_keys:
            out = split_dir / f"aoa{float(aoa_s):+.2f}_m{float(mach_s):.2f}.npz"
            if out.exists() and not force:
                continue
            hr_src, lr_src = paired[(aoa_s, mach_s)]
            hr = np.load(hr_src, allow_pickle=True)
            lr = np.load(lr_src, allow_pickle=True)

            # Sauvegarde seulement les features utiles pour l'entrainement
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
        print(f"processed/{split_name}: {len(split_keys)} samples, {n_written} sauvegardés")


def load_mesh(path):
    return np.load(path, allow_pickle=True).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/diamond/')
    parser.add_argument('--hr_res', type=float, default=0.025, help='Taille du maillage HR')
    parser.add_argument('--lr_res', type=float, default=0.1, help='Taille du maillage LR')
    parser.add_argument('--knn_k', type=int, nargs='+', default=[6, 16], help='k pour les voisinages kNN à construire')
    parser.add_argument('--force', action='store_true', help='Régénère les fichiers processed même s\'ils existent déjà')
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    hr_str = f'h{args.hr_res}'
    lr_str = f'h{args.lr_res}'
    hr_tag = _res_tag(args.hr_res)
    lr_tag = _res_tag(args.lr_res)

    print(f"\nPrétraitement  {data_dir}")
    print(f"  HR={hr_str}  LR={lr_str}\n")

    print("Construction du dataset traité...")
    build_processed(data_dir, hr_res=hr_str, lr_res=lr_str, force=args.force)

    mesh_hr = load_mesh(data_dir / f'mesh_{args.hr_res}.npy')
    mesh_lr = load_mesh(data_dir / f'mesh_{args.lr_res}.npy')

    graphs_dir = data_dir / 'graphs'
    build_graph(mesh_hr, graphs_dir / f'graph_{hr_tag}.npz')
    build_graph(mesh_lr, graphs_dir / f'graph_{lr_tag}.npz')
    knn_dir = data_dir / 'knn'
    c_hr = np.asarray(mesh_hr.barycenter, dtype=np.float32)
    c_lr = np.asarray(mesh_lr.barycenter, dtype=np.float32)
    for k in args.knn_k:
        build_knn(c_hr, c_lr, k=k, save_path=knn_dir / f'knn_{lr_tag}_to_{hr_tag}_k{k}.npz')

    compute_stats(data_dir, data_dir / 'stats.npz')

    print("\nTerminé ")


if __name__ == '__main__':
    main()
