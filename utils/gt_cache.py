"""Cache des métriques GT sur le dataset complet, évite de repasser sur les fichiers HR."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / 'euler') not in sys.path:
    sys.path.append(str(REPO_ROOT / 'euler'))

from utils.refs import _PROC_RE
from utils.layout import DataLayout
from utils.layout import load_sample



def build_gt_cache(all_cases: list[dict], wc,
                   mesh=None, mu: np.ndarray | None = None) -> dict:
    #Calcule les grandeurs GT pour tous les cas. Retourne le dict du cache
    from utils.aero import aero_coeffs
    from utils.metrics import enthalpy_rms, entropy_violation, fvm_euler_rms

    n = len(all_cases)
    stems = np.array([c['label'] for c in all_cases])
    has_ref = np.zeros(n, dtype=bool)
    CL = np.full(n, np.nan, dtype=np.float32)
    CD = np.full(n, np.nan, dtype=np.float32)
    mach_max = np.full(n, np.nan, dtype=np.float32)
    mach_min = np.full(n, np.nan, dtype=np.float32)
    grad_p_max = np.full(n, np.nan, dtype=np.float32)
    enthalpy_hr = np.full(n, np.nan, dtype=np.float32)
    entropy_hr = np.full(n, np.nan, dtype=np.float32)
    fvm_residual_hr = np.full(n, np.nan, dtype=np.float32)
    wall_mach_segs: list[np.ndarray] = []
    wall_mach_lens: list[int] = []
    hr_prim_list: list[np.ndarray | None] = []
    n_cells = None

    compute_fvm = mesh is not None and mu is not None
    if compute_fvm:
        print(f"  Construction GT cache ({n} cas) — résidu FVM activé...")
    else:
        print(f"  Construction GT cache ({n} cas)...")

    for i, c in enumerate(all_cases):
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{n}...")
        d = load_sample(c['path'])
        if 'hr_primitives' not in d:
            wall_mach_lens.append(0)
            hr_prim_list.append(None)
            continue

        hp = d['hr_primitives'].astype(np.float32)
        if n_cells is None:
            n_cells = hp.shape[0]
        has_ref[i] = True

        ac = aero_coeffs(hp, wc, c['mach_in'])
        CL[i] = ac['CL']
        CD[i] = ac['CD']
        mach_max[i] = ac['mach_max']
        mach_min[i] = ac['mach_min']
        grad_p_max[i] = ac['grad_p_max']
        enthalpy_hr[i] = enthalpy_rms(hp, c['mach_in'])
        entropy_hr[i] = entropy_violation(hp)

        if compute_fvm:
            try:
                fvm_residual_hr[i] = fvm_euler_rms(
                    hp, mesh, c['mach_in'], c['aoa_in'], mu)
            except Exception as e:
                print(f"    [WARN] résidu FVM cas {c['label']} : {e}")

        wall_mach_segs.append(ac['wall_mach'])
        wall_mach_lens.append(len(ac['wall_mach']))
        hr_prim_list.append(hp)

    if n_cells is None:
        n_cells = 0

    wall_mach_flat = np.concatenate(wall_mach_segs) if wall_mach_segs else np.array([], np.float32)
    wall_mach_offsets = np.concatenate([[0], np.cumsum(wall_mach_lens)]).astype(np.int64)

    hr_prim = np.full((n, n_cells, 4), np.nan, dtype=np.float32)
    for i, hp in enumerate(hr_prim_list):
        if hp is not None:
            hr_prim[i] = hp

    return {
        'stems': stems,
        'has_ref': has_ref,
        'CL': CL, 'CD': CD,
        'mach_max': mach_max, 'mach_min': mach_min,
        'grad_p_max': grad_p_max,
        'enthalpy_hr': enthalpy_hr,
        'entropy_hr': entropy_hr,
        'fvm_residual_hr': fvm_residual_hr,
        'wall_mach_flat': wall_mach_flat,
        'wall_mach_offsets': wall_mach_offsets,
        'hr_prim': hr_prim,
    }


def save_gt_cache(cache: dict, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **cache)
    size_mb = out_path.stat().st_size / 1e6
    n = len(cache['stems'])
    print(f"  -> GT cache : {out_path}  ({n} cas, {size_mb:.1f} MB)")


def load_gt_cache(npz_path: Path) -> dict:
    # Chargement du cache et reconstruction de la liste wall_mach par cas
    d = np.load(npz_path, allow_pickle=True)
    cache: dict = {k: d[k] for k in d.files}
    flat = cache['wall_mach_flat']
    offs = cache['wall_mach_offsets']
    cache['wall_mach'] = [flat[offs[i]:offs[i + 1]] for i in range(len(offs) - 1)]
    return cache


def _cache_valid(cache: dict, all_cases: list[dict]) -> bool:
    stems_new = np.array([c['label'] for c in all_cases])
    stems_old = cache.get('stems', np.array([]))
    if len(stems_new) != len(stems_old):
        return False
    return bool((stems_new == stems_old).all())


def load_or_build_gt_cache(layout: DataLayout, all_cases: list[dict], wc,
                           mesh=None, mu: np.ndarray | None = None) -> dict:
    cache_path = layout.gt_cache_path
    if cache_path.exists():
        try:
            cache = load_gt_cache(cache_path)
            if _cache_valid(cache, all_cases):
                n = len(cache['stems'])
                print(f"    GT cache chargé ({n} cas) <- {cache_path}")
                return cache
            print(" GT cache obsolète (dataset modifié) - reconstruction.")
        except Exception as e:
            print(f"    GT cache illisible ({e}) - reconstruction.")

    cache = build_gt_cache(all_cases, wc, mesh=mesh, mu=mu)
    save_gt_cache(cache, cache_path)
    flat = cache['wall_mach_flat']
    offs = cache['wall_mach_offsets']
    cache['wall_mach'] = [flat[offs[i]:offs[i + 1]] for i in range(len(offs) - 1)]
    return cache


def _find_all_cases(layout: DataLayout) -> list[dict]:
    cases = []
    for split in ('train', 'val', 'test'):
        split_dir = layout.proc_dir() / split
        if not split_dir.exists():
            continue
        for f in sorted(split_dir.glob('aoa*.npz')):
            m = _PROC_RE.match(f.stem)
            if not m:
                continue
            aoa, mach = float(m.group(1)), float(m.group(2))
            cases.append({'path': str(f), 'mach_in': mach, 'aoa_in': aoa,
                          'label': f.stem, 'split': split})
    return cases


def main():
    import euler.jax_fvm.src.mesh  # noqa: F401

    from utils.aero import build_wall_cache

    ap = argparse.ArgumentParser(description='Construit le GT cache pour l\'évaluation SR')
    ap.add_argument('--data', default='data/', help='Racine des données (ex: data/)')
    ap.add_argument('--geometry', default='diamond', help='Géométrie (ex: diamond, naca0012)')
    ap.add_argument('--out', default=None, help='Chemin de sortie (défaut: <gt_cache_path>)')
    ap.add_argument('--lr_res', type=float, default=0.1)
    ap.add_argument('--hr_res', type=float, default=0.025)
    ap.add_argument('--force', action='store_true', help='Reconstruire même si le cache existe')
    args = ap.parse_args()

    layout = DataLayout.from_root(args.data, args.geometry, args.lr_res, args.hr_res)
    cache_path = Path(args.out) if args.out else layout.gt_cache_path

    all_cases = _find_all_cases(layout)
    if not all_cases:
        print(f"Aucun cas trouvé dans {layout.proc_dir()}")
        return
    print(f"{len(all_cases)} cas trouvés.")

    if cache_path.exists() and not args.force:
        try:
            cache = load_gt_cache(cache_path)
            if _cache_valid(cache, all_cases):
                print(f"Cache déjà à jour : {cache_path}  (--force pour forcer)")
                return
        except Exception:
            pass

    mesh_hr = np.load(layout.mesh_path(args.hr_res), allow_pickle=True).item()
    wc = build_wall_cache(mesh_hr)
    print(f"  WallCache : {len(wc.unique_wall_cell_ids)} cellules paroi")

    stats = np.load(layout.stats_path)
    mu = stats['mu'].astype(np.float64)

    cache = build_gt_cache(all_cases, wc, mesh=mesh_hr, mu=mu)
    save_gt_cache(cache, cache_path)


if __name__ == '__main__':
    main()
