"""TestSet : géométrie + résolution + cas d'évaluation, avec KNN adaptatif."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import matplotlib.tri as mtri

from eval.loader import ModelEntry
from utils.aero import WallCache, build_wall_cache
from utils.refs import REFERENCE_CASES, NACA_REFERENCE_CASES, find_ref_file, _PROC_RE, _res_tag
from utils.layout import DataLayout

REPO_ROOT = Path(__file__).resolve().parent.parent

_OOD_LABELS = {
    'indistrib': 'in-distribution',
    'res_ood': 'OOD résolution',
    'geo_ood': 'OOD géométrique',
    'full_ood': 'OOD géo+résolution',
}

def _find_ref_cases(layout: DataLayout, specs: list[tuple]) -> list[dict]:
    for split in ('test', 'val', 'train'):
        split_dir = layout.proc_dir() / split
        if not split_dir.exists():
            continue
        files = sorted(split_dir.glob('aoa*.npz'))
        if not files:
            continue
        found = []
        for mach_t, aoa_t, lbl in specs:
            p = find_ref_file(files, mach_t, aoa_t)
            if p:
                found.append({'label': lbl, 'path': str(p),
                              'mach_in': mach_t, 'aoa_in': aoa_t})
        if found:
            return found
    return []


def _find_test_cases(layout: DataLayout) -> list[dict]:
    split_dir = layout.proc_dir() / 'test'
    if not split_dir.exists():
        return []
    cases = []
    for f in sorted(split_dir.glob('aoa*.npz')):
        m = _PROC_RE.match(f.stem)
        if not m:
            continue
        aoa, mach = float(m.group(1)), float(m.group(2))
        cases.append({'path': str(f), 'mach_in': mach, 'aoa_in': aoa,
                      'label': f.stem, 'split': 'test'})
    return cases


def _build_hierarchical_knn(mesh_paths: dict[float, Path], lr_res: float, cfg: dict) -> dict:
    """KNN hiérarchique pour FAM/DAM sur une géométrie quelconque."""
    from utils.aero import wall_feature_array as _wfa
    from preprocessing.knn import build_hierarchy as _bh

    arch = (cfg or {}).get('architecture', {})
    res = (cfg or {}).get('resolution', {})
    hr_res = res.get('hr', 0.025)
    levels = list(arch.get('levels', [0.05, 0.1]))
    k_pool = arch.get('k_pool', 9)
    k_up = arch.get('k_up', 4)
    k_self = arch.get('k_self', 9)
    k_cond = arch.get('k_cond', 6)

    def _bary(r: float) -> np.ndarray:
        return np.asarray(
            np.load(mesh_paths[r], allow_pickle=True).item().barycenter,
            dtype=np.float64)

    def _mesh(r: float):
        return np.load(mesh_paths[r], allow_pickle=True).item()

    mesh_hr_obj = _mesh(hr_res)
    pos_hr = np.asarray(mesh_hr_obj.barycenter, dtype=np.float64)
    lr_bary = _bary(lr_res)
    lvl_meshes = [mesh_hr_obj] + [_mesh(r) for r in levels]
    lvl_pos = [pos_hr] + [np.asarray(m.barycenter, dtype=np.float64)
                          for m in lvl_meshes[1:]]

    # Features de paroi par niveau (cf. load_hierarchical_knn)
    wall_feat = [jnp.array(_wfa(m, p, lr_res))
                 for m, p in zip(lvl_meshes, lvl_pos)]

    z = _bh(lvl_pos, lr_bary, k_pool, k_up, k_self, k_cond)
    L = len(levels) + 1
    fm = {
        'pe': [jnp.array(z[f'pe{l}']) for l in range(L)],
        'cond': [{'idx': jnp.array(z[f'cond{l}_idx']),
                  'w': jnp.array(z[f'cond{l}_w'])} for l in range(L)],
        'down': [{'idx': jnp.array(z[f'down{l}_idx']),
                  'rel': jnp.array(z[f'down{l}_rel'])} for l in range(L - 1)],
        'up': [{'idx': jnp.array(z[f'up{l}_idx']),
                  'rel': jnp.array(z[f'up{l}_rel'])} for l in range(L - 1)],
        'self': {'idx': jnp.array(z['self_idx']),
                 'rel': jnp.array(z['self_rel'])},
    }
    return {'fm': fm, 'wall_feat': wall_feat}


@dataclass
class TestSet:
    """Encapsule géométrie, résolution, cas et structures nécessaires à l'évaluation."""

    tag: str  # 'diamond', 'naca0012', 'diamond_lrh2'
    label: str  # titre pour les figures
    layout: DataLayout
    stats: dict
    wc: WallCache
    triang_hr: mtri.Triangulation
    ref_cases: list[dict]  # 3–5 cas canoniques
    test_cases: list[dict]  # sweep complet du test set

    @property
    def geometry(self) -> str:
        return self.layout.geometry

    @property
    def lr_res(self) -> float:
        return self.layout.lr_res

    @property
    def hr_res(self) -> float:
        return self.layout.hr_res

    def geom_id_for(self, entry: ModelEntry) -> int:
        """Indice de cette géométrie dans les datasets d'entraînement du modèle"""
        datasets = (entry.cfg or {}).get('datasets', [])
        for i, d in enumerate(datasets):
            if d.get('name', '') == self.geometry:
                return i
        return 0

    def ood_kind(self, entry: ModelEntry) -> str:
        """'indistrib' | 'res_ood' | 'geo_ood' | 'full_ood'"""
        cfg = entry.cfg or {}
        datasets = cfg.get('datasets', [])
        if datasets:
            # Modèle multi-géométrie : géométries vues à l'entraînement listées dans datasets
            trained_geoms = {d.get('name', '') for d in datasets}
        else:
            # Modèle single-géométrie : lire la clé 'geometry'
            single = cfg.get('geometry', '')
            trained_geoms = {single} if single else set()
        trained_lr = cfg.get('resolution', {}).get('lr', 0.1)
        same_geom = not trained_geoms or self.geometry in trained_geoms
        same_res = abs(trained_lr - self.lr_res) < 1e-6
        if same_geom and same_res: return 'indistrib'
        if same_geom: return 'res_ood'
        if same_res: return 'geo_ood'
        return 'full_ood'

    def ood_label(self, entry: ModelEntry) -> str:
        return _OOD_LABELS[self.ood_kind(entry)]

    def display_name(self, entry: ModelEntry) -> str:
        return entry.name

    def ts_title(self, models: list) -> str:
        """Label pour les titres de figures : nom du test set + marqueur (OOD) si applicable."""
        kinds = {self.ood_kind(m) for m in models}
        kinds.discard('indistrib')
        if not kinds:
            return self.label
        return f"{self.label}  (OOD)"

    def build_knn(self, entry: ModelEntry) -> dict:
        """Retourne le KNN adapté à (geometry, lr_res).

        Réutilise entry.knn si celui-ci a été construit pour ce TestSet exact.
        Sinon reconstruit à partir des maillages du layout.
        """
        trained_lr = (entry.cfg or {}).get('resolution', {}).get('lr', 0.1)
        knn_matches = (
            entry.layout is not None
            and entry.layout.geometry == self.geometry
            and entry.layout.root.resolve() == self.layout.root.resolve()
            and abs(trained_lr - self.lr_res) < 1e-6
        )
        if knn_matches:
            return entry.knn

        model_cls = type(entry.model).__name__
        if model_cls in ('FAM', 'DAM'):
            return self._knn_hierarchical(entry)
        return self._knn_simple(entry)

    def build_idw_knn(self, k: int = 6) -> dict:
        """KNN simple (LR→HR) pour l'IDW baseline sur ce TestSet.

        Toujours reconstruit depuis les maillages du layout courant,
        quelle que soit la géométrie ou résolution
        """
        from scipy.spatial import cKDTree
        mesh_hr = np.load(self.layout.mesh_path(self.hr_res), allow_pickle=True).item()
        mesh_lr = np.load(self.layout.mesh_path(self.lr_res), allow_pickle=True).item()
        hr_pos = np.asarray(mesh_hr.barycenter, dtype=np.float64)
        lr_pos = np.asarray(mesh_lr.barycenter, dtype=np.float64)
        dists, idx = cKDTree(lr_pos).query(hr_pos, k=k)
        return {
            'idx': jnp.array(idx.astype(np.int32)),
            'dist': jnp.array(dists.astype(np.float32)),
            'rel': jnp.array((hr_pos[:, None, :] - lr_pos[idx]).astype(np.float32)),
        }

    def _knn_simple(self, entry: ModelEntry) -> dict:
        from scipy.spatial import cKDTree
        mesh_hr = np.load(self.layout.mesh_path(self.hr_res), allow_pickle=True).item()
        mesh_lr = np.load(self.layout.mesh_path(self.lr_res), allow_pickle=True).item()
        hr_pos = np.asarray(mesh_hr.barycenter, dtype=np.float64)
        lr_pos = np.asarray(mesh_lr.barycenter, dtype=np.float64)

        k = int(entry.knn['idx'].shape[1]) if (entry.knn and 'idx' in entry.knn) else 6
        dists, idx = cKDTree(lr_pos).query(hr_pos, k=k)
        rel = hr_pos[:, None, :] - lr_pos[idx]
        knn: dict = {
            'idx': jnp.array(idx.astype(np.int32)),
            'dist': jnp.array(dists.astype(np.float32)),
            'rel': jnp.array(rel.astype(np.float32)),
        }
        if entry.knn and 'wall_feat' in entry.knn:
            try:
                from utils.aero import wall_feature_array as _wfa2
                knn['wall_feat'] = jnp.array(_wfa2(mesh_hr, hr_pos, self.lr_res))
            except Exception:
                pass
        return knn

    def _knn_hierarchical(self, entry: ModelEntry) -> dict:
        from eval.loader import rebuild_fm_cond

        arch = (entry.cfg or {}).get('architecture', {})
        res = (entry.cfg or {}).get('resolution', {})
        levels = list(arch.get('levels', type(entry.model).DEFAULT_LEVELS))
        hr_res = res.get('hr', 0.025)

        # Même géométrie, lr_res différent → reconstruire uniquement la partie cond
        if entry.layout is not None and entry.layout.geometry == self.geometry:
            mesh_lr = np.load(self.layout.mesh_path(self.lr_res), allow_pickle=True).item()
            lr_pos = np.asarray(mesh_lr.barycenter, dtype=np.float64)
            return rebuild_fm_cond(entry.knn, lr_pos, self.layout, entry.cfg)

        # Géométrie différente → reconstruction hiérarchique complète
        needed = levels + [hr_res, self.lr_res]
        mesh_paths = {r: self.layout.mesh_path(r) for r in needed}
        return _build_hierarchical_knn(mesh_paths, self.lr_res, entry.cfg)

    @classmethod
    def from_dir(cls, data_root: str | Path, geometry: str,
                 lr_res: float = 0.1, hr_res: float = 0.025) -> 'TestSet':
        """Construit un TestSet depuis la racine des données et le nom de géométrie."""
        layout = DataLayout.from_root(data_root, geometry, lr_res, hr_res)
        ref_specs = {
            'diamond': REFERENCE_CASES,
            'naca0012': NACA_REFERENCE_CASES,
        }.get(geometry, REFERENCE_CASES)
        lr_tag = _res_tag(lr_res)
        std = abs(lr_res - 0.1) < 1e-6
        tag = geometry if std else f'{geometry}_{lr_tag}'
        label = _geom_label(geometry) if std else f'{_geom_label(geometry)} LR h={lr_res:g}'
        return cls._build(tag, label, layout, ref_specs)

    @classmethod
    def from_diamond(cls, data_root: str | Path,
                     lr_res: float = 0.1, hr_res: float = 0.025) -> 'TestSet':
        return cls.from_dir(data_root, 'diamond', lr_res, hr_res)

    @classmethod
    def from_naca(cls, data_root: str | Path,
                  lr_res: float = 0.1, hr_res: float = 0.025) -> 'TestSet':
        return cls.from_dir(data_root, 'naca0012', lr_res, hr_res)

    @classmethod
    def _build(cls, tag: str, label: str, layout: DataLayout,
               ref_specs: list) -> 'TestSet':
        import euler.jax_fvm.src.mesh  # noqa: requis pour unpickle

        mesh_hr = np.load(layout.mesh_path(layout.hr_res), allow_pickle=True).item()
        triang = mtri.Triangulation(
            np.asarray(mesh_hr.points[:, 0]),
            np.asarray(mesh_hr.points[:, 1]),
            np.asarray(mesh_hr.tris))
        wc = build_wall_cache(mesh_hr)
        stats = np.load(layout.stats_path)

        ref_cases = _find_ref_cases(layout, ref_specs)
        test_cases = _find_test_cases(layout)

        print(f"  TestSet '{tag}' : {len(ref_cases)} cas ref, "
              f"{len(test_cases)} cas test")

        return cls(tag=tag, label=label, layout=layout,
                   stats=stats, wc=wc, triang_hr=triang,
                   ref_cases=ref_cases, test_cases=test_cases)

_GEOM_LABELS = {
    'diamond': 'Diamond',
    'naca0012': 'NACA 0012',
    'rae2822': 'RAE 2822',
}

def _geom_label(geometry: str) -> str:
    return _GEOM_LABELS.get(geometry, geometry)
