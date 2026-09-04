"""TestSet : géométrie + résolution + cas d'évaluation, avec KNN adaptatif."""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import matplotlib.tri as mtri

from eval.loader import ModelEntry
from utils.aero import WallCache, build_wall_cache
from utils.refs import REFERENCE_CASES, NACA_REFERENCE_CASES, find_ref_file, _PROC_RE
from utils.layout import DataLayout
from models.dam import _LR_REF

REPO_ROOT = Path(__file__).resolve().parent.parent

_OOD_LABELS = {
    'indistrib': 'in-distribution',
    'res_ood': 'OOD résolution',
    'geo_ood': 'OOD géométrique',
    'full_ood': 'OOD géo+résolution',
}

# Résolution HR canonique
_HR_REF = 0.025
_RAW_FNAME_RE = re.compile(r'AOA([+-]?[0-9.]+)_M([0-9.]+)_.*_t([0-9.]+)\.npz$')


def _index_raw_hr(layout: DataLayout, hr_res: float) -> dict[tuple, Path]:
    """Indexe les solutions FVM brutes d'une résolution par (aoa_str, mach_str),
    en gardant le plus grand temps d'intégration dispo pour chaque cas."""
    res_dir = layout.root / 'raw' / layout.geometry / f'h{hr_res:g}'
    by_key: dict[tuple, dict[float, Path]] = {}
    if not res_dir.exists():
        return {}
    for aoa_dir in sorted(res_dir.iterdir()):
        if not aoa_dir.is_dir():
            continue
        for f in sorted(aoa_dir.iterdir()):
            m = _RAW_FNAME_RE.search(f.name)
            if not m:
                continue
            key = (m.group(1), m.group(2))
            by_key.setdefault(key, {})[float(m.group(3))] = f
    return {k: ts[max(ts)] for k, ts in by_key.items()}


def _find_ref_cases(layout: DataLayout, specs: list[tuple]) -> list[dict]:
    extrapolate = abs(layout.hr_res - _HR_REF) > 1e-6
    raw_index = _index_raw_hr(layout, layout.hr_res) if extrapolate else None
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
            if not p:
                continue
            case = {'label': lbl, 'path': str(p), 'mach_in': mach_t, 'aoa_in': aoa_t}
            if extrapolate:
                m = _PROC_RE.match(Path(p).stem)
                if not m:
                    continue
                key = (f'{float(m.group(1)):.2f}', f'{float(m.group(2)):.2f}')
                raw_path = raw_index.get(key)
                if raw_path is None:
                    continue
                case['raw_hr_path'] = str(raw_path)
            found.append(case)
        if found:
            return found
    return []


def _find_split_cases(layout: DataLayout, split: str) -> list[dict]:
    split_dir = layout.proc_dir() / split
    if not split_dir.exists():
        return []
    extrapolate = abs(layout.hr_res - _HR_REF) > 1e-6
    raw_index = _index_raw_hr(layout, layout.hr_res) if extrapolate else None
    if extrapolate and not raw_index:
        print(f"  [WARN] pas de solutions brutes a hr={layout.hr_res:g} "
              f"dans raw/{layout.geometry}/h{layout.hr_res:g}/ : {split} vide.")
    cases = []
    for f in sorted(split_dir.glob('aoa*.npz')):
        m = _PROC_RE.match(f.stem)
        if not m:
            continue
        aoa, mach = float(m.group(1)), float(m.group(2))
        case = {'path': str(f), 'mach_in': mach, 'aoa_in': aoa,
                'label': f.stem, 'split': split}
        if extrapolate:
            raw_path = raw_index.get((f'{aoa:.2f}', f'{mach:.2f}'))
            if raw_path is None:
                continue
            case['raw_hr_path'] = str(raw_path)
        cases.append(case)
    return cases


def _find_test_cases(layout: DataLayout) -> list[dict]:
    return _find_split_cases(layout, 'test')


def _find_val_cases(layout: DataLayout) -> list[dict]:
    """Cas du split 'val' -- absent pour les géométries OOD préprocessées en
    test_only=True (cf preprocessing/preprocess.py:build_processed). Sert de
    pool de calibration disjoint du sweep de test pour _maybe_recalibrate
    (eval/runner.py), quand il existe."""
    return _find_split_cases(layout, 'val')


def _build_hierarchical_knn(mesh_paths: dict[float, Path], lr_res: float, cfg: dict,
                            hr_res: float | None = None) -> dict:
    """KNN hiérarchique pour FAM/DAM/SIAM sur une géométrie et/ou résolution HR quelconque.
    Renseigne aussi res_scalar (log(lr_res/_LR_REF), cf. load_hierarchical_knn)."""
    from utils.aero import wall_feature_array as _wfa
    from preprocessing.knn import build_hierarchy as _bh

    arch = (cfg or {}).get('architecture', {})
    res = (cfg or {}).get('resolution', {})
    if hr_res is None:
        hr_res = res.get('hr', 0.025)
    levels = list(arch.get('levels', [0.05, 0.1]))
    k_pool = arch.get('k_pool', 9)
    k_up = arch.get('k_up', 4)
    k_self = arch.get('k_self', 9)
    k_cond = arch.get('k_cond', 6)
    coord_norm = arch.get('coord_norm', 'object')

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

    z = _bh(lvl_pos, lr_bary, k_pool, k_up, k_self, k_cond, coord_norm, mesh_hr_obj.metadata)
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
    return {'fm': fm, 'wall_feat': wall_feat,
            'res_scalar': jnp.array([float(np.log(lr_res / _LR_REF))], jnp.float32)}


@dataclass
class TestSet:
    """Encapsule géométrie, résolution, cas et structures nécessaires à l'évaluation."""

    tag: str  # 'diamond_lr0.1', 'naca0012_lr0.2', 'diamond_lr0.05_hr0.05'
    label: str  # titre pour les figures
    layout: DataLayout
    stats: dict
    wc: WallCache
    triang_hr: mtri.Triangulation
    ref_cases: list[dict]  # 3–5 cas canoniques
    test_cases: list[dict]  # sweep complet du test set
    hr_mesh_meta: dict = None  # mesh_hr.metadata (chord, center) — coord_norm='object'

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
        """Indice de cette géométrie dans les datasets d'entraînement du modèle.
        Géométrie inconnue : token nul (index n_geoms) si geom_cfg_prob > 0, sinon 0."""
        cfg = entry.cfg or {}
        datasets = cfg.get('datasets', [])
        for i, d in enumerate(datasets):
            if d.get('name', '') == self.geometry:
                return i
        arch = cfg.get('architecture', {})
        if arch.get('use_geom_cond', False) and arch.get('geom_cfg_prob', 0.0) > 0:
            return arch.get('n_geoms', 8)
        return 0

    def ood_kind(self, entry: ModelEntry) -> str:
        """'indistrib' | 'res_ood' | 'geo_ood' | 'full_ood'"""
        cfg = entry.cfg or {}
        datasets = cfg.get('datasets', [])
        default_lr = cfg.get('resolution', {}).get('lr', 0.1)
        if datasets:
            # Modèle multi-géométrie : géométries vues à l'entraînement listées dans datasets
            trained_geoms = {d.get('name', '') for d in datasets}
            # Résolutions LR vues pour CETTE géométrie précisément (une branche peut
            # couvrir plusieurs lr_res, ex. diamond: [0.05, 0.1, 0.2])
            branch = next((d for d in datasets if d.get('name', '') == self.geometry), None)
            trained_lr_set = set(branch.get('lr_res', [default_lr])) if branch else {default_lr}
        else:
            # Modèle single-géométrie : lire la clé 'geometry'
            single = cfg.get('geometry', '')
            trained_geoms = {single} if single else set()
            trained_lr_set = {default_lr}
        trained_hr = cfg.get('resolution', {}).get('hr', 0.025)
        same_geom = not trained_geoms or self.geometry in trained_geoms
        # OOD résolution : soit la résolution LR d'entrée (une des lr_res vues pour
        # cette géométrie), soit la résolution HR cible (ex. super-résolution/
        # extrapolation vers un maillage plus fin que l'entraînement).
        same_res = (any(abs(r - self.lr_res) < 1e-6 for r in trained_lr_set)
                    and abs(trained_hr - self.hr_res) < 1e-6)
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
        trained_hr = (entry.cfg or {}).get('resolution', {}).get('hr', 0.025)
        knn_matches = (
            entry.layout is not None
            and entry.layout.geometry == self.geometry
            and entry.layout.root.resolve() == self.layout.root.resolve()
            and abs(trained_lr - self.lr_res) < 1e-6
            and abs(trained_hr - self.hr_res) < 1e-6
        )
        if knn_matches:
            return entry.knn

        model_cls = type(entry.model).__name__
        if model_cls in ('FAM', 'DAM', 'SIAM'):
            return self._knn_hierarchical(entry)
        if model_cls in ('FAMWall', 'DAMWall'):
            return self._knn_wall(entry)
        return self._knn_simple(entry)

    def build_idw_knn(self, k: int = 6, wall: bool = False) -> dict:
        """KNN simple (LR→HR) pour l'IDW baseline sur ce TestSet.

        Toujours reconstruit depuis les maillages du layout courant,
        quelle que soit la géométrie ou résolution.

        wall=True : baseline pertinente pour un testset FAMWall/DAMWall --
        IDW(bord) blend freestream (cf. models/fam_wall.wall_baseline) au lieu
        de l'IDW volumique classique (comparer un modèle bord-seul à l'IDW du
        champ LR complet n'a pas de sens, l'info dont il dispose n'est pas la
        même). Même rôle dans le framework -- ligne "baseline sans réseau" --
        juste la source qui change.
        """
        mesh_hr = np.load(self.layout.mesh_path(self.hr_res), allow_pickle=True).item()
        hr_pos = np.asarray(mesh_hr.barycenter, dtype=np.float64)
        if wall:
            from utils.aero import wall_feature_array
            wd_exp = wall_feature_array(mesh_hr, hr_pos, 0.1)[:, 1].astype(np.float32)
            return {'mode': 'wall', 'wd_exp': wd_exp}
        from scipy.spatial import cKDTree
        mesh_lr = np.load(self.layout.mesh_path(self.lr_res), allow_pickle=True).item()
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

    def _knn_wall(self, entry: ModelEntry) -> dict:
        """KNN pour FAMWall/DAMWall (bord uniquement) -- réutilise
        load_hierarchical_knn_wall (models/fam_wall.py), même principe que
        _knn_hierarchical mais sans champ LR volumique."""
        from models.fam_wall import load_hierarchical_knn_wall
        arch = (entry.cfg or {}).get('architecture', {})
        levels = list(arch.get('levels', type(entry.model).DEFAULT_LEVELS))
        return load_hierarchical_knn_wall(self.layout, levels, entry.cfg,
                                          tag=type(entry.model).__name__)

    def _knn_hierarchical(self, entry: ModelEntry) -> dict:
        from eval.loader import rebuild_fm_cond

        arch = (entry.cfg or {}).get('architecture', {})
        res = (entry.cfg or {}).get('resolution', {})
        levels = list(arch.get('levels', type(entry.model).DEFAULT_LEVELS))
        trained_hr = res.get('hr', 0.025)
        same_hr = abs(trained_hr - self.hr_res) < 1e-6

        # Même géométrie ET même hr_res, lr_res différent
        if same_hr and entry.layout is not None and entry.layout.geometry == self.geometry:
            mesh_lr = np.load(self.layout.mesh_path(self.lr_res), allow_pickle=True).item()
            lr_pos = np.asarray(mesh_lr.barycenter, dtype=np.float64)
            return rebuild_fm_cond(entry.knn, lr_pos, self.layout, entry.cfg)

        # hr_res différente (extrapolation) et/ou géométrie différente
        needed = levels + [self.hr_res, self.lr_res]
        mesh_paths = {r: self.layout.mesh_path(r) for r in needed}
        return _build_hierarchical_knn(mesh_paths, self.lr_res, entry.cfg, hr_res=self.hr_res)

    @classmethod
    def from_dir(cls, data_root: str | Path, geometry: str,
                 lr_res: float = 0.1, hr_res: float = 0.025,
                 mach_max: float | None = None) -> 'TestSet':
        """Construit un TestSet depuis la racine des données et le nom de géométrie.

        mach_max : si donné, exclut du sweep (et des cas de référence) tout cas
        avec mach_in > mach_max -- utile pour écarter les régimes générés avec
        un tf solveur raccourci (cf. logs/euler/*, tf réduit au-delà de M=2 sur
        la plupart des géométries d'entraînement) plutôt que de les laisser
        polluer silencieusement les métriques agrégées.
        """
        layout = DataLayout.from_root(data_root, geometry, lr_res, hr_res)
        ref_specs = {
            'diamond': REFERENCE_CASES,
            'naca0012': NACA_REFERENCE_CASES,
        }.get(geometry, REFERENCE_CASES)
        tag = f'{geometry}_lr{lr_res:g}'
        label = f'{_geom_label(geometry)} (LR h={lr_res:g})'
        if abs(hr_res - _HR_REF) > 1e-6:
            tag += f'_hr{hr_res:g}'
            label += f', HR h={hr_res:g}'
        if mach_max is not None:
            tag += f'_machle{mach_max:g}'
            label += f', Mach≤{mach_max:g}'
        return cls._build(tag, label, layout, ref_specs, mach_max=mach_max)

    @classmethod
    def _build(cls, tag: str, label: str, layout: DataLayout,
               ref_specs: list, mach_max: float | None = None) -> 'TestSet':
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

        cap_note = ''
        if mach_max is not None:
            n_ref0, n_test0 = len(ref_cases), len(test_cases)
            ref_cases = [c for c in ref_cases if c['mach_in'] <= mach_max]
            test_cases = [c for c in test_cases if c['mach_in'] <= mach_max]
            cap_note = (f"  [Mach≤{mach_max:g} : {n_test0 - len(test_cases)} cas test "
                       f"et {n_ref0 - len(ref_cases)} cas ref exclus]")

        print(f"  TestSet '{tag}' : {len(ref_cases)} cas ref, "
              f"{len(test_cases)} cas test{cap_note}")

        return cls(tag=tag, label=label, layout=layout,
                   stats=stats, wc=wc, triang_hr=triang,
                   ref_cases=ref_cases, test_cases=test_cases,
                   hr_mesh_meta=mesh_hr.metadata)

_GEOM_LABELS = {
    'diamond': 'Diamond',
    'naca0012': 'NACA 0012',
    'rae2822': 'RAE 2822',
    'oneraD': 'ONERA D',
}

def _geom_label(geometry: str) -> str:
    return _GEOM_LABELS.get(geometry, geometry)
