"""DataLayout : résolution canonique des chemins du projet SR

Structure attendue sous data_root/ :
  raw/{geometry}/h{lr}/AOA*/...npz
  processed/{geometry}_lr{tag}/{split}/aoa*.npz
  meshes/{geometry}_h{res}.npy
  knn/{geometry}/knn_*.npz  fm_*.npz
  graphs/{geometry}/graph_*.npz
  stats/{geometry}.npz
  gt_cache/{geometry}.npz
  fvm_times.json
  figures/
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from utils.refs import _res_tag  # noqa: F401


@dataclass
class DataLayout:
    """Chemins canoniques pour un jeu de données géométrie+résolution.

    root     : racine des données, ex: data/
    geometry : nom de la géométrie, ex: 'diamond', 'naca0012'
    lr_res   : résolution LR, ex: 0.1
    hr_res   : résolution HR cible, ex: 0.025
    """

    root: Path
    geometry: str
    lr_res: float
    hr_res: float

    @property
    def raw_dir(self) -> Path:
        return self.root / 'raw' / self.geometry

    def proc_dir(self, lr_res: float | None = None) -> Path:
        """processed/{geometry}_lr{tag}/ — store LR (lr_* uniquement)."""
        r = lr_res if lr_res is not None else self.lr_res
        return self.root / 'processed' / f'{self.geometry}_lr{_res_tag(r)}'

    @property
    def hr_proc_dir(self) -> Path:
        return self.root / 'processed' / f'{self.geometry}_hr'

    @property
    def meshes_dir(self) -> Path:
        return self.root / 'meshes'

    def mesh_path(self, res: float) -> Path:
        p = self.meshes_dir / f'{self.geometry}_h{res}.npy'
        if p.exists():
            return p
        raise FileNotFoundError(
            f"Maillage h={res} introuvable dans {self.meshes_dir}\n"
            f"  Cherché : {p.name}")

    @property
    def knn_dir(self) -> Path:
        return self.root / 'knn' / self.geometry

    @property
    def graphs_dir(self) -> Path:
        return self.root / 'graphs' / self.geometry

    @property
    def stats_path(self) -> Path:
        return self.root / 'stats' / f'{self.geometry}.npz'

    @property
    def gt_cache_path(self) -> Path:
        if abs(self.hr_res - 0.025) < 1e-6:
            return self.root / 'gt_cache' / f'{self.geometry}.npz'
        return self.root / 'gt_cache' / f'{self.geometry}_{_res_tag(self.hr_res)}.npz'

    @property
    def fvm_times_path(self) -> Path:
        """Fichier FVM timing global partagé entre toutes les géométries."""
        return self.root / 'fvm_times.json'

    @classmethod
    def from_root(cls, root: str | Path, geometry: str,
                  lr_res: float = 0.1, hr_res: float = 0.025) -> 'DataLayout':
        return cls(root=Path(root), geometry=geometry,
                   lr_res=lr_res, hr_res=hr_res)


""" Chargement des samples processed (fusion HR et LR) pour l'éval et le training. """
_LR_SET_RE = re.compile(r'_lrh[0-9]+$')


def hr_store_path(lr_path) -> Path:
    """Chemin du fichier HR partagé correspondant à un fichier du store LR. """
    lr_path = Path(lr_path)
    split = lr_path.parent.name
    set_dir = lr_path.parent.parent
    hr_set = _LR_SET_RE.sub('_hr', set_dir.name)
    return set_dir.parent / hr_set / split / lr_path.name


def load_sample(path, raw_hr_override: str | Path | None = None) -> dict:
    """Charge un sample processed en fusionnant la HR partagée et le LR."""
    d = dict(np.load(path))
    if raw_hr_override is not None:
        raw = np.load(raw_hr_override, allow_pickle=True)
        d['hr_node_pos'] = raw['node_pos']
        d['hr_primitives'] = raw['primitives']
        d['hr_grad_p'] = raw['primitives_grad'][:, 3, :]
    elif 'hr_primitives' not in d:
        hp = hr_store_path(path)
        if hp.exists():
            for k, v in np.load(hp).items():
                d.setdefault(k, v)
    return d
