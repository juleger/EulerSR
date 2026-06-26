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
from dataclasses import dataclass
from pathlib import Path

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
        """processed/{geometry}_lr{tag}/"""
        r = lr_res if lr_res is not None else self.lr_res
        return self.root / 'processed' / f'{self.geometry}_lr{_res_tag(r)}'

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
        return self.root / 'gt_cache' / f'{self.geometry}.npz'

    @property
    def fvm_times_path(self) -> Path:
        """Fichier FVM timing global — partagé entre toutes les géométries."""
        return self.root / 'fvm_times.json'

    @property
    def figures_dir(self) -> Path:
        return self.root / 'figures'

    @classmethod
    def from_root(cls, root: str | Path, geometry: str,
                  lr_res: float = 0.1, hr_res: float = 0.025) -> 'DataLayout':
        return cls(root=Path(root), geometry=geometry,
                   lr_res=lr_res, hr_res=hr_res)

    @classmethod
    def from_cfg(cls, cfg: dict, data_root: str | Path | None = None) -> 'DataLayout':
        """Construit un DataLayout depuis un YAML config (mode single-géométrie)."""
        root = Path(data_root or cfg['data_root'])
        geom = cfg['geometry']
        res = cfg.get('resolution', {})
        return cls(root=root, geometry=geom,
                   lr_res=res.get('lr', 0.1), hr_res=res.get('hr', 0.025))
