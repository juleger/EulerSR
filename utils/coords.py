"""Normalisation des coordonnées (x,y) en features réseau,partagée entre
l'entraînement (preprocessing/dataset.py), l'évaluation (eval/core.py) et les
panneaux de suivi (utils/viz/training.py), pour éviter la dérive entre trois
implémentations dupliquées de la même formule.

Permet de choisir entre deux comportements :
- 'domain' : normalisation par le bbox du nuage de points HR+LR (comportement historique)
- 'object' : normalisation par l'obstacle (mesh.metadata : chord, center), invariant
  par translation/mise à l'échelle de l'obstacle.
"""
from __future__ import annotations
import numpy as np


def domain_center_scale(hr_pos: np.ndarray, lr_pos: np.ndarray) -> tuple[np.ndarray, float]:
    """Comportement historique (défaut) : bbox du nuage de points HR+LR."""
    pts = np.concatenate([hr_pos, lr_pos], axis=0).astype(np.float64)
    ctr = (pts.max(0) + pts.min(0)) / 2
    scl = float((pts.max(0) - pts.min(0)).max() / 2)
    return ctr, scl


def object_center_scale(mesh_meta: dict) -> tuple[np.ndarray, float]:
    """Centre = mesh_meta['center'] (cx,cy) ; échelle = mesh_meta['chord']/2.
    Invariant par translation/mise à l'échelle de l'obstacle."""
    c = mesh_meta['center']
    ctr = np.array([c['cx'], c['cy']], dtype=np.float64)
    scl = float(mesh_meta['chord']) / 2
    return ctr, scl


def center_scale(hr_pos: np.ndarray, lr_pos: np.ndarray, mode: str = 'domain',
                 mesh_meta: dict | None = None) -> tuple[np.ndarray, float]:
    """Point d'entrée unique : dispatch 'domain'/'object'."""
    if mode == 'object':
        if mesh_meta is None:
            raise ValueError(
                "coord_norm='object' nécessite mesh_meta (mesh.metadata : chord, center).")
        return object_center_scale(mesh_meta)
    return domain_center_scale(hr_pos, lr_pos)
