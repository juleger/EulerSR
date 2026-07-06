"""Chargement des checkpoints modèles et construction/lecture des KNN."""
from __future__ import annotations
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import yaml

from utils.layout import DataLayout
from utils.metrics import idw_weights


_MODEL_REGISTRY = {
    'fam': ('models.fam', 'FAM'),
    'dam': ('models.dam', 'DAM'),
    'siam': ('models.siam', 'SIAM'),
}


@dataclass
class ModelEntry:
    name: str
    kind: str
    model: object
    knn: dict | None
    cfg: dict | None = None
    layout: DataLayout | None = None  # layout utilisé pour construire knn


def load_model(ckpt_path: Path, layout: DataLayout) -> ModelEntry:
    ckpt_path = Path(ckpt_path)
    cfg_path = ckpt_path.with_suffix('.yaml')
    if not cfg_path.exists():
        cfg_path = ckpt_path.parent / 'model.yaml'
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config manquante : {ckpt_path.with_suffix('.yaml')} ni model.yaml")
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)

    model_type = cfg.get('model', '').lower()
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(
            f"Type inconnu '{model_type}'. Connus: {list(_MODEL_REGISTRY)}")

    import importlib
    from flax import nnx as _nnx
    module_name, class_name = _MODEL_REGISTRY[model_type]
    cls = getattr(importlib.import_module(module_name), class_name)

    with open(ckpt_path, 'rb') as fh:
        loaded = pickle.load(fh)

    if isinstance(loaded, _nnx.Module):
        model = loaded
    else:
        dummy = cls(_nnx.Rngs(0), cfg)
        graphdef, _ = _nnx.split(dummy)
        try:
            model = _nnx.merge(graphdef, loaded)
        except ValueError as e:
            raise RuntimeError(
                f"{ckpt_path.name} : incompatibilité d'architecture avec le code actuel ({e}).\n"
            ) from None

    # model_type est garanti dans _MODEL_REGISTRY (fam/dam) : tous deux passent par cls.load_knn
    name = ckpt_path.parent.name
    knn = cls.load_knn(layout, cfg)
    return ModelEntry(name=name, kind='det', model=model, knn=knn, cfg=cfg,
                      layout=layout)


def rebuild_fm_cond(knn_fm: dict, lr_pos_ood: np.ndarray, layout: DataLayout, cfg: dict) -> dict:
    #Reconstruit fm['cond'] avec les positions du maillage LR OOD
    from scipy.spatial import cKDTree as _KDTree
    res = (cfg or {}).get('resolution', {})
    hr_res = res.get('hr', 0.025)
    arch = (cfg or {}).get('architecture', {})
    levels = list(arch.get('levels', [0.05, 0.1]))
    k_cond = int(arch.get('k_cond', 6))

    def _bary(r):
        return np.asarray(
            np.load(layout.mesh_path(r), allow_pickle=True).item().barycenter,
            dtype=np.float64)

    lvl_pos = [_bary(hr_res)] + [_bary(r) for r in levels]
    tree_ood = _KDTree(lr_pos_ood.astype(np.float64))
    new_cond = []
    for p in lvl_pos:
        dist, idx = tree_ood.query(p, k=k_cond, workers=-1)
        w = idw_weights(dist)
        new_cond.append({'idx': jnp.array(idx.astype(np.int32)),
                         'w': jnp.array(w.astype(np.float32))})

    patched = dict(knn_fm)
    patched['fm'] = dict(knn_fm['fm'])
    patched['fm']['cond'] = new_cond

    # Les clés plates idx/dist/rel de DAM.load_knn pointent vers le maillage LR d'entraînement
    # les reconstruire pour le maillage OOD afin que l'IDW baseline n'indexe pas hors-bornes.
    if 'idx' in knn_fm:
        pos_hr = lvl_pos[0]
        k_flat = int(np.asarray(knn_fm['idx']).shape[1])
        dists_ood, idx_ood = tree_ood.query(pos_hr, k=k_flat)
        patched['idx'] = jnp.array(idx_ood.astype(np.int32))
        patched['dist'] = jnp.array(dists_ood.astype(np.float32))
        patched['rel'] = jnp.array(
            (pos_hr[:, None, :] - lr_pos_ood[idx_ood]).astype(np.float32))

    return patched
