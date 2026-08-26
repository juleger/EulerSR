"""Utilitaires d'évaluation : features, inférence, métriques."""
from __future__ import annotations
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx

from eval.loader import ModelEntry
from preprocessing.dataset import _MACH_MID, _MACH_SCALE, _AOA_SCALE
from utils.metrics import idw_weights
from utils.coords import center_scale

_PRIM_NAMES = ['rho', 'u', 'v', 'p']
_BATCH_SIZE = 16

# Features
@dataclass(frozen=True)
class _MeshGeom:
    hr_n: np.ndarray  # (N_hr, 2)
    lr_n: np.ndarray  # (N_lr, 2)


def mesh_geom_from_case(d: dict, coord_norm: str = 'domain',
                        mesh_meta: dict | None = None) -> _MeshGeom:
    hr_pos = d['hr_node_pos'].astype(np.float64)
    lr_pos = d['lr_node_pos'].astype(np.float64)
    ctr, scl = center_scale(hr_pos, lr_pos, coord_norm, mesh_meta)
    return _MeshGeom(
        hr_n=((hr_pos - ctr) / scl).astype(np.float32),
        lr_n=((lr_pos - ctr) / scl).astype(np.float32),
    )


def _resolve_mach_norm(cfg: dict | None) -> tuple[float, float]:
    """(mid, scale) du conditionnement Mach vu par un checkpoint donné à
    l'entraînement, lu dans son cfg persisté (clé 'mach_norm', ajoutée par
    train.py). Repli sur l'historique (0.7, 3.0) si absente -- checkpoints
    entraînés avant ce champ (DAM/FAM_multi_final, 6geo, SIAM confirm...) :
    ne JAMAIS changer ce repli, c'est ce qui garantit qu'ils continuent de
    recevoir exactement le même conditionnement qu'à l'entraînement."""
    mn = (cfg or {}).get('mach_norm')
    if mn is None:
        return (_MACH_MID, _MACH_SCALE)
    lo, hi = float(mn[0]), float(mn[1])
    return ((lo + hi) / 2, (hi - lo) / 2)


def build_hr_feat(geom: _MeshGeom, mach_in: float, aoa_in: float,
                  geom_id: int = 0,
                  mach_norm: tuple[float, float] = (_MACH_MID, _MACH_SCALE)) -> np.ndarray:
    """mach_norm = (mid, scale) du conditionnement Mach vu par CE checkpoint à
    l'entraînement -- par défaut l'historique (0.7, 3.0) pour compatibilité,
    cf. eval/runner.py:_resolve_mach_norm qui le résout par entrée de modèle."""
    N = geom.hr_n.shape[0]
    mmid, mscale = mach_norm
    return np.stack([
        geom.hr_n[:, 0], geom.hr_n[:, 1],
        np.full(N, (mach_in - mmid) / mscale, np.float32),
        np.full(N, aoa_in / _AOA_SCALE, np.float32),
        np.full(N, float(geom_id), np.float32),
    ], axis=1)


def build_lr_feat(geom: _MeshGeom, d: dict, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    lr_prim = d['lr_primitives'].astype(np.float32)
    parts: list[np.ndarray] = [(lr_prim - mu) / sig, geom.lr_n]
    if 'lr_primitives_grad' in d:
        lg = d['lr_primitives_grad'].astype(np.float32)
        gp = lg[:, 3, :]
        du = lg[:, 1, 0] + lg[:, 2, 1]
        parts += [np.arcsinh(gp / (sig[3] + 1e-8)),
                  np.arcsinh(du / (sig[1] + 1e-8))[:, None]]
    return np.concatenate(parts, axis=1)


def build_wall_feat(d: dict, mu: np.ndarray, sig: np.ndarray, mesh_meta: dict) -> np.ndarray:
    """Équivalent de build_lr_feat pour FAMWall/DAMWall : observations de bord
    (wall_pos/wall_normal/wall_value/wall_s, fusionnées dans `d` par
    utils.layout.load_sample depuis le store _wall compagnon). coord_norm=
    'object' imposé pour les modèles bord (cf. models/fam_wall.py)."""
    from utils.coords import object_center_scale
    ctr, scl = object_center_scale(mesh_meta)
    wall_pos_n = (d['wall_pos'].astype(np.float32) - ctr) / scl
    wall_val_n = (d['wall_value'].astype(np.float32) - mu) / sig
    return np.concatenate([wall_pos_n, d['wall_normal'].astype(np.float32),
                           wall_val_n, d['wall_s'].astype(np.float32)[:, None]], axis=1)


def build_features(d: dict, mach_in: float, aoa_in: float,
                   stats: dict, geom_id: int = 0, coord_norm: str = 'domain',
                   mesh_meta: dict | None = None, is_wall: bool = False,
                   mach_norm: tuple[float, float] = (_MACH_MID, _MACH_SCALE)) -> tuple:
    mu = stats['mu'].astype(np.float32)
    sig = stats['sig'].astype(np.float32)
    geom = mesh_geom_from_case(d, coord_norm, mesh_meta)
    hr_feat = jnp.array(build_hr_feat(geom, mach_in, aoa_in, geom_id, mach_norm=mach_norm))
    if is_wall:
        cond_feat = jnp.array(build_wall_feat(d, mu, sig, mesh_meta))
    else:
        cond_feat = jnp.array(build_lr_feat(geom, d, mu, sig))
    hr_prim = d['hr_primitives'].astype(np.float32) if 'hr_primitives' in d else None
    return hr_feat, cond_feat, hr_prim, mu, sig

# Inférence
def predict_idw(lr_prim: np.ndarray, idx: np.ndarray, dist: np.ndarray,
                k: int = 6) -> tuple[np.ndarray, float]:
    idx_k = np.asarray(idx)[:, :k].astype(np.int32)
    w = idw_weights(np.asarray(dist)[:, :k])
    t0 = time.perf_counter()
    pred = (w[:, :, None].astype(np.float32) * lr_prim[idx_k]).sum(axis=1)
    t_ms = (time.perf_counter() - t0) * 1e3
    return pred, t_ms


def predict_wall_baseline(d: dict, mach_in: float, aoa_in: float,
                          wd_exp: np.ndarray, k: int = 6) -> tuple[np.ndarray, float]:
    """Baseline "sans réseau" pour un testset FAMWall/DAMWall (idw_knn['mode']
    == 'wall') : IDW(bord) blend freestream, en unités physiques directes
    (même esprit que predict_idw, qui n'est pas non plus normalisé -- cf.
    models/fam_wall.wall_baseline pour la version normalisée utilisée dans le
    réseau). d doit contenir wall_pos/wall_value (fusionnés par
    utils.layout.load_sample depuis le store _wall compagnon) et hr_node_pos."""
    from scipy.spatial import cKDTree
    t0 = time.perf_counter()
    wall_pos = d['wall_pos'].astype(np.float64)
    wall_val = d['wall_value'].astype(np.float32)
    k_eff = min(k, len(wall_pos))
    dist, idx = cKDTree(wall_pos).query(d['hr_node_pos'].astype(np.float64), k=k_eff, workers=-1)
    w = idw_weights(dist).astype(np.float32)
    idw = (w[:, :, None] * wall_val[idx]).sum(axis=1)
    c_inf = np.sqrt(1.4)
    u_inf = mach_in * c_inf * np.cos(np.deg2rad(aoa_in))
    v_inf = mach_in * c_inf * np.sin(np.deg2rad(aoa_in))
    freestream = np.array([1.0, u_inf, v_inf, 1.0], np.float32)
    pred = (wd_exp[:, None] * idw + (1.0 - wd_exp[:, None]) * freestream).astype(np.float32)
    t_ms = (time.perf_counter() - t0) * 1e3
    return pred, t_ms


def predict_det(entry: ModelEntry, d: dict, mach_in: float, aoa_in: float,
                stats: dict, knn: dict | None = None,
                geom_id: int = 0, mesh_meta: dict | None = None) -> tuple[np.ndarray, float]:
    coord_norm = (entry.cfg or {}).get('architecture', {}).get('coord_norm', 'domain')
    is_wall = hasattr(entry.model, 'wall_encoder')
    hr_feat, cond_feat, _, mu, sig = build_features(d, mach_in, aoa_in, stats, geom_id,
                                                    coord_norm, mesh_meta, is_wall=is_wall,
                                                    mach_norm=_resolve_mach_norm(entry.cfg))
    knn_used = knn if knn is not None else entry.knn
    t0 = time.perf_counter()
    pred = jax.block_until_ready(entry.model.predict(hr_feat, cond_feat, knn_used))
    t_ms = (time.perf_counter() - t0) * 1e3
    return (np.array(pred) * sig + mu).astype(np.float32), t_ms


def predict_ensemble(entry: ModelEntry, d: dict, mach_in: float, aoa_in: float,
                     stats: dict, knn: dict | None = None, geom_id: int = 0,
                     key: jax.Array | None = None, mesh_meta: dict | None = None,
                     ) -> tuple[np.ndarray, np.ndarray, float]:
    """Ensemble stochastique (SDE ou FAMWall) : renvoie (mean, std, t_ms) en unites physiques."""
    coord_norm = (entry.cfg or {}).get('architecture', {}).get('coord_norm', 'domain')
    is_wall = hasattr(entry.model, 'wall_encoder')
    hr_feat, cond_feat, _, mu, sig = build_features(d, mach_in, aoa_in, stats, geom_id,
                                                    coord_norm, mesh_meta, is_wall=is_wall,
                                                    mach_norm=_resolve_mach_norm(entry.cfg))
    knn_used = knn if knn is not None else entry.knn
    if key is None:
        key = jax.random.PRNGKey(int(getattr(entry.model, 'sample_seed', 0)))
    t0 = time.perf_counter()
    samples = jax.block_until_ready(
        entry.model.sample(hr_feat, cond_feat, knn_used, key=key))  # [S, N, 4] normalise
    t_ms = (time.perf_counter() - t0) * 1e3
    samples = np.asarray(samples) * sig + mu
    mean = samples.mean(axis=0).astype(np.float32)
    std = samples.std(axis=0).astype(np.float32)
    return mean, std, t_ms


def make_batch_predict(model, knn):
    @nnx.jit
    def _fn(hr_b: jax.Array, lr_b: jax.Array) -> jax.Array:
        return jax.vmap(lambda h, l: model.predict(h, l, knn))(hr_b, lr_b)
    return _fn


def run_batched(batch_fn, hr_feats: list, lr_feats: list,
                batch_size: int, mu: np.ndarray, sig: np.ndarray,
                ) -> tuple[list[np.ndarray], float]:
    n = len(hr_feats)
    all_preds: list[np.ndarray] = []
    t_total_ms = 0.0

    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        B = end - i
        pad = batch_size - B
        hr_np = np.stack(hr_feats[i:end] + [hr_feats[end - 1]] * pad)
        lr_np = np.stack(lr_feats[i:end] + [lr_feats[end - 1]] * pad)

        t0 = time.perf_counter()
        out = batch_fn(jnp.array(hr_np), jnp.array(lr_np))
        jax.block_until_ready(out)
        t_total_ms += (time.perf_counter() - t0) * 1e3

        preds = np.array(out)
        for j in range(B):
            all_preds.append((preds[j] * sig + mu).astype(np.float32))

    return all_preds, t_total_ms / n


def save_metrics_csv(results: dict, out_path: Path):
    cases = results['cases']
    all_rows = []
    if results['idw']:
        all_rows.append(results['idw'])
    all_rows.extend(results['rows'])

    with open(out_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['model'] + [c['label'] for c in cases] + ['mean_L2', 'mean_time_ms'])
        for row in all_rows:
            l2v = [f'{v:.4f}' if v is not None else 'N/A' for v in row['l2']]
            valid = [v for v in row['l2'] if v is not None]
            ml2 = f'{np.mean(valid):.4f}' if valid else 'N/A'
            mt = f'{np.mean(row["time_ms"]):.2f}'
            w.writerow([row['name']] + l2v + [ml2, mt])
    print(f"  > {out_path.name}")
