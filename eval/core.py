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

_PRIM_NAMES = ['rho', 'u', 'v', 'p']
_BATCH_SIZE = 16

# Features
@dataclass(frozen=True)
class _MeshGeom:
    hr_n: np.ndarray  # (N_hr, 2)
    lr_n: np.ndarray  # (N_lr, 2)


def mesh_geom_from_case(d: dict) -> _MeshGeom:
    hr_pos = d['hr_node_pos'].astype(np.float64)
    lr_pos = d['lr_node_pos'].astype(np.float64)
    pts = np.concatenate([hr_pos, lr_pos])
    ctr = (pts.max(0) + pts.min(0)) / 2
    scl = (pts.max(0) - pts.min(0)).max() / 2
    return _MeshGeom(
        hr_n=((hr_pos - ctr) / scl).astype(np.float32),
        lr_n=((lr_pos - ctr) / scl).astype(np.float32),
    )


def build_hr_feat(geom: _MeshGeom, mach_in: float, aoa_in: float,
                  geom_id: int = 0) -> np.ndarray:
    N = geom.hr_n.shape[0]
    return np.stack([
        geom.hr_n[:, 0], geom.hr_n[:, 1],
        np.full(N, (mach_in - _MACH_MID) / _MACH_SCALE, np.float32),
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


def build_features(d: dict, mach_in: float, aoa_in: float,
                   stats: dict, geom_id: int = 0) -> tuple:
    mu = stats['mu'].astype(np.float32)
    sig = stats['sig'].astype(np.float32)
    geom = mesh_geom_from_case(d)
    hr_feat = jnp.array(build_hr_feat(geom, mach_in, aoa_in, geom_id))
    lr_feat = jnp.array(build_lr_feat(geom, d, mu, sig))
    hr_prim = d['hr_primitives'].astype(np.float32) if 'hr_primitives' in d else None
    return hr_feat, lr_feat, hr_prim, mu, sig

# Inférence
def predict_idw(lr_prim: np.ndarray, idx: np.ndarray, dist: np.ndarray,
                k: int = 6) -> tuple[np.ndarray, float]:
    idx_k = np.asarray(idx)[:, :k].astype(np.int32)
    w = idw_weights(np.asarray(dist)[:, :k])
    t0 = time.perf_counter()
    pred = (w[:, :, None].astype(np.float32) * lr_prim[idx_k]).sum(axis=1)
    t_ms = (time.perf_counter() - t0) * 1e3
    return pred, t_ms


def predict_det(entry: ModelEntry, d: dict, mach_in: float, aoa_in: float,
                stats: dict, knn: dict | None = None,
                geom_id: int = 0) -> tuple[np.ndarray, float]:
    hr_feat, lr_feat, _, mu, sig = build_features(d, mach_in, aoa_in, stats, geom_id)
    knn_used = knn if knn is not None else entry.knn
    t0 = time.perf_counter()
    pred = jax.block_until_ready(entry.model.predict(hr_feat, lr_feat, knn_used))
    t_ms = (time.perf_counter() - t0) * 1e3
    return (np.array(pred) * sig + mu).astype(np.float32), t_ms


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
