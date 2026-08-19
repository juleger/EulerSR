from __future__ import annotations
import math
import time
from pathlib import Path

import numpy as np

from utils.refs import _PROC_RE
from utils.layout import DataLayout, load_sample
from utils.coords import center_scale

_MACH_MID = (0.7 + 3.0) / 2
_MACH_SCALE = (3.0 - 0.7) / 2
_AOA_SCALE = 5.0


class SRDataset:
    """Dataset de snapshots pour l'entraînement SR-CFD.

    Ne gère uniquement qu'une seule géométrie (diamond ou naca0012) et un seul couple LR/HR.
    Chaque sample = un snapshot CFD (hr+lr) pour une géométrie donnée.
    Gère les stats mu/sig pour normaliser les primitives LR et HR.
    Split train/val/test déterministe par AoA et Mach trié 
    Sous échantillonnage stratifié du train set pour ablation
    """

    def __init__(self, layout: DataLayout,
                 split: str = 'train',
                 use_lr_grad: bool = True,
                 mach_range: tuple | None = None,
                 aoa_range: tuple | None = None,
                 aoa_step: float | None = None,
                 preload: bool = True,
                 shock_weight_factor: float = 1.0,
                 geom_id: int = 0,
                 train_fraction: float = 1.0,
                 coord_norm: str = 'domain'):
        split_dir = layout.proc_dir() / split
        d = np.load(layout.stats_path)
        self.mu = d['mu'].astype(np.float32)
        self.sig = d['sig'].astype(np.float32)
        self.use_lr_grad = use_lr_grad

        self.entries: list[tuple] = []

        # Filtrage des snapshots par plage de Mach/AoA (+ grille d'AoA : aoa_step=1.0
        # ne garde que les AoA entiers, ex: exclut les demi-AoA -3.5, -2.5, ...)
        for f in sorted(split_dir.glob('aoa*.npz')):
            m = _PROC_RE.match(f.stem)
            if not m:
                continue
            aoa, mach = float(m.group(1)), float(m.group(2))
            if mach_range is not None and not (mach_range[0] <= mach <= mach_range[1]):
                continue
            if aoa_range is not None and not (aoa_range[0] <= aoa <= aoa_range[1]):
                continue
            if aoa_step is not None and abs(aoa / aoa_step - round(aoa / aoa_step)) > 1e-6:
                continue
            self.entries.append((f, mach, aoa))

        # Sous-échantillonnage stratifié reproductible du train set uniforme pour ablation
        if split == 'train' and train_fraction < 1.0:
            by_aoa: dict[float, list] = {}
            for e in self.entries:
                by_aoa.setdefault(e[2], []).append(e)
            sampled = []
            for aoa_val in sorted(by_aoa):
                group = sorted(by_aoa[aoa_val], key=lambda e: e[1])  # tri par Mach croissant
                k = max(1, int(round(len(group) * train_fraction)))
                idx = np.unique(np.linspace(0, len(group) - 1, k).astype(int))
                sampled.extend(group[i] for i in idx)
            self.entries = sampled

        self._geom_id = geom_id
        self._sw_factor = shock_weight_factor
        self._has_lr_grad = False
        self._pos_center = np.zeros(2, np.float32)
        self._pos_scale = 1.0
        self.coord_norm = coord_norm
        if self.entries:
            probe = load_sample(self.entries[0][0])
            if use_lr_grad:
                self._has_lr_grad = 'lr_primitives_grad' in probe
            mesh_meta = None
            if coord_norm == 'object':
                import euler.jax_fvm.src.mesh  # noqa: requis pour unpickle
                mesh_hr = np.load(layout.mesh_path(layout.hr_res), allow_pickle=True).item()
                mesh_meta = mesh_hr.metadata
            ctr, scl = center_scale(probe['hr_node_pos'], probe['lr_node_pos'],
                                    coord_norm, mesh_meta)
            self._pos_center = ctr.astype(np.float32)
            self._pos_scale = scl

        # Preload : charge tous les samples en RAM numpy une seule fois
        # Evite les I/O répétitifs mais demande d'allouer bcp de RAM sur slurm (> 10 Go pour train)
        self._cache: list | None = None
        if preload and self.entries:
            _t0 = time.time()
            print(f"  SRDataset [{split}] preload {len(self.entries)} samples...",
                  end='', flush=True)
            self._cache = [self._load(i) for i in range(len(self.entries))]
            _mb = sum(sum(a.nbytes for a in item) for item in self._cache) / 1e6
            print(f" {_mb:.0f} Mo  ({time.time()-_t0:.1f}s)")

    def __len__(self):
        return len(self.entries)

    def _load(self, idx: int) -> tuple:
        """Charge et prépare un sample depuis le disque. Appelé une fois si preload=True"""
        path, mach_in, aoa_in = self.entries[idx]
        d = load_sample(path)

        hr_pos = d['hr_node_pos'].astype(np.float32)
        hr_prim = d['hr_primitives'].astype(np.float32)
        hr_grad_p = (d['hr_grad_p'] if 'hr_grad_p' in d
                     else d['hr_primitives_grad'][:, 3, :]).astype(np.float32)
        lr_pos = d['lr_node_pos'].astype(np.float32)
        lr_prim = d['lr_primitives'].astype(np.float32)

        N = hr_pos.shape[0]

        hr_pos_n = (hr_pos - self._pos_center) / self._pos_scale
        lr_pos_n = (lr_pos - self._pos_center) / self._pos_scale
        mach_n = (mach_in - _MACH_MID) / _MACH_SCALE
        aoa_n = aoa_in / _AOA_SCALE

        hr_feat = np.stack([
            hr_pos_n[:, 0], hr_pos_n[:, 1],
            np.full(N, mach_n, np.float32),
            np.full(N, aoa_n, np.float32),
            np.full(N, float(self._geom_id), np.float32),
        ], axis=1)

        lr_prim_n = (lr_prim - self.mu) / self.sig

        lr_grad_feat = None
        if self._has_lr_grad:
            lr_grad = d['lr_primitives_grad'].astype(np.float32)
            grad_p = lr_grad[:, 3, :]
            div_u = lr_grad[:, 1, 0] + lr_grad[:, 2, 1]
            grad_p_feat = np.arcsinh(grad_p / (self.sig[3] + 1e-8))
            div_u_feat = np.arcsinh(div_u / (self.sig[1] + 1e-8))[:, None]
            lr_grad_feat = np.concatenate([grad_p_feat, div_u_feat], axis=1)

        base_lr = [lr_prim_n, lr_pos_n[:, 0:1], lr_pos_n[:, 1:2]]
        lr_feat = np.concatenate(base_lr + ([lr_grad_feat] if lr_grad_feat is not None else []), axis=1)

        target = (hr_prim - self.mu) / self.sig

        gp = np.sqrt((hr_grad_p ** 2).sum(-1))
        weights = np.minimum(1.0 + self._sw_factor * gp / (gp.mean() + 1e-8), 5.0).astype(np.float32)

        grad_p_target = np.arcsinh(hr_grad_p / self.sig[3]).astype(np.float32)

        return hr_feat, lr_feat, target, weights, grad_p_target

    def __getitem__(self, idx: int) -> tuple:
        if self._cache is not None:
            return self._cache[idx]
        return self._load(idx)


class MultiSRDataset:
    """Dataset multi-géométrie : agrège plusieurs SRDataset, chacun avec ses
    propres stats mu/sig, et fournit des batches mono-géométrie via
    iter_batches() — évite le catastrophic forgetting entre géométries tout
    en partageant les poids du modèle. Geometric embedding : un entier par géométrie.
    """

    def __init__(self, datasets: list, names: list[str],
                 weights: list[float] | None = None):
        self.datasets = datasets
        self.names = names
        # entries comme SRDataset
        self.entries = [e for ds in datasets for e in ds.entries]
        # mu/sig du dataset primaire (affiché dans les métriques globales)
        self.mu = datasets[0].mu
        self.sig = datasets[0].sig
        # Poids d'échantillonnage normalisés
        if weights is None:
            sizes = [len(d) for d in datasets]
            total = sum(sizes)
            self.weights = [s / total for s in sizes]
        else:
            total_w = sum(weights)
            self.weights = [w / total_w for w in weights]

    def __len__(self) -> int:
        return sum(len(d) for d in self.datasets)

    def iter_batches(self, batch_size: int, rng):
        """Un batch = une seule géométrie.

        La géométrie est tirée aléatoirement selon self.weights à chaque batch,
        garantissant une proportion globale proche de weights sur l'epoch.
        """
        indices = [list(rng.permutation(len(ds))) for ds in self.datasets]
        pointers = [0] * len(self.datasets)
        n_batches = sum(math.ceil(len(ds) / batch_size) for ds in self.datasets)
        geom_seq = rng.choice(len(self.datasets), size=n_batches, p=self.weights)

        for g in geom_seq:
            ds = self.datasets[g]
            idx = indices[g]
            p = pointers[g]
            chunk = idx[p:p + batch_size]
            pointers[g] += batch_size
            if not chunk:
                continue
            yield [ds[i] for i in chunk], int(g)


def compute_stats(layout: DataLayout, save_path, split: str = 'train'):
    """Calcule mu/sigma sur hr_primitives d'un split et sauvegarde en .npz"""
    train_dir = layout.hr_proc_dir / split
    n, s, s2 = 0, np.zeros(4, np.float64), np.zeros(4, np.float64)
    for f in sorted(train_dir.glob('aoa*.npz')):
        prim = np.load(f)['hr_primitives'].astype(np.float64)
        n += prim.shape[0]
        s += prim.sum(0)
        s2 += (prim ** 2).sum(0)
    mu = (s / n).astype(np.float32)
    sig = np.sqrt(np.maximum(s2 / n - (s / n) ** 2, 1e-12)).astype(np.float32)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path, mu=mu, sig=sig)
    print(f"Stats exportées: {save_path}  mu={mu.round(4)}  sig={sig.round(4)}")
