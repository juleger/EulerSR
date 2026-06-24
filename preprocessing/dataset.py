from __future__ import annotations
import math
import time

import numpy as np

from utils.refs import _PROC_RE
from utils.layout import DataLayout

_MACH_MID   = (0.7 + 4.0) / 2
_MACH_SCALE = (4.0 - 0.7) / 2
_AOA_SCALE  = 5.0


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
                 preload: bool = True,
                 shock_weight_factor: float = 1.0,
                 geom_id: int = 0,
                 train_fraction: float = 1.0,
                 seed: int = 42):
        split_dir = layout.proc_dir() / split
        d = np.load(layout.stats_path)
        self.mu = d['mu'].astype(np.float32)
        self.sig = d['sig'].astype(np.float32)
        self.use_lr_grad = use_lr_grad

        self.entries: list[tuple] = []

        # Filtrage des snapshots par plage de Mach/AoA
        for f in sorted(split_dir.glob('aoa*.npz')):
            m = _PROC_RE.match(f.stem)
            if not m:
                continue
            aoa, mach = float(m.group(1)), float(m.group(2))
            if mach_range is not None and not (mach_range[0] <= mach <= mach_range[1]):
                continue
            if aoa_range  is not None and not (aoa_range[0]  <= aoa  <= aoa_range[1]):
                continue
            self.entries.append((f, mach, aoa))

        # Sous-échantillonnage stratifié reproductible du train set
        # Stratification par AoA propre
        if split == 'train' and train_fraction < 1.0:
            rng = np.random.default_rng(seed)
            by_aoa: dict[float, list] = {}
            for e in self.entries:
                by_aoa.setdefault(e[2], []).append(e)
            sampled = []
            for aoa_val in sorted(by_aoa):
                group = by_aoa[aoa_val]
                k = max(1, int(len(group) * train_fraction))
                idx = rng.choice(len(group), size=k, replace=False)
                sampled.extend(group[i] for i in sorted(idx))
            self.entries = sampled

        self._geom_id = geom_id
        self._sw_factor = shock_weight_factor
        self._has_lr_grad = False
        self._pos_center  = np.zeros(2, np.float32)
        self._pos_scale   = 1.0
        if self.entries:
            probe = np.load(self.entries[0][0])
            if use_lr_grad:
                self._has_lr_grad = 'lr_primitives_grad' in probe
            pts = np.concatenate([probe['hr_node_pos'], probe['lr_node_pos']], axis=0).astype(np.float64)
            self._pos_center = ((pts.max(0) + pts.min(0)) / 2).astype(np.float32)
            self._pos_scale  = float((pts.max(0) - pts.min(0)).max() / 2)

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

    @property
    def lr_feat_dim(self) -> int:
        return 9 if self._has_lr_grad else 6   # prim(4)+pos(2)[+grad_p(2)+div_u(1)]

    def __len__(self):
        return len(self.entries)

    def _load(self, idx: int) -> tuple:
        """Charge et prépare un sample depuis le disque. Appelé une fois si preload=True"""
        path, mach_in, aoa_in = self.entries[idx]
        d = np.load(path)

        hr_pos  = d['hr_node_pos'].astype(np.float32)
        hr_prim = d['hr_primitives'].astype(np.float32)
        hr_grad = d['hr_primitives_grad'].astype(np.float32)
        lr_pos  = d['lr_node_pos'].astype(np.float32)
        lr_prim = d['lr_primitives'].astype(np.float32)

        N = hr_pos.shape[0]

        hr_pos_n = (hr_pos - self._pos_center) / self._pos_scale
        lr_pos_n = (lr_pos - self._pos_center) / self._pos_scale
        mach_n   = (mach_in - _MACH_MID) / _MACH_SCALE
        aoa_n    = aoa_in / _AOA_SCALE

        hr_feat = np.stack([
            hr_pos_n[:, 0], hr_pos_n[:, 1],
            np.full(N, mach_n, np.float32),
            np.full(N, aoa_n,  np.float32),
            np.full(N, float(self._geom_id), np.float32),
        ], axis=1)

        lr_prim_n = (lr_prim - self.mu) / self.sig

        lr_grad_feat = None
        if self._has_lr_grad:
            lr_grad = d['lr_primitives_grad'].astype(np.float32)
            grad_p  = lr_grad[:, 3, :]
            div_u   = lr_grad[:, 1, 0] + lr_grad[:, 2, 1]
            grad_p_feat = np.arcsinh(grad_p  / (self.sig[3] + 1e-8))
            div_u_feat  = np.arcsinh(div_u   / (self.sig[1] + 1e-8))[:, None]
            lr_grad_feat = np.concatenate([grad_p_feat, div_u_feat], axis=1)

        base_lr = [lr_prim_n, lr_pos_n[:, 0:1], lr_pos_n[:, 1:2]]
        lr_feat = np.concatenate(base_lr + ([lr_grad_feat] if lr_grad_feat is not None else []), axis=1)

        target = (hr_prim - self.mu) / self.sig

        gp = np.sqrt((hr_grad[:, 3, :] ** 2).sum(-1))
        weights = np.minimum(1.0 + self._sw_factor * gp / (gp.mean() + 1e-8), 5.0).astype(np.float32)

        grad_p_target = np.arcsinh(hr_grad[:, 3, :] / self.sig[3]).astype(np.float32)

        return hr_feat, lr_feat, target, weights, grad_p_target

    def __getitem__(self, idx: int) -> tuple:
        if self._cache is not None:
            return self._cache[idx]
        return self._load(idx)


class MultiSRDataset:
    """Dataset global multi-géométrie pour entraîner sur plusieurs datasets simultanément.

    Même approche que SRDataset, gère les deux datasets simultanément et fournit des batches mono-géométrie
    via iter_batches(), et ont leurs propres stats mu/sig pour normaliser les primitives LR et HR.
    Split train/val/test déterministe par AoA et Mach trié pour chaque dataset.

    Cette approche permet un modèle unique plus scalable, tout en empêchant un Catastrophic Forgetting entre géométries
    Car chaque batch ne contient qu'une seule géométrie, le modèle ne peut pas "oublier" l'autre géométrie.
    Ce n'est pas du fine tuning, car les poids sont partagés entre géométries, mais chaque batch est mono-géométrie.

    Geometric embedding : chaque géométrie est identifiée par un entier unique (0, 1, ...)
    """

    def __init__(self, datasets: list, names: list[str],
                 weights: list[float] | None = None):
        self.datasets = datasets
        self.names = names
        # entries comme SRDataset
        self.entries = [e for ds in datasets for e in ds.entries]
        # mu/sig du dataset primaire (affiché dans les métriques globales)
        self.mu  = datasets[0].mu
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
