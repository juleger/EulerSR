from __future__ import annotations
import re
from pathlib import Path

import numpy as np

_PROC_RE = re.compile(r'^aoa([+-]?\d+\.\d+)_m(\d+\.\d+)$')

_MACH_MID   = (0.7 + 4.0) / 2    # Normalise Mach vers -1, 1
_MACH_SCALE = (4.0 - 0.7) / 2
_AOA_SCALE  = 5.0 # -5 à 5° : normalise AoA vers [-1, 1]


def _res_tag(r: float) -> str:
    return 'h' + f'{r}'.replace('0.', '')


class SRDataset:
    """ Gère l'ensemble des snapshots et représente la base de données pour l'entraînement de la super-resolution. Fournit des méthodes pour créer des échantillons d'entraînement à partir des snapshots. """

    def __init__(self, data_dir: str | Path, stats_path: str | Path,
                 split: str = 'train',
                 lr_res: float = 0.1,
                 use_lr_grad: bool = True,
                 mach_range: tuple | None = None,
                 aoa_range: tuple | None = None):
        base = Path(data_dir) / 'processed'
        split_dir = base / f'lr{_res_tag(lr_res)}' / split
        if not split_dir.exists():
            split_dir = base / split
        d = np.load(stats_path)
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

    @property
    def lr_feat_dim(self) -> int:
        return 9 if self._has_lr_grad else 6   # prim(4)+pos(2)[+grad_p(2)+div_u(1)]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        # Méthode pour charger les données d'un sample d'entraînement 
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
        ], axis=1)

        lr_prim_n = (lr_prim - self.mu) / self.sig

        # SI besoin, ajoute les features de gradient de pression (direction x et y) et divergence de la vitesse
        lr_grad_feat = None
        if self._has_lr_grad:
            lr_grad = d['lr_primitives_grad'].astype(np.float32)
            grad_p  = lr_grad[:, 3, :] # anisotrope
            div_u   = lr_grad[:, 1, 0] + lr_grad[:, 2, 1]
            # Normalisation par arcsinh pour limiter les valeurs extrêmes
            grad_p_feat = np.arcsinh(grad_p  / (self.sig[3] + 1e-8))
            div_u_feat  = np.arcsinh(div_u   / (self.sig[1] + 1e-8))[:, None]
            lr_grad_feat = np.concatenate([grad_p_feat, div_u_feat], axis=1)

        base_lr = [lr_prim_n, lr_pos_n[:, 0:1], lr_pos_n[:, 1:2]]
        lr_feat = np.concatenate(base_lr + ([lr_grad_feat] if lr_grad_feat is not None else []), axis=1)

        target = (hr_prim - self.mu) / self.sig

        # Calcul des poids pour la loss : conditionnés par le gradient de pression, pour donner plus d'influence près des chocs
        gp = np.sqrt((hr_grad[:, 3, :] ** 2).sum(-1))
        weights = np.minimum(1.0 + 2.0 * gp / (gp.mean() + 1e-8), 5.0).astype(np.float32)

        # arcsinh : borne les valeurs choc
        grad_p_target = np.arcsinh(hr_grad[:, 3, :] * (self._pos_scale / self.sig[3])).astype(np.float32)

        return hr_feat, lr_feat, target, weights, grad_p_target
