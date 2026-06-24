import numpy as np
from pathlib import Path

from utils.layout import DataLayout


def compute(layout: DataLayout, save_path):
    train_dir = layout.proc_dir() / 'train'
    n, s, s2 = 0, np.zeros(4, np.float64), np.zeros(4, np.float64)
    for f in sorted(train_dir.glob('aoa*.npz')):
        prim = np.load(f)['hr_primitives'].astype(np.float64)
        n  += prim.shape[0]
        s  += prim.sum(0)
        s2 += (prim ** 2).sum(0)
    mu = (s / n).astype(np.float32)
    sig = np.sqrt(np.maximum(s2 / n - (s / n) ** 2, 1e-12)).astype(np.float32)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path, mu=mu, sig=sig)
    print(f"Stats exportées: {save_path}  mu={mu.round(4)}  sig={sig.round(4)}")
