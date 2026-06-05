import numpy as np
from pathlib import Path

"""
Calcul des statistiques globales (moyenne et écart-type) des primitives HR sur l'ensemble du dataset d'entraînement, utilisées pour la normalisation des données d'entrée et de sortie. La normalisation est obligatoire pour stabiliser les données.
"""

def compute(data_dir, save_path):
    train_dir = Path(data_dir) / 'processed' / 'train'
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


def load(path):
    d = np.load(path)
    return d['mu'], d['sig']
