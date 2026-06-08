
import re
from pathlib import Path

import numpy as np

GAMMA = 1.4


REFERENCE_CASES = [
    (0.8, -1.0, 'M08_AoA-1'),
    (0.9,  5.0, 'M09_AoA5'),
    (1.1,  2.0, 'M11_AoA2'),
    (2.0, -4.0, 'M20_AoA-4'),
    (3.0,  0.0, 'M30_AoA0'),
]

_CASE_RE = re.compile(r'^aoa([+-]?\d+\.\d+)_m(\d+\.\d+)$')


def to_mach(prim: np.ndarray) -> np.ndarray:
    rho = np.maximum(prim[:, 0], 1e-12)
    u, v = prim[:, 1], prim[:, 2]
    p = np.maximum(prim[:, 3], 1e-12)
    c = np.sqrt(GAMMA * p / rho)
    return np.sqrt(u ** 2 + v ** 2) / c


def parse_case(path) -> tuple:
    m = _CASE_RE.match(Path(path).stem)
    return (float(m.group(2)), float(m.group(1))) if m else (1.0, 0.0)


def snap_features(d: dict, path) -> np.ndarray:
    mach_in, aoa = parse_case(path)
    prim = d['hr_primitives'].astype(np.float32)
    mach = to_mach(prim)
    gp = np.linalg.norm(d['hr_primitives_grad'][:, 3, :].astype(np.float32), axis=-1)
    return np.array([mach_in, aoa, float(mach.max()), float(mach.std()), float(gp.max())],
                    dtype=np.float32)


def find_ref_file(files, mach_target: float, aoa_target: float):
    best, best_d = None, float('inf')
    for f in files:
        m = _CASE_RE.match(Path(f).stem)
        if not m:
            continue
        aoa, mach = float(m.group(1)), float(m.group(2))
        d = (mach - mach_target) ** 2 + (aoa - aoa_target) ** 2
        if d < best_d:
            best_d, best = d, f
    return best
