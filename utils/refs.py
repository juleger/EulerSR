
import re
from pathlib import Path

import numpy as np

GAMMA = 1.4

_PROC_RE = re.compile(r'^aoa([+-]?\d+\.\d+)_m(\d+\.\d+)$')


def _res_tag(r: float) -> str:
    return 'h' + f'{r}'.replace('0.', '')


REFERENCE_CASES = [
    (0.9, 5.0, 'M09_AoA5'),
    (1.10, -3.0, 'M110_AoA-3'),
    (2.0, 0.0, 'M20_AoA0'),
]

NACA_REFERENCE_CASES = [
    (0.80, 2.0, 'M080_AoA2'),
    (1.10, -4.0, 'M110_AoA-4'),
    (1.50, 0.0, 'M150_AoA0'),
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
