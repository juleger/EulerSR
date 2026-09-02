"""Résolution FVM live (LR + HR) pour un cas unique : mesure de temps non
biaisée, et mesure de convergence en warm-start (nb d'itérations pour
reconverger à partir d'un champ reconstruit). Tourne dans un subprocess dédié
(eval/_fvm_solve_worker.py, imports/JAX_ENABLE_X64 en conflit sinon).
"""
from __future__ import annotations
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
_WORKER = Path(__file__).resolve().parent / '_fvm_solve_worker.py'

# tf par géométrie, repris des scripts slurm/run_euler_grid*.sbatch.
_DEFAULT_TF = {'diamond': 5.0}
_DEFAULT_TF_FALLBACK = 3.0
_DEFAULT_THRESHOLD = 1e-8

# Même regex que utils/build_fvm_times.py:_TIME_FALLBACK_RE.
_TIME_LINE_RE = re.compile(r'Simulation termin.+? en ([\d.]+)s')
_SUMMARY_JSON_RE = re.compile(r'^FVM_SUMMARY_JSON=(.+)$', re.MULTILINE)
_PROBE_JSON_RE = re.compile(r'^FVM_PROBE_JSON=(.+)$', re.MULTILINE)

_CACHE_PATH = REPO_ROOT / 'results' / 'eval_case' / '_fvm_time_cache.json'
_SCRATCH_DIR = REPO_ROOT / 'results' / 'eval_case' / '_fvm_scratch'
_JAX_CACHE_DIR = REPO_ROOT / 'results' / 'eval_case' / '_jax_cache'
_CONVERGENCE_CACHE_PATH = REPO_ROOT / 'results' / 'convergence_sweep' / '_convergence_cache.json'

# valeur bien plus fine que le défaut config.toml (10000) : sinon un seul point
# de contrôle sur tout le run, impossible de distinguer un warm-start rapide.
_DEFAULT_CHECK_EVERY = 5
# Seuil par défaut de solve_convergence, distinct de _DEFAULT_THRESHOLD (celui
# de la génération du dataset, trop strict pour reconverger en pratique).
_DEFAULT_CONVERGENCE_THRESHOLD = 1e-5


def _cache_key(geometry: str, res_h: float, mach: float, aoa: float) -> str:
    return f'{geometry}|h={res_h:g}|M={mach:.4f}|AoA={aoa:.4f}'


def _convergence_cache_key(geometry: str, res_h: float, mach: float, aoa: float, label: str,
                            tf: float | None, check_every: int, threshold: float) -> str:
    safe_label = re.sub(r'[^A-Za-z0-9_.-]+', '_', label).strip('_') or 'run'
    return (f'{geometry}|h={res_h:g}|M={mach:.4f}|AoA={aoa:.4f}|label={safe_label}|'
            f'tf={tf!r}|check_every={check_every}|threshold={threshold:.6e}')


def _load_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text())
    except Exception:
        return {}


def _save_cache(cache_path: Path, cache: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2))


def solve_case(layout, mach: float, aoa: float, res_h: float,
                tf: float | None = None, threshold: float = _DEFAULT_THRESHOLD,
                scratch_dir: Path = _SCRATCH_DIR) -> float:
    """Résout (mach, aoa) sur le maillage 'res_h' et retourne le temps solveur
    non biaisé (wall_time_s, warm-up JIT exclu). Le champ résolu n'est pas
    renvoyé, seule la mesure de temps compte."""
    geometry = layout.geometry
    if tf is None:
        tf = _DEFAULT_TF.get(geometry, _DEFAULT_TF_FALLBACK)

    scratch = Path(scratch_dir) / geometry / f'h{res_h:g}'
    _JAX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(_WORKER),
           '--case', geometry, '--mesh-path', str(layout.mesh_path(res_h)),
           '--mach', str(float(mach)), '--aoa', str(float(aoa)),
           '--tf', str(float(tf)), '--stationarity-threshold', str(float(threshold)),
           '--scratch-dir', str(scratch)]
    env = {**os.environ,
           'JAX_ENABLE_X64': 'true',
           'JAX_COMPILATION_CACHE_DIR': str(_JAX_CACHE_DIR)}

    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Échec de la résolution FVM live (géométrie={geometry}, h={res_h}, "
            f"M={mach}, AoA={aoa}) :\n{proc.stderr[-4000:]}")

    m = _TIME_LINE_RE.search(proc.stdout)
    if m is None:
        raise RuntimeError(
            f"Impossible de récupérer le temps de résolution FVM depuis la sortie "
            f"du solveur (géométrie={geometry}, h={res_h}, M={mach}, AoA={aoa}).\n"
            f"stdout:\n{proc.stdout[-2000:]}")
    return float(m.group(1))


def solve_case_cached(layout, mach: float, aoa: float, res_h: float,
                       tf: float | None = None, threshold: float = _DEFAULT_THRESHOLD,
                       force: bool = False, cache_path: Path = _CACHE_PATH) -> dict:
    """Comme solve_case, mais met en cache le temps sur disque par (géométrie,
    h, mach, aoa) pour éviter de repayer une résolution HR à chaque relance."""
    key = _cache_key(layout.geometry, res_h, mach, aoa)
    cache = _load_cache(cache_path)

    if not force and key in cache:
        return {**cache[key], 'from_cache': True}

    wall_time_s = solve_case(layout, mach, aoa, res_h, tf=tf, threshold=threshold)

    cache[key] = {'wall_time_s': wall_time_s, 'geometry': layout.geometry,
                  'h': res_h, 'mach': mach, 'aoa': aoa}
    _save_cache(cache_path, cache)

    return {**cache[key], 'from_cache': False}


def solve_convergence(layout, mach: float, aoa: float, res_h: float,
                       init_prim: np.ndarray | None = None,
                       tf: float | None = None, threshold: float = _DEFAULT_CONVERGENCE_THRESHOLD,
                       check_every: int = _DEFAULT_CHECK_EVERY,
                       scratch_dir: Path = _SCRATCH_DIR, label: str = 'run') -> dict:
    """Résout (mach, aoa) jusqu'à stationnarité, depuis l'IC freestream
    (init_prim=None, cold start) ou un champ de primitives fourni (warm
    start). Retourne le résumé de run (stopping_step, converged, ...)."""
    geometry = layout.geometry
    if tf is None:
        tf = _DEFAULT_TF.get(geometry, _DEFAULT_TF_FALLBACK)

    safe_label = re.sub(r'[^A-Za-z0-9_.-]+', '_', label).strip('_') or 'run'
    scratch = Path(scratch_dir) / geometry / f'h{res_h:g}' / 'convergence' / safe_label
    scratch.mkdir(parents=True, exist_ok=True)
    _JAX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(_WORKER),
           '--case', geometry, '--mesh-path', str(layout.mesh_path(res_h)),
           '--mach', str(float(mach)), '--aoa', str(float(aoa)),
           '--tf', str(float(tf)), '--stationarity-threshold', str(float(threshold)),
           '--stationarity-check-every', str(int(check_every)),
           '--scratch-dir', str(scratch), '--emit-summary']

    if init_prim is not None:
        init_path = scratch / 'init_field.npy'
        np.save(init_path, np.asarray(init_prim, dtype=np.float64))
        cmd += ['--init-field', str(init_path)]

    env = {**os.environ,
           'JAX_ENABLE_X64': 'true',
           'JAX_COMPILATION_CACHE_DIR': str(_JAX_CACHE_DIR)}

    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Échec de la résolution FVM (convergence, géométrie={geometry}, h={res_h}, "
            f"M={mach}, AoA={aoa}, label={label}) :\n{proc.stderr[-4000:]}")

    m = _SUMMARY_JSON_RE.search(proc.stdout)
    if m is None:
        raise RuntimeError(
            f"Impossible de récupérer le résumé de convergence depuis la sortie "
            f"du solveur (géométrie={geometry}, h={res_h}, M={mach}, AoA={aoa}, "
            f"label={label}).\nstdout:\n{proc.stdout[-2000:]}")

    summary = json.loads(m.group(1))
    summary['label'] = label
    return summary


def solve_convergence_cached(layout, mach: float, aoa: float, res_h: float,
                              init_prim: np.ndarray | None = None, tf: float | None = None,
                              threshold: float = _DEFAULT_CONVERGENCE_THRESHOLD,
                              check_every: int = _DEFAULT_CHECK_EVERY,
                              scratch_dir: Path = _SCRATCH_DIR,
                              label: str = 'run', force: bool = False,
                              cache_path: Path = _CONVERGENCE_CACHE_PATH) -> dict:
    """Comme solve_convergence, mais met en cache le résumé sur disque par
    (géométrie, h, mach, aoa, label, tf, check_every, threshold)."""
    key = _convergence_cache_key(layout.geometry, res_h, mach, aoa, label, tf, check_every, threshold)
    cache = _load_cache(cache_path)

    if not force and key in cache:
        return {**cache[key], 'from_cache': True}

    summary = solve_convergence(layout, mach, aoa, res_h, init_prim=init_prim, tf=tf,
                                 threshold=threshold, check_every=check_every,
                                 scratch_dir=scratch_dir, label=label)
    cache[key] = summary
    _save_cache(cache_path, cache)
    return {**summary, 'from_cache': False}


def probe_residual(layout, mach: float, aoa: float, res_h: float, prim: np.ndarray,
                    scratch_dir: Path = _SCRATCH_DIR) -> dict:
    """Résidu relatif d'UN pas FVM depuis 'prim' (primitives (N,4)), sans
    évolution temporelle. Sert à calibrer le seuil de convergence dans
    eval.convergence_case."""
    geometry = layout.geometry
    scratch = Path(scratch_dir) / geometry / f'h{res_h:g}' / 'probe'
    scratch.mkdir(parents=True, exist_ok=True)
    _JAX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    init_path = scratch / f'probe_field_M{mach:.4f}_A{aoa:.4f}.npy'
    np.save(init_path, np.asarray(prim, dtype=np.float64))

    cmd = [sys.executable, str(_WORKER),
           '--case', geometry, '--mesh-path', str(layout.mesh_path(res_h)),
           '--mach', str(float(mach)), '--aoa', str(float(aoa)),
           '--tf', '1.0', '--stationarity-threshold', '1e-30',
           '--scratch-dir', str(scratch), '--init-field', str(init_path), '--probe-only']

    env = {**os.environ,
           'JAX_ENABLE_X64': 'true',
           'JAX_COMPILATION_CACHE_DIR': str(_JAX_CACHE_DIR)}

    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Échec du probe résidu (géométrie={geometry}, h={res_h}, M={mach}, "
            f"AoA={aoa}) :\n{proc.stderr[-4000:]}")

    m = _PROBE_JSON_RE.search(proc.stdout)
    if m is None:
        raise RuntimeError(
            f"Impossible de récupérer le résidu depuis la sortie du probe "
            f"(géométrie={geometry}, h={res_h}, M={mach}, AoA={aoa}).\n"
            f"stdout:\n{proc.stdout[-2000:]}")
    return json.loads(m.group(1))
