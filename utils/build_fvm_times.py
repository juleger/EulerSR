"""Construit data/fvm_times.json : temps de résolution FVM (GPU) par
géométrie, résolution et cas (Mach/AoA), à partir des logs bruts
(logs/euler/*.out). Source de vérité unique pour utils/viz/eval.py.

Parse en priorité le format bloc (« [geom] AOA<aoa>_M<mach> | h=<h> | ... »
puis « terminée en <t>s ») ; si rien n'est trouvé, replie sur l'ancien format
ligne à ligne (résolution/cas/temps annoncés séparément).

Usage :
    uv run python utils/build_fvm_times.py --logs_dir logs/euler
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

_GEOM_HEADER_RE = re.compile(r'^Case\s+:\s*(\S+)', re.M)
_GPU_TABLE_RE = re.compile(r'\|\s+\d+\s+(NVIDIA \S+)')
_GPU_FALLBACK_RE = re.compile(r'NVIDIA\s+([A-Z0-9]+)')

# Format bloc : une mesure complète en un seul match (geom, aoa, mach, h, t).
_CASE_BLOCK_RE = re.compile(
    r'\[(\w+)\] AOA([+-]?[0-9.]+)_M([0-9.]+) \| h=([0-9.]+) \|.*\n.*?terminée en ([0-9.]+)s')

# Format ligne à ligne (repli) : résolution et cas suivis séparément.
_H_FALLBACK_RE = re.compile(r'(?:mesh=\S*?_h([0-9.]+)\.npy'
                            r'|Maillage charg\S* depuis \S*?_h([0-9.]+)\.npy'
                            r'|\|\s*h=([0-9.]+)\s*\|)')
_CASE_FALLBACK_RE = re.compile(r'Case \d+/\d+.*?Mach=([\d.]+).*?AoA=([-+]?\d+\.?\d*)')
_TIME_FALLBACK_RE = re.compile(r'Simulation termin.+? en ([\d.]+)s')


def _parse_fallback(text: str, geometry: str) -> list[dict]:
    """Repli ligne à ligne : résolution/cas/temps annoncés sur des lignes séparées."""
    records: list[dict] = []
    cur_h: str | None = None
    cur_case: tuple[float, float] | None = None
    for line in text.splitlines():
        hm = _H_FALLBACK_RE.search(line)
        if hm:
            cur_h = hm.group(1) or hm.group(2) or hm.group(3)
        cm = _CASE_FALLBACK_RE.search(line)
        if cm:
            cur_case = (float(cm.group(1)), float(cm.group(2)))
        tm = _TIME_FALLBACK_RE.search(line)
        if tm and cur_h is not None:
            mach, aoa = cur_case if cur_case is not None else (None, None)
            records.append({'geometry': geometry, 'aoa': aoa, 'mach': mach,
                            'h': float(cur_h), 'time_s': float(tm.group(1))})
            cur_case = None
    return records


def _parse_log(path: Path) -> tuple[str, list[dict]]:
    """Retourne (gpu, records) pour un fichier .out, format bloc puis repli."""
    try:
        text = path.read_text(errors='ignore')
    except Exception:
        return 'inconnu', []

    gpu_m = _GPU_TABLE_RE.search(text) or _GPU_FALLBACK_RE.search(text)
    gpu = gpu_m.group(1) if gpu_m else 'inconnu'
    if gpu != 'inconnu' and not gpu.startswith('NVIDIA'):
        gpu = f'NVIDIA {gpu}'

    records = [{'geometry': geom, 'aoa': float(aoa), 'mach': float(mach),
               'h': float(h), 'time_s': float(t)}
              for geom, aoa, mach, h, t in _CASE_BLOCK_RE.findall(text)]
    if records:
        return gpu, records

    geom_m = _GEOM_HEADER_RE.search(text)
    if geom_m is None:
        return gpu, []
    return gpu, _parse_fallback(text, geom_m.group(1))


def _stats(times: np.ndarray) -> dict:
    return {
        'n_cases': int(len(times)),
        'time_mean_s': round(float(times.mean()), 3),
        'time_median_s': round(float(np.median(times)), 3),
        'time_std_s': round(float(times.std()), 3),
        'time_min_s': round(float(times.min()), 3),
        'time_max_s': round(float(times.max()), 3),
        'time_total_s': round(float(times.sum()), 2),
    }


def build(logs_dir: Path) -> dict:
    all_records: list[dict] = []
    n_parsed, n_skipped = 0, 0
    for f in sorted(logs_dir.glob('*.out')):
        gpu, records = _parse_log(f)
        if records:
            n_parsed += 1
            for r in records:
                r['gpu'] = gpu
            all_records.extend(records)
        else:
            n_skipped += 1

    if not all_records:
        raise RuntimeError(f"Aucune mesure trouvée dans {logs_dir}")

    geometries = sorted({r['geometry'] for r in all_records})
    gpus = sorted({r['gpu'] for r in all_records})
    resolutions = sorted({r['h'] for r in all_records})

    def _times(records):
        return np.array([r['time_s'] for r in records], dtype=float)

    def _mach_range(records):
        machs = [r['mach'] for r in records if r['mach'] is not None]
        return (round(min(machs), 2), round(max(machs), 2)) if machs else (None, None)

    global_stats = _stats(_times(all_records))

    by_gpu = {g: _stats(_times([r for r in all_records if r['gpu'] == g]))
              for g in gpus}
    by_resolution = {f'{h:g}': _stats(_times([r for r in all_records if r['h'] == h]))
                      for h in resolutions}

    by_geometry: dict[str, dict] = {}
    by_geometry_resolution: dict[str, dict] = {}
    detail = []
    for geom in geometries:
        geo_records = [r for r in all_records if r['geometry'] == geom]
        s = _stats(_times(geo_records))
        s['mach_min'], s['mach_max'] = _mach_range(geo_records)
        by_geometry[geom] = s

        by_geometry_resolution[geom] = {}
        geo_res = sorted({r['h'] for r in geo_records})
        for h in geo_res:
            hr_records = [r for r in geo_records if r['h'] == h]
            by_geometry_resolution[geom][f'{h:g}'] = _stats(_times(hr_records))

            for gpu in sorted({r['gpu'] for r in hr_records}):
                gpu_records = [r for r in hr_records if r['gpu'] == gpu]
                aoas = sorted({r['aoa'] for r in gpu_records if r['aoa'] is not None})
                d = _stats(_times(gpu_records))
                mach_min, mach_max = _mach_range(gpu_records)
                d.update({'geometry': geom, 'gpu': gpu, 'h': h,
                          'mach_min': mach_min, 'mach_max': mach_max,
                          'aoa_values': aoas, 'n_aoa': len(aoas)})
                detail.append(d)

    return {
        'source_logs_dir': str(logs_dir.relative_to(REPO_ROOT)) if logs_dir.is_relative_to(REPO_ROOT) else str(logs_dir),
        'n_logs_parsed': n_parsed,
        'n_logs_skipped': n_skipped,
        'n_measurements': len(all_records),
        'geometries': geometries,
        'gpus': gpus,
        'resolutions': resolutions,
        'global': global_stats,
        'by_gpu': by_gpu,
        'by_resolution': by_resolution,
        'by_geometry': by_geometry,
        'by_geometry_resolution': by_geometry_resolution,
        'detail': detail,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--logs_dir', default=str(REPO_ROOT / 'logs' / 'euler'))
    ap.add_argument('--out', default=str(REPO_ROOT / 'data' / 'fvm_times.json'))
    args = ap.parse_args()

    result = build(Path(args.logs_dir))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as fh:
        json.dump(result, fh, indent=2)
    print(f"-> {out_path}  ({result['n_measurements']} mesures, "
          f"{len(result['geometries'])} géométries, {result['n_logs_parsed']} logs, "
          f"{result['n_logs_skipped']} logs écartés)")


if __name__ == '__main__':
    main()
