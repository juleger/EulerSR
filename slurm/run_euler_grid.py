#!/usr/bin/env python3
"""Usage:
  uv run python slurm/run_euler_grid.py --case diamond --mesh-path meshes/diamond/diamond_h0.05.npy \
    --mach 0.7:1.5:0.1 --aoa 0:10:2
"""
from __future__ import annotations

import argparse
import copy
import itertools
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EULER_DIR = REPO_ROOT / "euler"
for path in (REPO_ROOT, EULER_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from euler import main as euler_main
from euler.config import load_mesh, setup_dirs, print_config


def parse_range(spec: str):
    if ":" in spec:
        start, stop, step = (float(x) for x in spec.split(":"))
        n_steps = int(round((stop - start) / step))
        return [round(start + i * step, 10) for i in range(n_steps + 1)]
    if "," in spec:
        return [round(float(x), 10) for x in spec.split(",") if x.strip()]
    return [round(float(spec), 10)]


def normalize_optional_value(argv, option_name: str):
    normalized = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == option_name and i + 1 < len(argv):
            value = argv[i + 1]
            if value.startswith("-") and not value.startswith("--"):
                normalized.append(f"{option_name}={value}")
                i += 2
                continue
        normalized.append(token)
        i += 1
    return normalized


def build_case_cfg(base_cfg, case, mach, aoa):
    cfg = copy.deepcopy(base_cfg)
    cfg["case"] = case
    cfg["Mach"] = round(float(mach), 10)
    cfg["aoa"] = 0.0 if case == "bump" else round(float(aoa), 10)
    cfg["stationarity_threshold"] = float(cfg.get("stationarity_threshold", 1e-5))
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("bump", "diamond", "naca0012", "naca2412", "rae2822", "oneraD", "oa209"), required=True)
    parser.add_argument("--mesh-path", required=True)
    parser.add_argument("--mach", default="0.7:1.5:0.1")
    parser.add_argument("--aoa", default="0:0:1")
    parser.add_argument("--tf", type=float, default=None)
    parser.add_argument("--stationarity-threshold", type=float, default=None)
    parser.add_argument("--verbose", action="store_true", help="Logs détaillés par cas (par défaut : mode concis pour les logs slurm)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(normalize_optional_value(sys.argv[1:], "--aoa"))

    base_cfg = copy.deepcopy(euler_main.CFG)
    base_cfg["case"] = args.case
    base_cfg["mesh_path"] = args.mesh_path
    base_cfg["verbose"] = bool(args.verbose)
    if args.tf is not None:
        base_cfg["tf"] = float(args.tf)
    if args.stationarity_threshold is not None:
        base_cfg["stationarity_threshold"] = float(args.stationarity_threshold)

    mach_list = parse_range(args.mach)
    aoa_list = [0.0] if args.case == "bump" else parse_range(args.aoa)
    grid = list(itertools.product(mach_list, aoa_list))

    mesh = load_mesh(base_cfg)
    print(f"Loaded mesh once: {mesh.metadata.get('name', 'unknown')}")
    print(f"Cases to run: {len(grid)}")

    if args.dry_run:
        for mach, aoa in grid:
            print(f"case={args.case} Mach={mach} AoA={aoa}")
        return 0

    for idx, (mach, aoa) in enumerate(grid, start=1):
        cfg = build_case_cfg(base_cfg, args.case, mach, aoa)
        if cfg.get("verbose", True):
            print("\n" + "=" * 78)
        print(f"Case {idx}/{len(grid)}: case={cfg['case']} Mach={cfg['Mach']} AoA={cfg.get('aoa', 0.0)}")

        out_dirs = setup_dirs(cfg, mesh)
        print_config(cfg, mesh, out_dirs)

        W, inlet = euler_main.initialize(mesh, cfg)
        euler_main.run(W, mesh, inlet, cfg, out_dirs)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())