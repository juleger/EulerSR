from __future__ import annotations

from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
import tomllib

import numpy as np


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.toml")
repo_root = Path(__file__).resolve().parents[1]


def load_default_config(config_path: str | Path | None = None):
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    config["export"] = deepcopy(config.get("export", {}))
    return config


def build_config_from_cli(default_cfg):
    parser = ArgumentParser(description="Simulation supersonique avec Euler 2D")
    parser.add_argument("--case", choices=("bump", "diamond", "naca0012", "rae2822", "oneraD"), default=default_cfg["case"], help="Cas test : bump, diamond, naca0012, rae2822 ou oneraD")
    parser.add_argument("--mach", type=float, default=default_cfg["Mach"], help="Nombre de Mach du flux entrant")
    parser.add_argument("--mesh-path", type=str, default=default_cfg["mesh_path"], help="Chemin vers le fichier de maillage (.npy)")
    parser.add_argument("--time-scheme", choices=("EE", "RK2", "RK4", "SRK2", "SSP_RK2"), default=default_cfg["time_scheme"], help="Schéma temporel")
    parser.add_argument("--flux", choices=("Rusanov", "Tadmor", "AUSM", "Roe", "HLLC"), default=default_cfg["flux"], help="Solveur de flux numérique")
    parser.add_argument("--reconstruction", choices=("constant", "MUSCL", "muscl"), default=default_cfg["reconstruction"], help="Type de reconstruction")
    parser.add_argument("--aoa", type=float, default=default_cfg.get("aoa", 0.0), help="Angle d'attaque en degrés (utilisé pour le cas diamond)")
    parser.add_argument("--stationarity-threshold", type=float, default=default_cfg.get("stationarity_threshold", 1e-6), help="Seuil d'arrêt sur le résidu relatif")
    parser.add_argument("--stationarity-check-every", type=int, default=default_cfg.get("stationarity_check_every", 25), help="Nombre de pas entre deux tests de stationnarité")
    parser.add_argument("--verbose", dest="verbose", action="store_true", default=default_cfg.get("verbose", True), help="Affiche les logs détaillés de la simulation")
    parser.add_argument("--quiet", dest="verbose", action="store_false", help="Mode concis : uniquement les infos principales au début et le temps de simulation à la fin")
    args = parser.parse_args()

    cfg = deepcopy(default_cfg)
    cfg["case"] = args.case
    cfg["Mach"] = args.mach
    cfg["mesh_path"] = args.mesh_path
    cfg["time_scheme"] = "SRK2" if args.time_scheme == "SSP_RK2" else args.time_scheme
    cfg["flux"] = args.flux
    cfg["reconstruction"] = "MUSCL" if str(args.reconstruction).lower() == "muscl" else args.reconstruction
    cfg["aoa"] = float(args.aoa)
    cfg["stationarity_threshold"] = float(args.stationarity_threshold)
    cfg["stationarity_check_every"] = max(1, int(args.stationarity_check_every))
    cfg["verbose"] = bool(args.verbose)
    return cfg


def load_mesh(cfg):
    if cfg["mesh_path"] is None:
        raise ValueError("Le chemin du maillage doit être spécifié via --mesh-path ou dans la config par défaut.")

    try:
        from jax_fvm.src.mesh import Mesh
    except ModuleNotFoundError:
        import sys

        euler_root = Path(__file__).resolve().parent
        if str(euler_root) not in sys.path:
            sys.path.insert(0, str(euler_root))
        from jax_fvm.src.mesh import Mesh

    mesh = Mesh()
    mesh.load_mesh(cfg["mesh_path"])
    return mesh


def format_h(mesh):
    h = mesh.metadata.get("h", None)
    if h is None:
        return "h"
    return f"h{float(h):.4f}".rstrip("0").rstrip(".")


def format_condition_tag(cfg):
    mach = f"M{float(cfg.get('Mach')):.2f}"
    if str(cfg.get("case", "")).lower() in ("diamond", "naca0012", "rae2822", "onerad"):
        aoa = float(cfg.get("aoa", 0.0))
        return f"AOA{aoa:.2f}_{mach}"
    return mach


def format_snapshot_name(cfg, t):
    condition_tag = format_condition_tag(cfg)
    flux = str(cfg["flux"]).upper()
    reconstruction = str(cfg["reconstruction"]).upper()
    time_scheme = str(cfg["time_scheme"]).upper()
    prefix = ""
    return f"{prefix}{condition_tag}_{flux}_{reconstruction}_{time_scheme}_t{t:.2f}"


def format_sample_id(cfg):
    return format_condition_tag(cfg)


def format_subtitle(cfg, mesh, t):
    reconstruction = str(cfg["reconstruction"]).upper()
    time_scheme = str(cfg["time_scheme"]).upper()
    if str(cfg.get("case", "")).lower() in ("diamond", "naca0012", "rae2822", "onerad"):
        return (
            f"t={t:.2f}s, M={cfg['Mach']}, AoA={float(cfg.get('aoa', 0.0)):.1f}°, "
            f"h={mesh.metadata.get('h', 'n/a')} | Solver : {cfg['flux']}, {reconstruction}, {time_scheme}"
        )
    return (
        f"t={t:.2f}s, M={cfg['Mach']}, h={mesh.metadata.get('h', 'n/a')} "
        f"| Solver : {cfg['flux']}, {reconstruction}, {time_scheme}"
    )


def setup_dirs(cfg, mesh):
    h_dir = format_h(mesh)
    case = str(cfg.get("case", ""))
    if case.lower() in ("diamond", "naca0012", "rae2822", "onerad"):
        aoa = float(cfg.get("aoa", 0.0))
        aoa_tag = f"AOA{aoa:.2f}"
        dirs = {
            "fig": repo_root / "euler" / "figures" / case / h_dir / aoa_tag,
            "res": repo_root / "data" / "raw" / case / h_dir / aoa_tag,
        }
    else:
        dirs = {
            "fig": repo_root / "euler" / "figures" / case / h_dir,
            "res": repo_root / "euler" / "results" / case / h_dir,
        }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    if cfg.get("verbose", True):
        print(f"Output : {dirs['res']}")
    return dirs


def print_config(cfg, mesh, out_dirs):
    reconstruction = str(cfg["reconstruction"]).upper()
    time_scheme = str(cfg["time_scheme"]).upper()
    if not cfg.get("verbose", True):
        # Mode concis : une seule ligne avec les infos principales
        tag = format_condition_tag(cfg)
        print(
            f"[{cfg['case']}] {tag} | h={mesh.metadata.get('h', 'n/a')} "
            f"| {cfg['flux']}/{reconstruction}/{time_scheme} -> {out_dirs['res']}"
        )
        return
    print("\n" + "-" * 78)
    print("CONFIGURATION SIMULATION")
    print("-" * 78)
    print(f"Cas : {cfg['case']}")
    print(f"Mesh path : {cfg['mesh_path']} (h={mesh.metadata.get('h', 'n/a')})")
    print("-" * 78)
    print("Physique")
    if str(cfg.get("case", "")).lower() in ("diamond", "naca0012", "rae2822", "onerad"):
        print(f"  Mach : {cfg['Mach']}, AoA : {float(cfg.get('aoa', 0.0)):.1f}°, gamma={cfg['gamma']}, p_inf={cfg['p_inf']}, rho_inf={cfg['rho_inf']}")
    else:
        print(f"  Mach : {cfg['Mach']}, gamma={cfg['gamma']}, p_inf={cfg['p_inf']}, rho_inf={cfg['rho_inf']}")
    print("-" * 78)
    print("Solveur")
    print(f"  scheme FVM : {cfg['flux']}, {reconstruction}, {time_scheme}")
    print(f"  CFL : {cfg['CFL']}, tf={cfg['tf']}")
    print(f"  stationarity_threshold={cfg.get('stationarity_threshold', 'n/a')} | check_every={cfg.get('stationarity_check_every', 'n/a')}")
    try:
        import jax
        print(f"  Parallélisation JAX : {jax.default_backend()} (device count: {jax.device_count()})")
    except Exception:
        pass
    print("-" * 78)
