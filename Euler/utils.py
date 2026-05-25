from argparse import ArgumentParser
from pathlib import Path
import json
import sys
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

base_hot = mpl.cm.get_cmap("hot")
hot_colors = base_hot(np.linspace(0.0, 0.8, 256))
HOT_CMAP_CROPPED = mpl.colors.LinearSegmentedColormap.from_list("hot_cropped", hot_colors)
repo_root = Path(__file__).resolve().parents[1]

from jax_fvm.src.mesh import Mesh


def build_config_from_cli(default_cfg):
    # Parse les arguments de la ligne de commande et retourne une config dict
    parser = ArgumentParser(description="Simulation supersonique avec Euler 2D")
    parser.add_argument("--case", choices=("bump", "diamond"), default=default_cfg["case"], help="Cas test : bump ou diamond")
    parser.add_argument("--mach", type=float, default=default_cfg["Mach"], help="Nombre de Mach du flux entrant")
    parser.add_argument("--mesh-path", type=str, default=default_cfg["mesh_path"], help="Chemin vers le fichier de maillage (.npy)")
    parser.add_argument("--time-scheme", choices=("EE", "RK2", "RK4", "SRK2", "SSP_RK2"), default=default_cfg["time_scheme"], help="Schéma temporel")
    parser.add_argument("--flux", choices=("Rusanov", "Tadmor", "AUSM", "Roe", "HLLC"), default=default_cfg["flux"], help="Solveur de flux numérique")
    parser.add_argument("--reconstruction", choices=("constant", "MUSCL", "muscl"), default=default_cfg["reconstruction"], help="Type de reconstruction")
    args = parser.parse_args()

    cfg = json.loads(json.dumps(default_cfg))
    cfg["case"] = args.case
    cfg["Mach"] = args.mach
    cfg["mesh_path"] = args.mesh_path
    cfg["time_scheme"] = "SRK2" if args.time_scheme == "SSP_RK2" else args.time_scheme
    cfg["flux"] = args.flux
    cfg["reconstruction"] = "MUSCL" if str(args.reconstruction).lower() == "muscl" else args.reconstruction
    return cfg


def load_mesh(cfg):
    # Création de l'objet Mesh à partir du fichier de maillage spécifié dans la config
    if cfg["mesh_path"] is None:
        raise ValueError("Le chemin du maillage doit être spécifié via --mesh-path ou dans la config par défaut.")

    mesh = Mesh()
    mesh.load_mesh(cfg["mesh_path"])
    mesh.print_statistics()
    return mesh


def format_h(mesh):
    h = mesh.metadata.get("h", None)
    if h is None:
        return "h"
    return f"h{float(h):.4f}".rstrip("0").rstrip(".")


def format_snapshot_name(cfg, t):
    mach = f"M{cfg['Mach']:.1f}"
    flux = str(cfg["flux"]).upper()
    reconstruction = str(cfg["reconstruction"]).upper()
    time_scheme = str(cfg["time_scheme"]).upper()
    if time_scheme == "SSP_RK2":
        time_scheme = "SRK2"
    return f"{mach}_{flux}_{reconstruction}_{time_scheme}_t{t:.2f}"

def setup_dirs(cfg, mesh):
    # Préparation des dossiers de sortie
    h_dir = format_h(mesh)
    dirs = {
        "fig": repo_root / "figures" / cfg["case"] / h_dir,
        "res": repo_root / "results" / cfg["case"] / h_dir,
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    with open(dirs["res"] / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Output : {dirs['res']}")
    return dirs


def print_config(cfg, mesh, out_dirs):
    print("\n" + "-" * 78)
    print("CONFIGURATION SIMULATION")
    print("-" * 78)
    print(f"Cas : {cfg['case']}")
    print(f"Mesh path : {cfg['mesh_path']} (h={mesh.metadata.get('h', 'n/a')})")
    print(f"Output res : {out_dirs['res']}")
    print("-" * 78)
    print("Physique")
    print(f"  Mach : {cfg['Mach']}, gamma={cfg['gamma']}, p_inf={cfg['p_inf']}, rho_inf={cfg['rho_inf']}")
    print("-" * 78)
    print("Solveur")
    reconstruction = "MUSCL" if str(cfg["reconstruction"]).lower() == "muscl" else str(cfg["reconstruction"])
    time_scheme = "SRK2" if str(cfg["time_scheme"]).upper() == "SSP_RK2" else str(cfg["time_scheme"])
    print(f"  scheme FVM : {cfg['flux']}, {reconstruction}, {time_scheme}")
    print(f"  CFL : {cfg['CFL']}, tf={cfg['tf']}")
    print("-" * 78)


def export_snapshot(W, mesh, t, cfg, out_dirs, helper):
    # Exporte les résultats sous forme de figures ou array numpy selon la config

    exp, gamma = cfg["export"], cfg["gamma"]
    tag = f"t{t:.2f}"
    snapshot_name = format_snapshot_name(cfg, t)

    if exp["results"]:
        np.save(str(out_dirs["res"] / f"{snapshot_name}.npy"), np.array(W))
    if exp["figures"]:
        prims = helper.getPrimitive(W, gamma=gamma, M=1.0)
        Mach_field = helper.get_mach_number(prims, gamma=gamma)
        cmaps = ["inferno", "viridis", "plasma", "viridis", HOT_CMAP_CROPPED]
        title_map = {
            "rho": "Masse volumique $\\rho$",
            "u": "Vitesse $u$",
            "v": "Vitesse $v$",
            "p": "Pression $p$",
            "M": "Mach local $M$",
        }
        fields = ["rho", "u", "v", "p", "M"]
        texs = [r"$\rho$", r"$u$", r"$v$", r"$p$", r"$M$"]
        for i, s in enumerate(fields):
            tex = texs[i]
            title = title_map.get(s, tex)
            time_scheme = "SRK2" if str(cfg["time_scheme"]).upper() == "SSP_RK2" else str(cfg["time_scheme"])
            reconstruction = "MUSCL" if str(cfg["reconstruction"]).lower() == "muscl" else str(cfg["reconstruction"])
            subtitle = f"t={t:.2f}s, M={cfg['Mach']}, h={mesh.metadata.get('h', 'n/a')} | Solver : {cfg['flux']}, {reconstruction}, {time_scheme}"
            field = Mach_field if s == "M" else prims[:, i]
            mesh.plot_solution(field, labels=tex, filename=out_dirs["fig"] / f"{s}_{snapshot_name}.png",
                dpi=400, title=title, subtitle=subtitle, cmap=cmaps[i % len(cmaps)])


def build_run_summary(cfg, mesh, cd, cl, delta_s, wall_time_s):
    return {
        "case": cfg["case"],
        "h": mesh.metadata.get("h"),
        "cd": float(cd) if cd is not None else None,
        "cl": float(cl) if cl is not None else None,
        "deltaS": float(delta_s) if delta_s is not None else None,
        "wall_time_s": float(wall_time_s),
        "time_scheme": cfg["time_scheme"],
        "flux": cfg["flux"],
        "reconstruction": cfg["reconstruction"],
        "mach": cfg["Mach"],
        "mesh_path": cfg["mesh_path"],
    }


def export_run_summary(out_dirs, summary, snapshot_name):
    # Exporte le résumé de la simulation (temps de calcul, C_D, etc...) dans un fichier JSON
    path = out_dirs["res"] / f"{snapshot_name}.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written : {path}")

def configure_jax_cpu_runtime():
    # Configuration JAX pour parallélisation CPU efficace
    # Evite l'hyperthreading
    cpu_count = os.cpu_count() // 2 or 1 
    os.environ["XLA_FLAGS"] = f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={cpu_count}"
    os.environ["JAX_ENABLE_X64"] = "true"
    os.environ["OMP_NUM_THREADS"] = str(cpu_count)
    os.environ["MKL_NUM_THREADS"] = str(cpu_count)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_count)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_count)
    os.environ["JAX_NUM_THREADS"] = str(cpu_count)
    print(f"JAX configuré pour CPU avec {cpu_count} threads (hyperthreading désactivé)")