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

FIDELITY_PRESETS = {
    "lf": {"flux": "Rusanov", "reconstruction": "constant", "time_scheme": "EE"},
    "hf": {"flux": "HLLC", "reconstruction": "MUSCL", "time_scheme": "SRK2"},
}

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
    parser.add_argument("--aoa", type=float, default=default_cfg.get("aoa", 0.0), help="Angle d'attaque en degrés (utilisé pour le cas diamond)")
    parser.add_argument("--fidelity", choices=("off", "lf", "hf"), default=default_cfg.get("fidelity", "off"), help="Preset fidelity mode (off/lf/hf)")
    parser.add_argument("--stationarity-threshold", type=float, default=default_cfg.get("stationarity_threshold", 1e-6), help="Seuil d'arrêt sur le résidu relatif")
    parser.add_argument("--stationarity-check-every", type=int, default=default_cfg.get("stationarity_check_every", 25), help="Nombre de pas entre deux tests de stationnarité")
    args = parser.parse_args()

    cfg = json.loads(json.dumps(default_cfg))
    cfg["case"] = args.case
    cfg["Mach"] = args.mach
    cfg["mesh_path"] = args.mesh_path
    cfg["time_scheme"] = "SRK2" if args.time_scheme == "SSP_RK2" else args.time_scheme
    cfg["flux"] = args.flux
    cfg["reconstruction"] = "MUSCL" if str(args.reconstruction).lower() == "muscl" else args.reconstruction
    cfg["aoa"] = float(args.aoa)
    cfg["fidelity"] = str(args.fidelity).lower()
    cfg["stationarity_threshold"] = float(args.stationarity_threshold)
    cfg["stationarity_check_every"] = max(1, int(args.stationarity_check_every))
    if cfg["fidelity"] in ("lf", "hf"):
        cfg["dataset_mode"] = "fidelity"

        cfg.update(FIDELITY_PRESETS[cfg["fidelity"]])
    else:
        cfg["dataset_mode"] = None
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
    condition_tag = format_condition_tag(cfg)

    if cfg.get("dataset_mode") == "fidelity":
        side = str(cfg.get("fidelity", "off")).upper()
        return f"{side}_{condition_tag}"
    flux = str(cfg["flux"]).upper()
    reconstruction = str(cfg["reconstruction"]).upper()
    time_scheme = str(cfg["time_scheme"]).upper()
    prefix = ""
    return f"{prefix}{condition_tag}_{flux}_{reconstruction}_{time_scheme}_t{t:.2f}"


def format_condition_tag(cfg):
    mach = f"M{float(cfg.get('Mach')):.2f}"
    if str(cfg.get("case", "")).lower() == "diamond":
        aoa = float(cfg.get("aoa", 0.0))
        return f"AOA{aoa:.2f}_{mach}"
    return mach


def format_sample_id(cfg):
    return format_condition_tag(cfg)


def format_subtitle(cfg, mesh, t):
    reconstruction = str(cfg["reconstruction"]).upper()
    time_scheme = str(cfg["time_scheme"]).upper()
    if str(cfg.get("case", "")).lower() == "diamond":
        return (
            f"t={t:.2f}s, M={cfg['Mach']}, AoA={float(cfg.get('aoa', 0.0)):.2f}deg, "
            f"h={mesh.metadata.get('h', 'n/a')} | Solver : {cfg['flux']}, {reconstruction}, {time_scheme}"
        )
    return (
        f"t={t:.2f}s, M={cfg['Mach']}, h={mesh.metadata.get('h', 'n/a')} "
        f"| Solver : {cfg['flux']}, {reconstruction}, {time_scheme}"
    )

def setup_dirs(cfg, mesh):
    # Préparation des dossiers de sortie
    h_dir = format_h(mesh)
    dirs = {
        "fig": repo_root / "figures" / cfg["case"] / h_dir,
        "res": repo_root / "results" / cfg["case"] / h_dir,
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    # with open(dirs["res"] / "config.json", "w") as f:
    #     json.dump(cfg, f, indent=2)
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
    if str(cfg.get("case", "")).lower() == "diamond":
        print(f"  AoA : {cfg.get('aoa', 0.0)} deg")
    print("-" * 78)
    print("Solveur")
    reconstruction = str(cfg["reconstruction"]).upper()
    time_scheme = str(cfg["time_scheme"]).upper()
    print(f"  scheme FVM : {cfg['flux']}, {reconstruction}, {time_scheme}")
    print(f"  CFL : {cfg['CFL']}, tf={cfg['tf']}")
    print(f"  fidelity : {cfg.get('fidelity', 'off')} | stationarity_threshold={cfg.get('stationarity_threshold', 'n/a')} | check_every={cfg.get('stationarity_check_every', 'n/a')}")
    print("-" * 78)


def export_snapshot(W, mesh, t, cfg, out_dirs, helper, inlet=None):
    # Exporte les résultats sous forme de figures ou array numpy selon la config

    exp, gamma = cfg["export"], cfg["gamma"]
    snapshot_name = format_snapshot_name(cfg, t)
    diagnostics = None
    is_dataset = cfg.get("dataset_mode") == "fidelity"

    if exp["results"]:
        if not is_dataset:
            np.save(str(out_dirs["res"] / f"{snapshot_name}.npy"), np.array(W))

    if exp.get("bundle", True):
        prims = helper.getPrimitive(W, gamma=gamma, M=1.0)
        W_L = np.repeat(np.asarray(W)[..., None, :], 3, axis=-2)
        W_R = np.asarray(W)[np.asarray(mesh.neighbors)]
        W_R = helper.BC_state(W_R, W_L, mesh, value=cfg.get("inlet_state", np.array([1.0, 1.0, 0.0, 1.0])))
        prims_L = np.repeat(np.asarray(prims)[..., None, :], 3, axis=-2)
        prims_R = helper.getPrimitive(W_R, gamma=gamma, M=1.0)
        prims_grad = helper.getgradientLSQ(prims_L, prims_R, mesh)
        mach_field = helper.get_mach_number(prims, gamma=gamma)
        schlieren_field = helper.get_schlieren_field(W, mesh, gamma=gamma, M=1.0)
        if is_dataset:
            dataset_root = repo_root / 'dataset' / cfg['case']
            sample_id = format_sample_id(cfg)
            sample_dir = dataset_root / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            bundle_path = sample_dir / f"{snapshot_name}.npz"
        else:
            bundle_path = out_dirs["res"] / f"bundle_{snapshot_name}.npz"
        np.savez_compressed(
            str(bundle_path),
            node_pos=np.asarray(mesh.barycenter, dtype=np.float32),
            node_area=np.asarray(mesh.area, dtype=np.float32),
            conservatives=np.asarray(W, dtype=np.float32),
            primitives=np.asarray(prims, dtype=np.float32),
            primitives_grad=np.asarray(prims_grad, dtype=np.float32),
            mach=np.asarray(mach_field, dtype=np.float32),
            schlieren=np.asarray(schlieren_field, dtype=np.float32),
            time=np.asarray([float(t)], dtype=np.float32),
            mach_in=np.asarray([float(cfg["Mach"])], dtype=np.float32),
            aoa_in=np.asarray([float(cfg.get("aoa", 0.0))], dtype=np.float32),
            fidelity=np.asarray([str(cfg.get("fidelity", "off"))]),
        )
        if cfg.get("dataset_mode") == "fidelity":
            side = str(cfg.get("fidelity", "off")).lower()
            sample_id = format_sample_id(cfg)
            dataset_root = repo_root / 'dataset' / cfg['case']
            dataset_root.mkdir(parents=True, exist_ok=True)
            manifest_path = dataset_root / "dataset_labels.json"
            manifest = {}
            try:
                if manifest_path.exists():
                    manifest = json.loads(manifest_path.read_text())
            except Exception:
                manifest = {}
            entry = manifest.get(sample_id, {})
            try:
                rel_bundle = str(bundle_path.relative_to(repo_root)) if bundle_path.exists() else str(bundle_path)
            except Exception:
                rel_bundle = str(bundle_path)
            entry[side] = rel_bundle
            entry[f"{side}_mesh"] = str(cfg.get('mesh_path', ''))
            entry['mesh'] = str(cfg.get('mesh_path', ''))
            entry['mach'] = float(cfg.get('Mach'))
            if str(cfg.get("case", "")).lower() == "diamond":
                entry['aoa'] = float(cfg.get('aoa', 0.0))
            entry['lf_scheme'] = "Rusanov/constant/EE"
            entry['hf_scheme'] = "HLLC/MUSCL/SRK2"
            manifest[sample_id] = entry
            # Ecriture atomique
            try:
                tmp_path = manifest_path.with_suffix('.tmp')
                with open(tmp_path, 'w') as mf:
                    json.dump(manifest, mf, indent=2)
                tmp_path.replace(manifest_path)
            except Exception:
                try:
                    with open(manifest_path, 'w') as mf:
                        json.dump(manifest, mf, indent=2)
                except Exception:
                    pass

    if exp["figures"]:
        prims = helper.getPrimitive(W, gamma=gamma, M=1.0)
        Mach_field = helper.get_mach_number(prims, gamma=gamma)
        subtitle = format_subtitle(cfg, mesh, t)

        if exp.get("bundle", True):
            # Bundle mode keeps only the Mach figure to reduce dataset creation overhead.
            mesh.plot_solution(Mach_field, labels=r"$M$", filename=out_dirs["fig"] / f"M_{snapshot_name}.png",
                dpi=400, title="Mach local", subtitle=subtitle, cmap="viridis")
        else:
            cmaps = ["inferno", "viridis", "plasma", "viridis", HOT_CMAP_CROPPED, "Greys"]
            title_map = {
                "rho": "Masse volumique $\\rho$",
                "u": "Vitesse $u$",
                "v": "Vitesse $v$",
                "p": "Pression $p$",
                "M": "Mach local $M$",
                "S": "Schlieren (log(1+|\\nabla M|))",
            }
            fields = ["rho", "u", "v", "p", "M", "S"]
            texs = [r"$\rho$", r"$u$", r"$v$", r"$p$", r"$M$", r"$S$"]
            schlieren_field = helper.get_schlieren_field(W, mesh, gamma=gamma, M=1.0)
            for i, s in enumerate(fields):
                tex = texs[i]
                title = title_map.get(s, tex)
                field = Mach_field if s == "M" else schlieren_field if s == "S" else prims[..., i]
                mesh.plot_solution(field, labels=tex, filename=out_dirs["fig"] / f"{s}_{snapshot_name}.png",
                    dpi=400, title=title, subtitle=subtitle, cmap=cmaps[i % len(cmaps)])

    diagnostics = None
    if cfg["case"] == "bump" and inlet is not None:
        diagnostics = helper.get_bump_diagnostics(W, mesh, inlet, gamma=gamma, M=1.0)
        wall_profile = diagnostics["wall_profile"]
        wall_profile_data = np.column_stack([
            wall_profile["s"],
            wall_profile["x"],
            wall_profile["y"],
            wall_profile["mach"],
        ])

        if exp["results"]:
            np.save(str(out_dirs["res"] / f"wall_profile_{snapshot_name}.npy"), wall_profile_data)

        if exp["figures"] and wall_profile["s"].size > 0:
            subtitle = format_subtitle(cfg, mesh, t)
            mesh.plot_profile(wall_profile["s"], wall_profile["mach"], labels=r'$M_{wall}$', filename=out_dirs["fig"] / f"Mwall_{snapshot_name}.png",
                dpi=400, title="Profil de Mach de paroi", subtitle=subtitle, xlabel=r"Abscisse curviligne $s$")

    return diagnostics


def build_run_summary(cfg, mesh, cd, cl, delta_s, wall_time_s, stationarity_rel=None, converged=None, stopping_step=None):
    return {
        "case": cfg["case"],
        "h": mesh.metadata.get("h"),
        "n_cells": int(np.asarray(mesh.tris).shape[0]) if mesh.tris is not None else None,
        "cd": float(cd) if cd is not None else None,
        "cl": float(cl) if cl is not None else None,
        "deltaS": float(delta_s) if delta_s is not None else None,
        "stationarity_rel": float(stationarity_rel) if stationarity_rel is not None else None,
        "converged": bool(converged) if converged is not None else None,
        "stopping_step": int(stopping_step) if stopping_step is not None else None,
        "stationarity_threshold": float(cfg.get("stationarity_threshold")) if cfg.get("stationarity_threshold") is not None else None,
        "stationarity_check_every": int(cfg.get("stationarity_check_every")) if cfg.get("stationarity_check_every") is not None else None,
        "wall_time_s": float(wall_time_s),
        "time_scheme": cfg["time_scheme"],
        "flux": cfg["flux"],
        "reconstruction": cfg["reconstruction"],
        "mach": cfg["Mach"],
        "mesh_path": cfg["mesh_path"],
        "fidelity": cfg.get("fidelity", "off"),
        "dataset_mode": cfg.get("dataset_mode"),
        "aoa": float(cfg.get("aoa", 0.0)),
    }


def export_run_summary(out_dirs, summary, snapshot_name):
    # Exporte le résumé de la simulation (temps de calcul, C_D, etc...) dans un fichier JSON
    out_dirs["res"].mkdir(parents=True, exist_ok=True)
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