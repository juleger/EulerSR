from pathlib import Path
import json
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    from config import format_snapshot_name, format_subtitle
except ModuleNotFoundError:
    from euler.config import format_snapshot_name, format_subtitle

base_hot = mpl.cm.get_cmap("hot")
hot_colors = base_hot(np.linspace(0.0, 0.8, 256))
HOT_CMAP_CROPPED = mpl.colors.LinearSegmentedColormap.from_list("hot_cropped", hot_colors)

repo_root = Path(__file__).resolve().parents[1]


def export_snapshot(W, mesh, t, cfg, out_dirs, helper, inlet=None):
    # Exporte les résultats sous forme de figures ou array numpy selon la config

    exp, gamma = cfg["export"], cfg["gamma"]
    snapshot_name = format_snapshot_name(cfg, t)
    diagnostics = None
    figure_fields = normalize_figure_fields(exp.get("figures", False))

    if exp["results"]:
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
        bundle_path = out_dirs["res"] / f"{snapshot_name}.npz"
        print(f"Bundle export : {bundle_path}")

        mesh_path_val = str(cfg.get('mesh_path', ''))
        n_cells_val = int(np.asarray(mesh.tris).shape[0]) if getattr(mesh, 'tris', None) is not None else -1
        h_val = mesh.metadata.get('h', float('nan'))
        flux_val = str(cfg.get('flux', ''))
        recon_val = str(cfg.get('reconstruction', ''))
        time_scheme_val = str(cfg.get('time_scheme', ''))
        np.savez_compressed(
            str(bundle_path),
            node_pos=np.asarray(mesh.barycenter, dtype=np.float32),
            node_area=np.asarray(mesh.area, dtype=np.float32),
            conservatives=np.asarray(W, dtype=np.float32),
            primitives=np.asarray(prims, dtype=np.float32),
            primitives_grad=np.asarray(prims_grad, dtype=np.float32),
            mach=np.asarray(mach_field, dtype=np.float32),
            time=np.asarray([float(t)], dtype=np.float32),
            mach_in=np.asarray([float(cfg["Mach"])], dtype=np.float32),
            aoa_in=np.asarray([float(cfg.get("aoa", 0.0))], dtype=np.float32),
            case=np.asarray([str(cfg.get("case", ""))]),
            mesh_path=np.asarray([mesh_path_val]),
            n_cells=np.asarray([int(n_cells_val)]),
            h=np.asarray([float(h_val)]),
            flux=np.asarray([flux_val]),
            reconstruction=np.asarray([recon_val]),
            time_scheme=np.asarray([time_scheme_val]),
        )

    if figure_fields:
        prims = helper.getPrimitive(W, gamma=gamma, M=1.0)
        subtitle = format_subtitle(cfg, mesh, t)
        cmaps = {
            "rho": "inferno",
            "u": "viridis",
            "v": "plasma",
            "p": "viridis",
            "M": "viridis",
            "S": HOT_CMAP_CROPPED,
        }
        title_map = {
            "rho": "Masse volumique $\\rho$",
            "u": "Vitesse $u$",
            "v": "Vitesse $v$",
            "p": "Pression $p$",
            "M": "Mach local $M$",
            "S": "Schlieren (log(1+|\\nabla M|))",
        }
        labels = {
            "rho": r"$\rho$",
            "u": r"$u$",
            "v": r"$v$",
            "p": r"$p$",
            "M": r"$M$",
            "S": r"$S$",
        }
        field_lookup = {
            "rho": prims[..., 0],
            "u": prims[..., 1],
            "v": prims[..., 2],
            "p": prims[..., 3],
            "M": helper.get_mach_number(prims, gamma=gamma),
            "S": helper.get_schlieren_field(W, mesh, gamma=gamma, M=1.0),
        }
        for field_name in figure_fields:
            if field_name not in field_lookup:
                raise ValueError(f"Unknown figure field '{field_name}'. Expected one of: rho, u, v, p, M, S")
            fig_path = out_dirs["fig"] / f"{field_name}_{snapshot_name}.png"
            print(f"Figure export : {fig_path}")
            mesh.plot_solution(
                field_lookup[field_name],
                labels=labels[field_name],
                filename=fig_path,
                dpi=400,
                title=title_map[field_name],
                subtitle=subtitle,
                cmap=cmaps[field_name],
            )

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
            fig_path = out_dirs["fig"] / f"Mwall_{snapshot_name}.png"
            print(f"Figure export : {fig_path}")
            mesh.plot_profile(wall_profile["s"], wall_profile["mach"], labels=r'$M_{wall}$', filename=fig_path,
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
        "aoa": float(cfg.get("aoa", 0.0)),
    }


def export_run_summary(out_dirs, summary, snapshot_name):
    # Exporte le résumé de la simulation (temps de calcul, C_D, etc...) dans un fichier JSON
    out_dirs["res"].mkdir(parents=True, exist_ok=True)
    path = out_dirs["res"] / f"summary_{snapshot_name}.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written : {path}")


def normalize_figure_fields(figures_value):
    if figures_value in (False, None):
        return []
    if figures_value is True:
        return ["M"]
    if isinstance(figures_value, str):
        value = figures_value.strip()
        if not value:
            return []
        if value.lower() in ("false", "none", "off"):
            return []
        return [value]
    if isinstance(figures_value, (list, tuple)):
        return [str(item).strip() for item in figures_value if str(item).strip()]
    return [str(figures_value).strip()] if str(figures_value).strip() else []
