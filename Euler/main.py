import os
from utils import configure_jax_cpu_runtime

#configure_jax_cpu_runtime() #facultatif car JAX par défaut est bien configuré

import jax
import numpy as np
import jax.numpy as jnp
from utils import build_config_from_cli, build_run_summary, export_snapshot, export_run_summary, load_mesh, print_config, setup_dirs

import jax_fvm.src.helper as helper
import jax_fvm.src.euler_solver as Euler
from graph import export_graph
import time


CFG = {
    # Configuration globale, modifiable via CLI

    "case": "diamond",  # "bump" | "diamond"

    # Physique
    "rho_inf": 1.0,  # Densité amont
    "p_inf": 1.0,  # Pression amont
    "Mach": 2.0,  # Nombre de Mach du flux entrant
    "gamma": 1.4,  # Ratio de chaleur spécifique (diatomique ici)

    # Solveur
    "time_scheme": "SSP_RK2",  # EE | RK2 | RK4 | SSP_RK2
    "flux": "HLLC",  # Rusanov | Roe | HLLC
    "reconstruction": "muscl",  # constant | muscl
    "CFL": 0.25,  # Nombre de Courant (<1 en explicite). Ici dt est fixe pour simplifier, donc il faut une marge sur la CFL (si l'écoulement accélère par ex)
    "tf": 2.0,  # Temps final de la simulation

    # Fichier de maillage
    "mesh_path": "meshes/diamond/diamond_h0.05.npy",

    # Export
    "export": {
        "results": True,  # Exporter les data de solutions (npy)
        "figures": True,  # Exporter les figures (png)
        "graph": False,  # Exporter la simulation sous forme de graph (npz)
        "summary": False,  # Exporter le résumé machine-readable (json)
           "n_snaps": 1,  # Nombre de snapshots
           "cmap_crop": {"hot": [0.0, 0.8]},
    },
}

def initialize(mesh, cfg):
    # Initialisation des conditions initiales
    rho_inf, p_inf = cfg["rho_inf"], cfg["p_inf"]
    gamma, Mach = cfg["gamma"], cfg["Mach"]
    c_inf = float(jnp.sqrt(gamma * p_inf / rho_inf))
    prim0 = jnp.array([rho_inf, Mach * c_inf, 0.0, p_inf])
    W = helper.getConserved(jnp.repeat(prim0[None], len(mesh.area), axis=0), gamma=gamma, M=1.0)
    inlet = helper.getConserved(prim0[None], gamma=gamma, M=1.0)[0]
    return W, inlet


def run(W, mesh, inlet, cfg, out_dirs):
    # Résolution numérique d'Euler compressible

    # Schéma en temps
    fn = {"EE": Euler.time_step_Euler, "RK2": Euler.time_step_RK2, "RK4": Euler.time_step_RK4,
          "SSP_RK2": Euler.time_step_RK2_SSP}[cfg["time_scheme"]]

    # kwargs pour le solver (flux, reconstruction, etc...)
    kw = dict(gamma=cfg["gamma"], M=1.0, reconstruction=cfg["reconstruction"],
              flux=cfg["flux"], value=inlet, entropy=False)
    exp = cfg["export"]

    # Calcul dt selon CFL (fixe dans ce cas)
    dt = helper.get_dt(W, mesh, CFL=cfg["CFL"], gamma=cfg["gamma"], M=1.0)
    N = int(cfg["tf"] / dt) + 1
    n_snaps = max(1, int(exp["n_snaps"]))

    snap_steps = sorted(set(np.rint(np.linspace(1, N, n_snaps)).astype(int).tolist()) | {N})
    if n_snaps == 1:
        snap_steps = [N]

    print(f"    dt={float(dt):.2e}, Nt={N}, n_snaps={n_snaps}")

    # Warm-up pour compilation JIT
    print("\nCompilation JIT (warm-up)...")
    W = fn(W, mesh, dt, **kw)
    jax.block_until_ready(W) 

    t = 0.0
    W_snapshots = {}
    ct = 0

    print("\nRésolution numérique en cours...")
    start_time = time.time()

    def scan_body(W, _):
        return fn(W, mesh, dt, **kw), None
    
    W, _ = jax.lax.scan(scan_body, W, None, length = N)
    export_snapshot(W, mesh, CFG["tf"], cfg, out_dirs, helper)
    W_snapshots[round(CFG["tf"], 6)] = np.array(W)

    end_time = time.time()
    wall_time_s = end_time - start_time
    print("\n" + "-" * 78)
    print(f"Simulation terminée en {wall_time_s:.2f}s ({ct} snapshots)")

    if exp["graph"]:
        export_graph(mesh, W_snapshots, inlet, save_path=str(out_dirs["res"] / "graph.npz"))

    summary = None
    if cfg["case"] == "diamond":
        U_inf = cfg["Mach"] * np.sqrt(cfg["gamma"] * cfg["p_inf"] / cfg["rho_inf"])
        C_D = helper.get_drag_coefficient(W=W, mesh=mesh, rho_inf=cfg["rho_inf"], U_inf=U_inf, L_ref=mesh.metadata["obstacle_length"])
        print(f"Coefficient de trainée C_D = {C_D:.4f}")
        summary = build_run_summary(cfg, mesh, C_D, wall_time_s)

    if exp.get("summary", True) and summary is not None:
        export_run_summary(out_dirs, summary)
    return W

if __name__ == "__main__":

    CFG = build_config_from_cli(CFG)
    mesh = load_mesh(CFG)
    out_dirs = setup_dirs(CFG, mesh)
    print_config(CFG, mesh, out_dirs)
    W, inlet = initialize(mesh, CFG)
    run(W, mesh, inlet, CFG, out_dirs)