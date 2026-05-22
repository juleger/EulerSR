import numpy as np
import jax.numpy as jnp

from utils import build_config_from_cli, export_snapshot, load_mesh, print_config, setup_dirs

import jax_fvm.src.solvers.helper as helper
import jax_fvm.src.solvers.Euler.Euler as Euler
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
    "reconstruction": "constant",  # constant | muscl
    "CFL": 0.25,  # Nombre de Courant (<1 en explicite). Ici dt est fixe pour simplifier, donc il faut une marge sur la CFL (si l'écoulement accélère par ex)
    "tf": 2.0,  # Temps final de la simulation

    # Fichier de maillage
    "mesh_path": "meshes/diamond/diamond_mesh0.05.npy",

    # Export
    "export": {
        "results": True,  # Exporter les data de solutions (npy)
        "figures": True,  # Exporter les figures (png)
        "graph": False,  # Exporter la simulation sous forme de graph (npz)
        "n_snaps": 1,  # Nombre de snapshots
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

    snap_steps = set(np.rint(np.linspace(1, N, n_snaps)).astype(int).tolist())
    snap_steps.add(N)
    if n_snaps == 1:
        snap_steps = {N}

    print(f"    dt={float(dt):.2e}, Nt={N}, n_snaps={n_snaps}")

    t = 0.0
    W_snapshots = {}
    ct = 0
    
    print("\nRésolution numérique en cours...")
    start_time = time.time()
    for n in range(1, N + 1):
        W = fn(W, mesh, dt, **kw)
        t += float(dt)
        if n in snap_steps:
            export_snapshot(W, mesh, t, cfg, out_dirs, helper)
            W_snapshots[round(t, 6)] = np.array(W)
            ct += 1
        # On affiche la progression tous les 10%
        if n % max(1, N // 10) == 0 or n == N:
            print(f"    step {n}/{N}  ({100 * n / N:.1f}%)")

    end_time = time.time()
    print("\n" + "-" * 78)
    print(f"Simulation terminée en {end_time - start_time:.2f}s ({ct} snapshots)")

    if exp["graph"]:
        export_graph(mesh, W_snapshots, inlet, save_path=str(out_dirs["res"] / "graph.npz"))

    if cfg["case"] == "diamond":
        U_inf = cfg["Mach"] * np.sqrt(cfg["gamma"] * cfg["p_inf"] / cfg["rho_inf"])
        C_D = helper.get_drag_coefficient(W=W, mesh=mesh, rho_inf=cfg["rho_inf"], U_inf=U_inf, L_ref=mesh.metadata["obstacle_length"])
        print(f"Coefficient de trainée C_D = {C_D:.4f}")
    return W

if __name__ == "__main__":
    
    CFG = build_config_from_cli(CFG)
    mesh = load_mesh(CFG)
    out_dirs = setup_dirs(CFG, mesh)
    print_config(CFG, mesh, out_dirs)
    W, inlet = initialize(mesh, CFG)
    run(W, mesh, inlet, CFG, out_dirs)