import os
from utils import configure_jax_cpu_runtime

#configure_jax_cpu_runtime() #facultatif car JAX par défaut est bien configuré

import jax
import numpy as np
import jax.numpy as jnp
from utils import build_config_from_cli, build_run_summary, export_snapshot, export_run_summary, format_snapshot_name, load_mesh, print_config, setup_dirs

import jax_fvm.src.helper as helper
import jax_fvm.src.euler_solver as Euler
from graph import export_graph
import time


CFG = {
    # Configuration globale, modifiable via CLI

    "case": "bump",  # "bump" | "diamond"

    # Physique
    "rho_inf": 1.0,  # Densité amont
    "p_inf": 1.0,  # Pression amont
    "Mach": 1.4,  # Nombre de Mach du flux entrant
    "gamma": 1.4,  # Ratio de chaleur spécifique (diatomique ici)

    # Solveur
    "time_scheme": "SRK2",  # EE | RK2 | RK4 | SRK2 (SSP_RK2)
    "flux": "HLLC",  # Rusanov | Tadmor | AUSM+ | Roe | HLLC
    "reconstruction": "MUSCL",  # constant | MUSCL
    "CFL": 0.25,  # Nombre de Courant (<1 en explicite). Ici dt est fixe pour simplifier, donc il faut une marge sur la CFL (si l'écoulement accélère par ex)
    "tf": 5.0,  # Temps final de la simulation (dépend du régime pour atteindre l'état stationnaire : plus rapide en supersonique que subsonique)

    # Fichier de maillage
    "mesh_path": "meshes/bump/bump_h0.05.npy",

    # Export
    "export": {
        "results": True,  # Exporter les data de solutions (npy)
        "figures": True,  # Exporter les figures (png)
        "graph": False,  # Exporter la simulation sous forme de graph (npz)
        "summary": True,  # Exporter le résumé machine-readable (json)
           "n_snaps": 1,  # Nombre de snapshots
           "cmap_crop": {"hot": [0.0, 0.8]},
    },
}

def initialize(mesh, cfg):
    # Initialisation des conditions initiales
    rho_inf, p_inf = cfg["rho_inf"], cfg["p_inf"]
    gamma, Mach = cfg["gamma"], cfg["Mach"]
    c_inf = (gamma * p_inf / rho_inf) ** 0.5
    prim0 = jnp.array([rho_inf, Mach * c_inf, 0.0, p_inf])
    W = helper.getConserved(jnp.repeat(prim0[None], len(mesh.area), axis=0), gamma=gamma, M=1.0)
    inlet = helper.getConserved(prim0[None], gamma=gamma, M=1.0)[0]
    return W, inlet


def run(W, mesh, inlet, cfg, out_dirs):
    # Résolution numérique d'Euler compressible
    W_initial = W

    # Schéma en temps
    scheme = str(cfg["time_scheme"]).upper()
    fn = {"EE": Euler.time_step_Euler, "RK2": Euler.time_step_RK2, "RK4": Euler.time_step_RK4,
          "SRK2": Euler.time_step_RK2_SSP, "SSP_RK2": Euler.time_step_RK2_SSP}[scheme]

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

    print("\n" + "-" * 78)
    print("Parallélisation JAX :")
    print(f"  Backend : {jax.default_backend()}")
    print(f"  Device(s) : {jax.devices()}")
    print("-" * 78)
    # Warm-up pour compilation JIT
    print("\nCompilation JIT (warm-up)...")
    W = fn(W, mesh, dt, **kw)
    jax.block_until_ready(W) 

    W_snapshots = {}

    print("\nRésolution numérique en cours...")
    start_time = time.time()

    def advance(W, n_steps):
        def scan_body(W, _):
            return fn(W, mesh, dt, **kw), 0.0
        W, _ = jax.lax.scan(scan_body, W, None, length=n_steps)
        return W

    current_step = 0
    for snap_step in snap_steps:
        n_steps = int(snap_step - current_step)
        if n_steps <= 0:
            continue
        W = advance(W, n_steps)
        current_step = int(snap_step)

        if exp["results"] or exp["figures"]:
            t_snap = float(current_step * float(dt))
            export_snapshot(W, mesh, t_snap, cfg, out_dirs, helper, inlet=inlet)
            W_snapshots[round(t_snap, 6)] = np.array(W)
            print(f"    Snap exporté à t={t_snap:.2f}s ({current_step}/{N} steps)")
    
    end_time = time.time()
    wall_time_s = end_time - start_time
    snapshot_name = format_snapshot_name(cfg, current_step * float(dt))
    print("\n" + "-" * 78)
    print(f"Simulation terminée en {wall_time_s:.2f}s ({n_snaps} snapshots)")
    if exp["graph"]:
        export_graph(mesh, W_snapshots, inlet, save_path=str(out_dirs["res"] / "graph.npz"))

    final_residual = Euler.residual(W, mesh, **kw)
    stationarity_rel = float(jnp.linalg.norm(dt * final_residual) / (jnp.linalg.norm(W) + 1e-16))
    print(f"Résidu final relatif ||dt·R(W)||/||W|| = {stationarity_rel:.3e}")

    summary = None
    if cfg["case"] in ("diamond", "bump"):
        U_inf = cfg["Mach"] * np.sqrt(cfg["gamma"] * cfg["p_inf"] / cfg["rho_inf"])
        C_D = helper.get_drag_coefficient(W=W, mesh=mesh, rho_inf=cfg["rho_inf"], U_inf=U_inf, L_ref=mesh.metadata["obstacle_length"])
        C_L = helper.get_lift_coefficient(W=W, mesh=mesh, rho_inf=cfg["rho_inf"], U_inf=U_inf, L_ref=mesh.metadata["obstacle_length"])
        delta_S = helper.get_entropy_creation(W_initial, W, mesh, gamma=cfg["gamma"])
        bump_diag = helper.get_bump_diagnostics(W, mesh, inlet, gamma=cfg["gamma"], M=1.0)
        delta_m = bump_diag.get("deltaM")
        delta_m_rel = bump_diag.get("deltaM_rel")
        mass_in = bump_diag.get("mass_in")
        mass_out = bump_diag.get("mass_out")
        max_grad_p = bump_diag.get("max_grad_p")
        print(f"Coefficient de trainée C_D = {C_D:.4f}")
        print(f"Coefficient de portance C_L = {C_L:.4f}")
        print(f"Création totale d'entropie deltaS = {delta_S:.6e}")
        if delta_m is not None:
            print(f"Bilan de masse deltaM = {delta_m:.6e} (rel={delta_m_rel:.6e})")
        if max_grad_p is not None:
            print(f"Gradient max de pression |grad p|_max = {max_grad_p:.6e}")
        summary = build_run_summary(cfg, mesh, C_D, C_L, delta_S, wall_time_s, delta_m=delta_m,
            mass_in=mass_in, mass_out=mass_out, max_pressure_gradient=max_grad_p, delta_m_rel=delta_m_rel,
            stationarity_rel=stationarity_rel)

    if exp.get("summary", True) and summary is not None:
        export_run_summary(out_dirs, summary, snapshot_name)
    return W

if __name__ == "__main__":

    CFG = build_config_from_cli(CFG)
    mesh = load_mesh(CFG)
    out_dirs = setup_dirs(CFG, mesh)
    print_config(CFG, mesh, out_dirs)
    W, inlet = initialize(mesh, CFG)
    run(W, mesh, inlet, CFG, out_dirs)