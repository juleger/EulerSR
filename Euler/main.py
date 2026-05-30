import os
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

    "case": "diamond",  # "bump" | "diamond"

    # Physique
    "rho_inf": 1.0,  # Densité amont
    "p_inf": 1.0,  # Pression amont
    "Mach": 1.2,  # Nombre de Mach du flux entrant
    "gamma": 1.4,  # Ratio de chaleur spécifique (diatomique ici)

    # Solveur
    "time_scheme": "SRK2",  # EE | RK2 | RK4 | SRK2 (SSP_RK2)
    "flux": "HLLC",  # Rusanov | Tadmor | AUSM+ | Roe | HLLC
    "reconstruction": "MUSCL",  # constant | MUSCL
    "aoa": 0.0,  # Angle d'attaque en degrés (utilisé pour diamond)
    "CFL": 0.6,  # Nombre de Courant (<1 en explicite). Ici dt est fixe pour simplifier, donc il faut une marge sur la CFL (si l'écoulement accélère par ex)
    "tf": 10.0,  # Temps final de la simulation (dépend du régime pour atteindre l'état stationnaire : plus rapide en supersonique que subsonique)
    "fidelity": "off",
    "stationarity_threshold": 2e-6, # Seuil de convergence pour la stationnarité (résidu relatif)
    "stationarity_check_every": 100,

    # Fichier de maillage
    "mesh_path": "meshes/diamond/diamond_h0.035.npy",

    # Export
    "export": {
        "results": False,  # Exporter les data de solutions (npy)
        "figures": True,  # Exporter les figures (png)
        "graph": False,  # Exporter la simulation sous forme de graph (npz)
        "summary": False,  # Exporter le résumé machine-readable (json)
        "bundle": True,  # Exporter un bundle data riche pour le ML
        "n_snaps": 1,  # Nombre de snapshots
    },
}

def initialize(mesh, cfg):
    # Initialisation des conditions initiales
    rho_inf, p_inf = cfg["rho_inf"], cfg["p_inf"]
    gamma, Mach = cfg["gamma"], cfg["Mach"]
    c_inf = (gamma * p_inf / rho_inf) ** 0.5
    aoa_deg = float(cfg.get("aoa", 0.0)) if cfg.get("case") == "diamond" else 0.0
    aoa_rad = jnp.deg2rad(jnp.asarray(aoa_deg))
    u_inf = Mach * c_inf * jnp.cos(aoa_rad)
    v_inf = Mach * c_inf * jnp.sin(aoa_rad)
    prim0 = jnp.asarray([rho_inf, u_inf, v_inf, p_inf])
    W = helper.getConserved(jnp.tile(prim0[None, :], (mesh.area.shape[0], 1)), gamma=gamma, M=1.0)
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
    stationarity_threshold = float(cfg.get("stationarity_threshold"))
    stationarity_check_every = int(cfg.get("stationarity_check_every"))

    # Calcul dt selon CFL (fixe dans ce cas)
    if 0.6 < cfg["Mach"] < 1.1 and cfg["CFL"] > 0.4:
        print(f"Attention : régime transsonique avec CFL={cfg['CFL']} potentiellement instable. CFL réduite à 0.4")
        cfg["CFL"] = 0.4
    dt = helper.get_dt(W, mesh, CFL=cfg["CFL"], gamma=cfg["gamma"], M=1.0)
    N = int(cfg["tf"] / dt) + 1
    n_snaps = max(1, int(exp["n_snaps"]))

    snap_steps = sorted(set(np.rint(np.linspace(1, N, n_snaps)).astype(int).tolist()) | {N})
    if n_snaps == 1:
        snap_steps = [N]

    print(f"    dt={float(dt):.2e}, Nt={N}, n_snaps={n_snaps}")

    # Warm-up pour compilation JIT
    print("\nCompilation JIT (warm-up)...")
    W_warmup = fn(W, mesh, dt, **kw)
    jax.block_until_ready(W_warmup)

    print("\nRésolution numérique en cours...")
    start_time = time.time()

    def scan_body(carry, _):
        W, W_ref, ref_step, last_stationarity_rel, stop, current_step, stopping_step = carry
        next_step = current_step + 1

        W_next = jax.lax.cond(stop, lambda _: W, lambda _: fn(W, mesh, dt, **kw), operand=None)
        check_now = jnp.logical_or(next_step % stationarity_check_every == 0, next_step == N)

        def do_check(_):
            block_steps = jnp.maximum(1, next_step - ref_step)
            delta_W = W_next - W_ref
            rel = jnp.linalg.norm(delta_W) / (block_steps * (jnp.linalg.norm(W_ref) + 1e-16))
            stop_new = rel <= stationarity_threshold
            stopping_step_new = jnp.where(jnp.logical_and(jnp.logical_not(stop), stop_new), next_step, stopping_step)
            return (W_next, W_next, next_step, rel, jnp.logical_or(stop, stop_new), next_step, stopping_step_new), None

        def no_check(_):
            return (W_next, W_ref, ref_step, last_stationarity_rel, stop, next_step, stopping_step), None

        return jax.lax.cond(check_now, do_check, no_check, operand=None)

    init_carry = (
        W,
        W,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(jnp.inf, dtype=W.dtype),
        jnp.asarray(False),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(N, dtype=jnp.int32),
    )
    final_carry, _ = jax.lax.scan(scan_body, init_carry, None, length=N)
    W, _, _, last_stationarity_rel, converged, stopping_step, stopping_step = final_carry
    jax.block_until_ready(W)

    current_step = int(stopping_step)
    converged = bool(converged)
    stopping_step = int(stopping_step)
    last_stationarity_rel = float(last_stationarity_rel)

    W_probe = fn(W, mesh, dt, **kw)
    jax.block_until_ready(W_probe)
    final_residual = float(jnp.linalg.norm(W_probe - W) / (jnp.linalg.norm(W) + 1e-16))
    
    end_time = time.time()
    wall_time_s = end_time - start_time
    final_time = current_step * float(dt)
    snapshot_name = format_snapshot_name(cfg, final_time)
    print("\n" + "-" * 78)
    print(f"Simulation terminée en {wall_time_s:.2f}s (converged={converged}, t={final_time:.4f}s)")
    print(f"Résidu final : {final_residual:.6e}")
    if exp["results"] or exp["figures"] or exp.get("bundle", True):
        export_snapshot(W, mesh, final_time, cfg, out_dirs, helper, inlet=inlet)
    if exp["graph"]:
        W_snapshots = {round(final_time, 6): np.array(W)}
        export_graph(mesh, W_snapshots, inlet, save_path=str(out_dirs["res"] / "graph.npz"))
    summary = None
    if cfg["case"] in ("diamond", "bump") and (exp.get("summary", True)):
        U_inf = cfg["Mach"] * np.sqrt(cfg["gamma"] * cfg["p_inf"] / cfg["rho_inf"])
        C_D = helper.get_drag_coefficient(W=W, mesh=mesh, rho_inf=cfg["rho_inf"], U_inf=U_inf, L_ref=mesh.metadata["obstacle_length"])
        C_L = helper.get_lift_coefficient(W=W, mesh=mesh, rho_inf=cfg["rho_inf"], U_inf=U_inf, L_ref=mesh.metadata["obstacle_length"])
        delta_S = helper.get_entropy_creation(W_initial, W, mesh, gamma=cfg["gamma"])
        print(f"C_D = {C_D:.6f}, C_L = {C_L:.6f}, ΔS = {delta_S:.6e}")
        summary = build_run_summary(cfg, mesh, C_D, C_L, delta_S, wall_time_s,
            stationarity_rel=final_residual, converged=converged, stopping_step=stopping_step)

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