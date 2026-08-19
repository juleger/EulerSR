import jax
import numpy as np
import jax.numpy as jnp
import jax_fvm.src.helper as helper
import jax_fvm.src.euler_solver as euler
from graph import export_graph
import time

try:
    from config import build_config_from_cli, load_default_config, load_mesh, print_config, setup_dirs
    from config import _AOA_CASES_LC, _KNOWN_CASES_LC
    from utils import build_run_summary, export_snapshot, export_run_summary, format_snapshot_name
except ModuleNotFoundError:
    from euler.config import build_config_from_cli, load_default_config, load_mesh, print_config, setup_dirs
    from euler.config import _AOA_CASES_LC, _KNOWN_CASES_LC
    from euler.utils import build_run_summary, export_snapshot, export_run_summary, format_snapshot_name

"""
Script principal pour la résolution numérique d'Euler compressible sur les cas tests prédéfinis (diamond, bump) via la méthodes des volumes finis. Le script est configuré via un fichier TOML (euler/config.toml) ou bien via arguments CLI. Différentes options de schémas temporels, flux numériques, reconstruction spatiale, exports, etc... sont disponibles. 

Usage :
    uv run python euler/main.py (si config.toml est correctement configuré)
    uv run python euler/main.py --case diamond --Mach 2.5 --aoa 5 --tf 0.5 --flux HLLC --reconstruction MUSCL --time_scheme RK2
"""

CFG = load_default_config()

def initialize(mesh, cfg):
    # Initialisation des conditions initiales
    rho_inf, p_inf = cfg["rho_inf"], cfg["p_inf"]
    gamma, Mach = cfg["gamma"], cfg["Mach"]
    c_inf = (gamma * p_inf / rho_inf) ** 0.5
    aoa_deg = float(cfg.get("aoa", 0.0)) if str(cfg.get("case", "")).lower() in _AOA_CASES_LC else 0.0
    aoa_rad = jnp.deg2rad(jnp.asarray(aoa_deg))
    u_inf = Mach * c_inf * jnp.cos(aoa_rad)
    v_inf = Mach * c_inf * jnp.sin(aoa_rad)
    prim0 = jnp.asarray([rho_inf, u_inf, v_inf, p_inf])
    W = helper.getConserved(jnp.tile(prim0[None, :], (mesh.area.shape[0], 1)), gamma=gamma, M=1.0)
    inlet = helper.getConserved(prim0[None], gamma=gamma, M=1.0)[0]
    return W, inlet


def run(W, mesh, inlet, cfg, out_dirs):
    # Résolution numérique d'euler compressible
    verbose = bool(cfg.get("verbose", True))
    W_initial = W

    # Schéma en temps
    scheme = str(cfg["time_scheme"]).upper()
    fn = {"EE": euler.time_step_Euler, "RK2": euler.time_step_RK2, "RK4": euler.time_step_RK4,
          "SRK2": euler.time_step_RK2_SSP, "SSP_RK2": euler.time_step_RK2_SSP}[scheme]

    # kwargs pour le solver (flux, reconstruction, etc...)
    kw = dict(gamma=cfg["gamma"], M=1.0, reconstruction=cfg["reconstruction"],
              flux=cfg["flux"], value=inlet, entropy=False)
    exp = cfg["export"]
    stationarity_threshold = float(cfg.get("stationarity_threshold"))
    stationarity_check_every = int(cfg.get("stationarity_check_every"))

    # Calcul dt selon CFL (fixe dans ce cas)
    if 0.6 < cfg["Mach"] < 1.1 and cfg["CFL"] > 0.45:
        if verbose:
            print(f"Attention : régime transsonique avec CFL={cfg['CFL']} potentiellement instable. CFL réduite à 0.45")
        cfg["CFL"] = 0.45
    dt = helper.get_dt(W, mesh, CFL=cfg["CFL"], gamma=cfg["gamma"], M=1.0)
    N = int(cfg["tf"] / dt) + 1
    n_snaps = max(1, int(exp["n_snaps"]))

    snap_steps = sorted(set(np.rint(np.linspace(1, N, n_snaps)).astype(int).tolist()) | {N})
    if n_snaps == 1:
        snap_steps = [N]

    if verbose:
        print(f"    dt={float(dt):.2e}, Nt={N}, n_snaps={n_snaps}")

    # Warm-up pour compilation JIT
    if verbose:
        print("\nCompilation JIT (warm-up)...")
    W_warmup = fn(W, mesh, dt, **kw)
    jax.block_until_ready(W_warmup)

    if verbose:
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
    if verbose:
        print("\n" + "-" * 78)
        print(f"Simulation terminée en {wall_time_s:.2f}s (converged={converged}, t={final_time:.4f}s)")
        print(f"Résidu final : {final_residual:.6e}")
    else:
        print(f"Simulation terminée en {wall_time_s:.2f}s (converged={converged}, t={final_time:.4f}s, résidu={final_residual:.2e})")
    if exp["results"] or exp["figures"] or exp.get("bundle", True):
        export_snapshot(W, mesh, final_time, cfg, out_dirs, helper, inlet=inlet)
    if exp["graph"]:
        W_snapshots = {round(final_time, 6): np.array(W)}
        export_graph(mesh, W_snapshots, inlet, save_path=str(out_dirs["res"] / "graph.npz"))
    summary = None
    if str(cfg["case"]).lower() in _KNOWN_CASES_LC and (exp.get("summary", True)):
        U_inf = cfg["Mach"] * np.sqrt(cfg["gamma"] * cfg["p_inf"] / cfg["rho_inf"])
        C_D = helper.get_drag_coefficient(W=W, mesh=mesh, rho_inf=cfg["rho_inf"], U_inf=U_inf, L_ref=mesh.metadata["obstacle_length"])
        C_L = helper.get_lift_coefficient(W=W, mesh=mesh, rho_inf=cfg["rho_inf"], U_inf=U_inf, L_ref=mesh.metadata["obstacle_length"])
        delta_S = helper.get_entropy_creation(W_initial, W, mesh, gamma=cfg["gamma"])
        if verbose:
            print(f"C_D = {C_D:.6f}, C_L = {C_L:.6f}, ΔS = {delta_S:.6e}")
        summary = build_run_summary(cfg, mesh, C_D, C_L, delta_S, wall_time_s,
            stationarity_rel=final_residual, converged=converged, stopping_step=stopping_step)

    if exp.get("summary", True) and summary is not None:
        export_run_summary(out_dirs, summary, snapshot_name, verbose=verbose)
    return W

if __name__ == "__main__":

    CFG = build_config_from_cli(CFG)
    mesh = load_mesh(CFG)
    out_dirs = setup_dirs(CFG, mesh)
    print_config(CFG, mesh, out_dirs)
    W, inlet = initialize(mesh, CFG)
    run(W, mesh, inlet, CFG, out_dirs)