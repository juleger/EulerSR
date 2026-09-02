"""Validation théorique (choc-détente) du solveur sur le cas diamant.

Théorie de choc oblique / Prandtl-Meyer (Anderson, *Fundamentals of
Aerodynamics*) : chaque panneau plan du losange est un choc ou une détente
isentropique, en cascade depuis l'amont. Déflexion positive = compression
(choc), négative = détente. 4 panneaux (avant/arrière x extrados/intrados) :
    theta_1 = eps - alpha        theta_3 = eps + alpha
    theta_2 = theta_4 = -2*eps   (arêtes milieu-corde, détente pure)

Deux campagnes, sélectionnables via --campaign :
  - "theory" (Campagne B) : compare Cd/Cl/Cp/angle de choc FVM vs théorie sur
    le maillage le plus fin (h=0.0125), point par point.
  - "wall"   (Campagne C) : convergence vs théorie sur toute la chaîne de
    maillage, avec ordre calculé par régression log-log (h<=0.1).
Usage :
    uv run python euler/postproc/theory_validation_diamond.py --campaign both
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
from scipy.optimize import brentq

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
EULER_DIR = REPO_ROOT / "euler"
for _p in (REPO_ROOT, EULER_DIR, EULER_DIR / "meshing", EULER_DIR / "postproc"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from jax_fvm.src.mesh import Mesh
from jax_fvm.src import helper
from mesh_convergence_diamond import H_ALL

GAMMA = 1.4


def prandtl_meyer(M: float, gamma: float = GAMMA) -> float:
    """nu(M) en radians, pour M >= 1."""
    M = max(float(M), 1.0 + 1e-12)
    k = np.sqrt((gamma + 1.0) / (gamma - 1.0))
    return float(k * np.arctan(np.sqrt((M**2 - 1.0) / k**2)) - np.arctan(np.sqrt(M**2 - 1.0)))


def inverse_prandtl_meyer(nu: float, gamma: float = GAMMA, M_max: float = 60.0) -> float:
    """Inverse numérique de prandtl_meyer (nu en radians -> M)."""
    nu_max = prandtl_meyer(M_max, gamma)
    if nu <= 0.0:
        return 1.0
    if nu >= nu_max:
        raise ValueError(f"nu={nu:.4f} rad hors de portee (nu_max={nu_max:.4f} a M={M_max})")
    return brentq(lambda M: prandtl_meyer(M, gamma) - nu, 1.0 + 1e-9, M_max, xtol=1e-10)


def _theta_of_beta(M1: float, beta: float, gamma: float = GAMMA) -> float:
    """tan(theta) implicite : renvoie theta(beta) pour M1 donne (rad)."""
    num = M1**2 * np.sin(beta) ** 2 - 1.0
    den = M1**2 * (gamma + np.cos(2.0 * beta)) + 2.0
    tan_theta = 2.0 / np.tan(beta) * num / den
    return float(np.arctan(tan_theta))


def theta_max(M1: float, gamma: float = GAMMA) -> tuple[float, float]:
    """Déflexion maximale (choc attaché) et beta correspondant, pour M1 donné."""
    mu = np.arcsin(1.0 / M1)
    betas = np.linspace(mu + 1e-6, np.pi / 2 - 1e-6, 4000)
    thetas = np.array([_theta_of_beta(M1, b, gamma) for b in betas])
    i = int(np.argmax(thetas))
    return float(thetas[i]), float(betas[i])


def oblique_shock_beta(M1: float, theta: float, gamma: float = GAMMA) -> float | None:
    """Angle de choc beta (rad, racine faible) pour une déflexion theta (rad) > 0.
    Renvoie None si le choc est détaché (theta > theta_max(M1))."""
    tmax, beta_at_tmax = theta_max(M1, gamma)
    if theta > tmax:
        return None
    mu = np.arcsin(1.0 / M1)
    try:
        return brentq(lambda b: _theta_of_beta(M1, b, gamma) - theta, mu + 1e-9, beta_at_tmax, xtol=1e-10)
    except ValueError:
        return None


@dataclass
class PanelState:
    name: str
    deflection_deg: float      # deflexion locale (repere ecoulement amont du panneau), + = compression
    regime: str                # "shock" | "expansion" | "none" (deflexion nulle)
    attached: bool
    M: float
    p_ratio_upstream: float    # p_panneau / p_amont-immediat (panneau precedent ou infini)
    p_ratio_freestream: float  # p_panneau / p_infini (cumule le long de la cascade)
    Cp: float                  # base sur M_infini, p_infini
    beta_deg: float | None     # angle de choc (deg, repere ecoulement AMONT du panneau) ; None si detente


def _apply_deflection(M_up: float, p_ratio_up_to_inf: float, deflection_rad: float,
                       gamma: float = GAMMA):
    """Applique une deflexion locale (rad, + = compression) a un etat amont
    (M_up, p_up/p_inf). Renvoie (M_aval, p_aval/p_up, attached, beta_rad|None)."""
    if abs(deflection_rad) < 1e-12:
        return M_up, 1.0, True, None
    if deflection_rad > 0.0:
        beta = oblique_shock_beta(M_up, deflection_rad, gamma)
        if beta is None:
            return M_up, 1.0, False, None
        Mn1 = M_up * np.sin(beta)
        p_ratio = 1.0 + 2.0 * gamma / (gamma + 1.0) * (Mn1**2 - 1.0)
        Mn2_sq = (1.0 + 0.5 * (gamma - 1.0) * Mn1**2) / (gamma * Mn1**2 - 0.5 * (gamma - 1.0))
        M_down = float(np.sqrt(Mn2_sq) / np.sin(beta - deflection_rad))
        return M_down, float(p_ratio), True, beta
    else:
        nu_up = prandtl_meyer(M_up, gamma)
        nu_down = nu_up + (-deflection_rad)
        M_down = inverse_prandtl_meyer(nu_down, gamma)
        p0_ratio_up = (1.0 + 0.5 * (gamma - 1.0) * M_up**2) ** (gamma / (gamma - 1.0))
        p0_ratio_down = (1.0 + 0.5 * (gamma - 1.0) * M_down**2) ** (gamma / (gamma - 1.0))
        p_ratio = p0_ratio_up / p0_ratio_down  # p_down/p_up (isentropique, p0 constant)
        return M_down, float(p_ratio), True, None


def diamond_panels(M_inf: float, alpha_deg: float, eps_deg: float,
                    gamma: float = GAMMA) -> dict[str, PanelState]:
    """Etats theoriques (choc-detente) sur les 4 panneaux du profil losange.

    M_inf   : Mach amont (doit etre > 1, ecoulement pleinement supersonique)
    alpha_deg : incidence (deg), meme convention que le solveur (u=M cos a, v=M sin a)
    eps_deg : demi-angle de pointe du losange (deg) ; eps = atan(height/chord)
    """
    alpha = np.deg2rad(alpha_deg)
    eps = np.deg2rad(eps_deg)
    q_inf = 0.5 * gamma * M_inf**2

    theta1 = eps - alpha    # avant-extrados
    theta3 = eps + alpha    # avant-intrados

    M1, r1, ok1, beta1 = _apply_deflection(M_inf, 1.0, theta1, gamma)
    M3, r3, ok3, beta3 = _apply_deflection(M_inf, 1.0, theta3, gamma)

    # arete milieu-corde : detente pure de 2*eps, toujours (coin convexe)
    M2, r2, ok2, beta2 = _apply_deflection(M1, r1, -2.0 * eps, gamma)
    M4, r4, ok4, beta4 = _apply_deflection(M3, r3, -2.0 * eps, gamma)

    p1 = r1
    p2 = r1 * r2
    p3 = r3
    p4 = r3 * r4

    def _mk(name, defl, regime, ok, M, p_ratio_inf, beta):
        return PanelState(
            name=name, deflection_deg=float(np.rad2deg(defl)),
            regime=regime, attached=bool(ok), M=float(M),
            p_ratio_upstream=float('nan'), p_ratio_freestream=float(p_ratio_inf),
            Cp=float((p_ratio_inf - 1.0) / q_inf),
            beta_deg=None if beta is None else float(np.rad2deg(beta)),
        )

    return {
        "upper_front": _mk("upper_front", theta1, "shock" if theta1 > 0 else "expansion", ok1, M1, p1, beta1),
        "upper_rear":  _mk("upper_rear", -2 * eps, "expansion", ok2, M2, p2, beta2),
        "lower_front": _mk("lower_front", theta3, "shock" if theta3 > 0 else "expansion", ok3, M3, p3, beta3),
        "lower_rear":  _mk("lower_rear", -2 * eps, "expansion", ok4, M4, p4, beta4),
    }


def all_attached(panels: dict[str, PanelState]) -> bool:
    return all(p.attached for p in panels.values())

RHO_INF = 1.0
P_INF = 1.0
H_REF = 0.0125  # maillage le plus fin disponible

CASE = "diamond"
RAW_DIR = REPO_ROOT / "data" / "raw" / CASE
MESH_DIR = REPO_ROOT / "data" / "meshes"

# Fenetre de recherche du choc dans le champ, pour la mesure d'angle beta
# (cf. _measure_shock_angle) : marge amont/aval du bord d'attaque (en corde)
# et hauteur de fenetre (en multiples du demi-angle*corde, i.e. de `height`).
SHOCK_WINDOW_X = (-0.35, 0.45)   # relatif a x_le, en unites de corde (amont du LE -> juste avant mi-corde)
SHOCK_WINDOW_Y_FACTOR = 3.0      # relatif a `height`
SHOCK_TOP_PCT = 97.0             # percentile |grad p| retenu comme "points de choc"

# Fenetre autour de l'obstacle pour le Mach max global (evite de capter du
# bruit loin en champ lointain) : centree sur (cx, cy), demi-largeur en
# unites de corde / demi-hauteur en unites de `height`.
OBJECT_WINDOW_X_FACTOR = 0.6
OBJECT_WINDOW_Y_FACTOR = 4.0

VALIDATION_POINTS = [
    ("P2_supersonique_marginal", 1.05, 0.0),
    ("P3_supersonique_propre", 1.50, 0.0),
    ("P4_supersonique_fort", 2.50, 0.0),
    ("P5_incidence", 2.00, 3.0),
    ("P6_mach2_symetrique", 2.00, 0.0),
    ("P7_incidence_forte", 2.50, 5.0),
]

PANEL_NAMES = ("upper_front", "upper_rear", "lower_front", "lower_rear")

_MESH_CACHE: dict[float, Mesh] = {}


def _format_h(h: float) -> str:
    return f"h{h:.4f}".rstrip("0").rstrip(".")


def _find_bundle(h: float, mach: float, aoa: float) -> Path | None:
    import re
    aoa_dir = RAW_DIR / _format_h(h) / f"AOA{aoa:.2f}"
    if not aoa_dir.exists():
        return None
    rex = re.compile(r'AOA([+-]?[0-9.]+)_M([0-9.]+)_.*_t([0-9.]+)\.npz$')
    best_path, best_t = None, -1.0
    for f in aoa_dir.glob(f"AOA{aoa:.2f}_M{mach:.2f}_*.npz"):
        m = rex.search(f.name)
        if m and float(m.group(3)) > best_t:
            best_t, best_path = float(m.group(3)), f
    return best_path


def _load_mesh(h: float) -> Mesh:
    if h not in _MESH_CACHE:
        mesh = Mesh()
        mesh.load_mesh(str(MESH_DIR / f"{CASE}_h{h}.npy"))
        _MESH_CACHE[h] = mesh
    return _MESH_CACHE[h]


def _panel_of(x, y, cx, cy):
    if x < cx and y >= cy:
        return "upper_front"
    if x >= cx and y >= cy:
        return "upper_rear"
    if x < cx and y < cy:
        return "lower_front"
    return "lower_rear"


def _wall_face_ids(mesh, side: str):
    """Variante de helper._lower_wall_face_ids, generalisee haut/bas."""
    points = np.asarray(mesh.points)
    faces = np.asarray(mesh.faces)
    face_markers = np.asarray(mesh.face_markers)
    wall_mask = face_markers == 2  # force_marker par defaut (cf. get_drag_coefficient)
    face_midpoints = np.mean(points[faces], axis=1)
    y_thresh = 0.5 * (points[:, 1].min() + points[:, 1].max())
    side_mask = face_midpoints[:, 1] <= y_thresh if side == "lower" else face_midpoints[:, 1] >= y_thresh
    mask = wall_mask & side_mask
    face_ids = np.where(mask)[0].astype(np.int32)
    mids = face_midpoints[face_ids]
    order = np.lexsort((mids[:, 1], mids[:, 0]))
    return face_ids[order]


def _wall_profile(mesh, W, side: str):
    face_connectivity = np.asarray(mesh.face_connectivity)
    face_ids = _wall_face_ids(mesh, side)
    if face_ids.size == 0:
        return None
    cell_ids = []
    for fid in face_ids:
        matches = np.argwhere(face_connectivity == int(fid))
        if matches.size:
            cell_ids.append(int(matches[0, 0]))
    cell_ids = np.asarray(cell_ids, dtype=np.int32)
    points, faces = np.asarray(mesh.points), np.asarray(mesh.faces)
    mids = np.mean(points[faces[face_ids]], axis=1)
    prim = np.asarray(helper.getPrimitive(W, gamma=GAMMA, M=1.0))
    wall_prim = prim[cell_ids]
    p = wall_prim[:, 3]
    mach = np.asarray(helper.get_mach_number(wall_prim, gamma=GAMMA))
    return dict(x=mids[:, 0], y=mids[:, 1], p=p, mach=mach)


def _plateau_value(profile, panel_masks, key, transform, margin_frac=0.15):
    """Moyenne d'un champ de paroi (key='p' ou 'mach') par panneau, en
    excluant une marge pres des aretes"""
    out = {}
    for name, mask in panel_masks.items():
        xs, vals = profile["x"][mask], profile[key][mask]
        if xs.size < 3:
            out[name] = None
            continue
        order = np.argsort(xs)
        xs, vals = xs[order], vals[order]
        n = xs.size
        lo, hi = int(n * margin_frac), int(n * (1 - margin_frac))
        if hi - lo < 1:
            lo, hi = 0, n
        out[name] = float(np.mean(transform(vals[lo:hi])))
    return out


def _panel_masks(profile, cx, cy):
    return {
        "upper_front": (profile["x"] < cx) & (profile["y"] >= cy),
        "upper_rear": (profile["x"] >= cx) & (profile["y"] >= cy),
        "lower_front": (profile["x"] < cx) & (profile["y"] < cy),
        "lower_rear": (profile["x"] >= cx) & (profile["y"] < cy),
    }


def _measure_shock_angle(d, mesh, cx, cy, chord, height, alpha_deg, side):
    """Mesure l'angle du choc de bord d'attaque directement dans le champ
    (pas seulement a la paroi)"""
    pos = np.asarray(d["node_pos"])
    grad_p = np.asarray(d["primitives_grad"])[:, 3, :]
    gmag = np.linalg.norm(grad_p, axis=1)

    x_le = cx - chord / 2.0
    x_lo, x_hi = x_le + SHOCK_WINDOW_X[0] * chord, x_le + SHOCK_WINDOW_X[1] * chord
    if side == "upper":
        y_mask = (pos[:, 1] > cy + 0.005) & (pos[:, 1] < cy + SHOCK_WINDOW_Y_FACTOR * height)
    else:
        y_mask = (pos[:, 1] < cy - 0.005) & (pos[:, 1] > cy - SHOCK_WINDOW_Y_FACTOR * height)
    window = (pos[:, 0] > x_lo) & (pos[:, 0] < x_hi) & y_mask
    if window.sum() < 10:
        return None

    xs, ys, gs = pos[window, 0], pos[window, 1], gmag[window]
    thresh = np.percentile(gs, SHOCK_TOP_PCT)
    sel = gs >= thresh
    if sel.sum() < 5:
        return None

    pts = np.column_stack([xs[sel], ys[sel]])
    pts_c = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts_c)
    direction = vt[0]

    alpha = np.deg2rad(alpha_deg)
    free_dir = np.array([np.cos(alpha), np.sin(alpha)])
    cos_ang = abs(float(np.dot(direction, free_dir)))
    return float(np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0))))


def _theory_forces(mesh, panels, cx, cy, U_inf, L_ref):
    """Cd/Cl "theoriques" : pression synthetique = plateau theorique du
    panneau, plaquee sur les memes cellules de paroi que le FVM puis integree
    avec les memes routines (helper.get_drag/lift_coefficient). Recalcule par
    maillage (pas seulement au maillage le plus fin) pour une comparaison
    coherente a chaque niveau h de la chaine de convergence."""
    n_cells = mesh.tris.shape[0]
    p_synth = np.full(n_cells, P_INF, dtype=np.float64)
    points, faces = np.asarray(mesh.points), np.asarray(mesh.faces)
    face_markers = np.asarray(mesh.face_markers)
    face_connectivity = np.asarray(mesh.face_connectivity)
    wall_face_ids = np.where(face_markers == 2)[0]
    mids = np.mean(points[faces[wall_face_ids]], axis=1)
    for fid, (mx, my) in zip(wall_face_ids, mids):
        matches = np.argwhere(face_connectivity == int(fid))
        if matches.size == 0:
            continue
        cell_id = int(matches[0, 0])
        panel_name = _panel_of(mx, my, cx, cy)
        p_synth[cell_id] = P_INF * panels[panel_name].p_ratio_freestream

    prim_synth = np.zeros((n_cells, 4), dtype=np.float64)
    prim_synth[:, 0] = RHO_INF
    prim_synth[:, 3] = p_synth
    W_synth = np.asarray(helper.getConserved(prim_synth, gamma=GAMMA, M=1.0))
    Cd_theory = float(helper.get_drag_coefficient(W=W_synth, mesh=mesh, rho_inf=RHO_INF, U_inf=U_inf, L_ref=L_ref))
    Cl_theory = float(helper.get_lift_coefficient(W=W_synth, mesh=mesh, rho_inf=RHO_INF, U_inf=U_inf, L_ref=L_ref))
    return Cd_theory, Cl_theory


def run_point(label: str, mach: float, aoa: float, eps_deg: float):
    print(f"\n=== {label} : Mach={mach} AoA={aoa} (eps={eps_deg}deg) ===")
    panels = diamond_panels(mach, aoa, eps_deg, GAMMA)
    if not all_attached(panels):
        print("    [SKIP] choc detache sur au moins un panneau -- theorie choc-oblique non applicable")
        for name, p in panels.items():
            print(f"      {name}: defl={p.deflection_deg:+.2f} attached={p.attached}")
        return None

    bundle_path = _find_bundle(H_REF, mach, aoa)
    if bundle_path is None:
        print(f"    [WARN] pas de bundle HR (h={H_REF}) pour ce point")
        return None
    d = np.load(bundle_path)
    W = np.asarray(d["conservatives"], dtype=np.float64)
    mesh = Mesh()
    mesh.load_mesh(str(MESH_DIR / f"{CASE}_h{H_REF}.npy"))
    cx, cy = mesh.metadata["center"]["cx"], mesh.metadata["center"]["cy"]
    chord = float(mesh.metadata["chord"])
    height = float(mesh.metadata["height"])
    L_ref = float(mesh.metadata["obstacle_length"])
    U_inf = mach * np.sqrt(GAMMA * P_INF / RHO_INF)

    Cd_fvm = float(helper.get_drag_coefficient(W=W, mesh=mesh, rho_inf=RHO_INF, U_inf=U_inf, L_ref=L_ref))
    Cl_fvm = float(helper.get_lift_coefficient(W=W, mesh=mesh, rho_inf=RHO_INF, U_inf=U_inf, L_ref=L_ref))
    Cd_theory, Cl_theory = _theory_forces(mesh, panels, cx, cy, U_inf, L_ref)

    # -- Cp et Mach de plateau par panneau (FVM, paroi haute + basse) vs theorie
    q_inf = 0.5 * RHO_INF * U_inf**2
    profile_lo = _wall_profile(mesh, W, "lower")
    profile_hi = _wall_profile(mesh, W, "upper")
    cp_panel, mach_panel = {}, {}
    for side_name, profile in (("lower", profile_lo), ("upper", profile_hi)):
        if profile is None:
            continue
        masks = _panel_masks(profile, cx, cy)
        cp_panel.update({k: v for k, v in _plateau_value(
            profile, masks, "p", lambda p: (p - P_INF) / q_inf).items() if v is not None})
        mach_panel.update({k: v for k, v in _plateau_value(
            profile, masks, "mach", lambda m: m).items() if v is not None})

    beta_meas = {
        "upper_front": _measure_shock_angle(d, mesh, cx, cy, chord, height, aoa, "upper"),
        "lower_front": _measure_shock_angle(d, mesh, cx, cy, chord, height, aoa, "lower"),
    }

    print(f"    Cd : FVM={Cd_fvm:+.5f}  theorie={Cd_theory:+.5f}  ecart={Cd_fvm - Cd_theory:+.5f} "
          f"({100 * (Cd_fvm - Cd_theory) / Cd_theory:+.2f}%)")
    print(f"    Cl : FVM={Cl_fvm:+.5f}  theorie={Cl_theory:+.5f}  ecart={Cl_fvm - Cl_theory:+.5f}")
    print(f"    {'panneau':14s} {'Cp_FVM':>10s} {'Cp_theorie':>12s} {'M_FVM':>8s} {'M_theorie':>10s} "
          f"{'beta_FVM':>9s} {'beta_theorie':>13s}  regime")
    rows = []
    for name in PANEL_NAMES:
        cp_f, m_f = cp_panel.get(name), mach_panel.get(name)
        cp_t, m_t = panels[name].Cp, panels[name].M
        b_f, b_t = beta_meas.get(name), panels[name].beta_deg
        cp_f_s = f"{cp_f:+.4f}" if cp_f is not None else "n/a"
        m_f_s = f"{m_f:.4f}" if m_f is not None else "n/a"
        b_f_s = f"{b_f:.2f}" if b_f is not None else "n/a"
        b_t_s = f"{b_t:.2f}" if b_t is not None else "n/a"
        print(f"    {name:14s} {cp_f_s:>10s} {cp_t:>+12.4f} {m_f_s:>8s} {m_t:>10.4f} "
              f"{b_f_s:>9s} {b_t_s:>13s}  {panels[name].regime}")
        rows.append((name, cp_f, cp_t, m_f, m_t, b_f, b_t, panels[name].regime))

    return dict(label=label, mach=mach, aoa=aoa, Cd_fvm=Cd_fvm, Cd_theory=Cd_theory,
                Cl_fvm=Cl_fvm, Cl_theory=Cl_theory, panels=rows,
                profile_lo=profile_lo, profile_hi=profile_hi, panel_theory=panels, cx=cx)


def plot_cp_theory(res: dict, out_path: Path):
    if res is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    U_inf = res["mach"] * np.sqrt(GAMMA * P_INF / RHO_INF)
    q_inf = 0.5 * RHO_INF * U_inf**2
    for side_name, profile in (("bas", res["profile_lo"]), ("haut", res["profile_hi"])):
        if profile is None:
            continue
        order = np.argsort(profile["x"])
        x, p = profile["x"][order], profile["p"][order]
        cp = (p - P_INF) / q_inf
        ax.plot(x, cp, ".", ms=2.5, alpha=0.6, label=f"FVM (paroi {side_name}, HR h={H_REF})")
    cx = res["cx"]
    for name, panel in res["panel_theory"].items():
        x0, x1 = (0, cx) if "front" in name else (cx, 2 * cx)
        ax.hlines(panel.Cp, x0, x1, color="black", linewidth=2.0, linestyle="--")
    ax.axvline(cx, color="gray", linewidth=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("$C_p$")
    ax.set_title(f"{res['label']} -- Mach={res['mach']}, AoA={res['aoa']}° "
                 f"(pointilles = theorie choc-detente)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    figure -> {out_path}")


def run_theory_validation(case: str = "diamond") -> None:
    """Campagne "theory" : Cd/Cl/Cp/angle de choc, FVM vs theorie (HR seul)."""
    global CASE, RAW_DIR
    CASE = case
    RAW_DIR = REPO_ROOT / "data" / "raw" / CASE
    out_dir = REPO_ROOT / "data" / "validation" / f"theory_{CASE}"
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = ["label,mach,aoa,Cd_fvm,Cd_theory,Cl_fvm,Cl_theory,panel,"
             "Cp_fvm,Cp_theory,M_fvm,M_theory,beta_fvm_deg,beta_theory_deg,regime"]
    for label, mach, aoa in VALIDATION_POINTS:
        res = run_point(label, mach, aoa, eps_deg=5.0)
        if res is None:
            continue
        plot_cp_theory(res, out_dir / f"cp_{label}.png")
        for name, cp_f, cp_t, m_f, m_t, b_f, b_t, regime in res["panels"]:
            cp_f_s = "" if cp_f is None else f"{cp_f:.6f}"
            m_f_s = "" if m_f is None else f"{m_f:.6f}"
            b_f_s = "" if b_f is None else f"{b_f:.4f}"
            b_t_s = "" if b_t is None else f"{b_t:.4f}"
            lines.append(f"{res['label']},{res['mach']},{res['aoa']},{res['Cd_fvm']:.6f},"
                         f"{res['Cd_theory']:.6f},{res['Cl_fvm']:.6f},{res['Cl_theory']:.6f},"
                         f"{name},{cp_f_s},{cp_t:.6f},{m_f_s},{m_t:.6f},{b_f_s},{b_t_s},{regime}")
    (out_dir / "theory_vs_fvm.csv").write_text("\n".join(lines) + "\n")
    print(f"\nCSV -> {out_dir / 'theory_vs_fvm.csv'}")

def _full_profile_rms(profile, panels, cx, cy, U_inf):
    if profile is None or profile["x"].size == 0:
        return None
    q_inf = 0.5 * RHO_INF * U_inf**2
    cp_fvm = (profile["p"] - P_INF) / q_inf
    cp_theory = np.array([panels[_panel_of(x, y, cx, cy)].Cp
                           for x, y in zip(profile["x"], profile["y"])])
    return cp_fvm - cp_theory


def compute_errors(label, mach, aoa, eps_deg=5.0):
    panels = diamond_panels(mach, aoa, eps_deg, GAMMA)
    if not all_attached(panels):
        return None
    U_inf = mach * np.sqrt(GAMMA * P_INF / RHO_INF)
    mach_max_theory = max(p.M for p in panels.values())

    rows = []
    for h in H_ALL:
        bundle_path = _find_bundle(h, mach, aoa)
        if bundle_path is None:
            continue
        d = np.load(bundle_path)
        W = np.asarray(d["conservatives"], dtype=np.float64)
        mesh = _load_mesh(h)
        cx, cy = mesh.metadata["center"]["cx"], mesh.metadata["center"]["cy"]
        chord = float(mesh.metadata["chord"])
        height = float(mesh.metadata["height"])
        L_ref = float(mesh.metadata["obstacle_length"])

        # -- Cd/Cl FVM vs. theorie (theorie recalculee sur ce maillage, cf.
        # _theory_forces : comparaison coherente a iso-discretisation)
        Cd_fvm = float(helper.get_drag_coefficient(W=W, mesh=mesh, rho_inf=RHO_INF, U_inf=U_inf, L_ref=L_ref))
        Cl_fvm = float(helper.get_lift_coefficient(W=W, mesh=mesh, rho_inf=RHO_INF, U_inf=U_inf, L_ref=L_ref))
        Cd_theory, Cl_theory = _theory_forces(mesh, panels, cx, cy, U_inf, L_ref)
        err_cd = float(abs(Cd_fvm - Cd_theory))
        err_cl = float(abs(Cl_fvm - Cl_theory))

        profile_lo = _wall_profile(mesh, W, "lower")
        profile_hi = _wall_profile(mesh, W, "upper")

        # -- Cp/Mach "plateau" (moyenne par panneau, RMS sur les 4)
        cp_errs, mach_plateau_errs = [], []
        for side_name, profile in (("lower", profile_lo), ("upper", profile_hi)):
            if profile is None:
                continue
            masks = _panel_masks(profile, cx, cy)
            q_inf = 0.5 * RHO_INF * U_inf**2
            cp_panel = _plateau_value(profile, masks, "p", lambda p: (p - P_INF) / q_inf)
            mach_panel = _plateau_value(profile, masks, "mach", lambda m: m)
            for name, cp_f in cp_panel.items():
                if cp_f is not None:
                    cp_errs.append(cp_f - panels[name].Cp)
            for name, m_f in mach_panel.items():
                if m_f is not None:
                    mach_plateau_errs.append(m_f - panels[name].M)
        if not cp_errs:
            continue
        rms_plateau = float(np.sqrt(np.mean(np.square(cp_errs))))
        rms_mach_plateau = float(np.sqrt(np.mean(np.square(mach_plateau_errs)))) if mach_plateau_errs else None

        # -- Cp "profil complet" (ponctuel, haut+bas, y compris transitions)
        full_errs = []
        for profile in (profile_lo, profile_hi):
            diff = _full_profile_rms(profile, panels, cx, cy, U_inf)
            if diff is not None:
                full_errs.append(diff)
        if not full_errs:
            continue
        rms_full = float(np.sqrt(np.mean(np.square(np.concatenate(full_errs)))))

        # -- Mach max global (fenetre autour de l'obstacle) vs. Mach theorique
        # max des 4 panneaux : la dissipation numerique lisse le pic
        # d'expansion au coin milieu-corde sur maillage grossier, et le
        # maximum discret doit remonter vers la valeur theorique quand h -> 0.
        prim_all = np.asarray(helper.getPrimitive(W, gamma=GAMMA, M=1.0))
        mach_field = np.asarray(helper.get_mach_number(prim_all, gamma=GAMMA))
        centroids = np.asarray(d["node_pos"])  # = mesh.barycenter (cf. euler/utils.py)
        win = ((np.abs(centroids[:, 0] - cx) < OBJECT_WINDOW_X_FACTOR * chord) &
               (np.abs(centroids[:, 1] - cy) < OBJECT_WINDOW_Y_FACTOR * height))
        mach_max_fvm = float(mach_field[win].max()) if win.any() else float(mach_field.max())
        err_mach_max = float(abs(mach_max_theory - mach_max_fvm))

        # -- angle de choc de bord d'attaque (mesure directe dans le champ,
        # RMS sur les panneaux avant haut/bas)
        beta_errs = []
        for side, name in (("upper", "upper_front"), ("lower", "lower_front")):
            b_f = _measure_shock_angle(d, mesh, cx, cy, chord, height, aoa, side)
            b_t = panels[name].beta_deg
            if b_f is not None and b_t is not None:
                beta_errs.append(b_f - b_t)
        err_beta = float(np.sqrt(np.mean(np.square(beta_errs)))) if beta_errs else None

        rows.append(dict(h=h, n_cells=int(mesh.tris.shape[0]),
                          cd_fvm=Cd_fvm, cd_theory=Cd_theory, err_cd=err_cd,
                          cl_fvm=Cl_fvm, cl_theory=Cl_theory, err_cl=err_cl,
                          rms_plateau=rms_plateau, rms_full=rms_full,
                          rms_mach_plateau=rms_mach_plateau,
                          mach_max_fvm=mach_max_fvm, mach_max_theory=mach_max_theory,
                          err_mach_max=err_mach_max, err_beta=err_beta))
        mp_s = f"{rms_mach_plateau:.3e}" if rms_mach_plateau is not None else "n/a"
        mm_s = f"{err_mach_max:.3e}" if err_mach_max is not None else "n/a"
        b_s = f"{err_beta:.3e}" if err_beta is not None else "n/a"
        print(f"    h={h:<8g} n_cells={rows[-1]['n_cells']:>7d}  "
              f"err_Cd={err_cd:.3e}  err_Cl={err_cl:.3e}  "
              f"RMS_Cp_plateau={rms_plateau:.3e}  RMS_Cp_profil_complet={rms_full:.3e}  "
              f"RMS_M_plateau={mp_s}  err_M_max={mm_s}  RMS_err_beta={b_s}")
    return rows


def fit_order(rows, key, h_max=0.1):
    """Regression log-log (moindres carres) sur les h <= h_max."""
    pts = [(r["h"], r[key]) for r in rows if r["h"] <= h_max and r[key] is not None and r[key] > 0]
    if len(pts) < 3:
        return None, None, len(pts)
    h = np.log(np.array([p[0] for p in pts]))
    e = np.log(np.array([p[1] for p in pts]))
    p_order, log_c = np.polyfit(h, e, 1)
    pred = p_order * h + log_c
    ss_res = np.sum((e - pred) ** 2)
    ss_tot = np.sum((e - np.mean(e)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(p_order), float(r2), len(pts)


def _plot_slope_refs(ax, h_sub, curve_sub):
    if h_sub.size < 2:
        return
    for ref_p, style in ((1, ":"), (2, "--")):
        h_ref = np.array([h_sub.min(), h_sub.max()])
        c = curve_sub[np.argmin(h_sub)] / h_sub.min() ** ref_p
        ax.loglog(h_ref, c * h_ref ** ref_p, style, color="gray", linewidth=1,
                   label=f"pente {ref_p} (repere)")


def plot_wall_convergence(label, rows, out_path):
    if not rows:
        return
    h = np.array([r["h"] for r in rows])
    plateau = np.array([r["rms_plateau"] for r in rows])
    full = np.array([r["rms_full"] for r in rows])
    err_cd = np.array([r["err_cd"] for r in rows])
    err_cl = np.array([r["err_cl"] for r in rows])
    mach_plateau = np.array([r["rms_mach_plateau"] if r["rms_mach_plateau"] is not None else np.nan for r in rows])
    mach_max_err = np.array([r["err_mach_max"] if r["err_mach_max"] is not None else np.nan for r in rows])
    beta_err = np.array([r["err_beta"] if r["err_beta"] is not None else np.nan for r in rows])

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    axes = axes.ravel()

    axes[0].loglog(h, err_cd, "o-", color="tab:blue", label="$|C_D^{fvm}-C_D^{theo}|$")
    axes[0].loglog(h, err_cl, "s-", color="tab:orange", label="$|C_L^{fvm}-C_L^{theo}|$")
    _plot_slope_refs(axes[0], h, err_cd)
    axes[0].set_ylabel("erreur vs. theorie choc-detente")
    axes[0].set_title("$C_D$ / $C_L$")

    axes[1].loglog(h, plateau, "o-", color="tab:blue", label="RMS $C_p$ plateau")
    axes[1].loglog(h, full, "s-", color="tab:orange", label="RMS $C_p$ profil complet")
    _plot_slope_refs(axes[1], h, full)
    axes[1].set_title("$C_p$ de paroi")

    m1, m2 = np.isfinite(mach_plateau), np.isfinite(mach_max_err)
    if np.any(m1):
        axes[2].loglog(h[m1], mach_plateau[m1], "o-", color="tab:blue", label="RMS $M$ plateau")
    if np.any(m2):
        axes[2].loglog(h[m2], mach_max_err[m2], "^-", color="tab:red",
                        label=r"$|M_{max}^{theo} - M_{max}^{fvm}|$")
        _plot_slope_refs(axes[2], h[m2], mach_max_err[m2])
    axes[2].set_ylabel("erreur vs. theorie choc-detente")
    axes[2].set_title("Mach")

    m3 = np.isfinite(beta_err)
    if np.any(m3):
        axes[3].loglog(h[m3], beta_err[m3], "d-", color="tab:green",
                        label=r"RMS $|\beta_{fvm}-\beta_{theo}|$ (deg, bord d'attaque)")
        _plot_slope_refs(axes[3], h[m3], beta_err[m3])
    axes[3].set_title("Angle de choc")

    for ax in axes:
        ax.set_xlabel("h")
        ax.invert_xaxis()
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle(f"Convergence de maillage vs. theorie choc-detente -- {label}")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    figure -> {out_path}")


def run_wall_convergence(case: str = "diamond") -> None:
    """Campagne "wall" : convergence du profil de paroi complet vs theorie."""
    global CASE, RAW_DIR
    CASE = case
    RAW_DIR = REPO_ROOT / "data" / "raw" / CASE
    out_dir = REPO_ROOT / "data" / "validation" / f"wall_profile_convergence_{CASE}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _MESH_CACHE.clear()

    csv_lines = ["label,mach,aoa,h,n_cells,cd_fvm,cd_theory,err_cd,cl_fvm,cl_theory,err_cl,"
                 "rms_cp_plateau,rms_cp_profil_complet,"
                 "rms_mach_plateau,mach_max_fvm,mach_max_theory,err_mach_max,rms_err_beta_deg"]
    summary_lines = ["label,mach,aoa,quantity,p_order,r2,n_points"]

    for label, mach, aoa in VALIDATION_POINTS:
        print(f"\n=== {label} : Mach={mach} AoA={aoa} ===")
        rows = compute_errors(label, mach, aoa)
        if rows is None:
            print("    [SKIP] choc detache")
            continue
        for r in rows:
            mp = "" if r["rms_mach_plateau"] is None else f"{r['rms_mach_plateau']:.6e}"
            be = "" if r["err_beta"] is None else f"{r['err_beta']:.6e}"
            csv_lines.append(f"{label},{mach},{aoa},{r['h']},{r['n_cells']},"
                              f"{r['cd_fvm']:.6f},{r['cd_theory']:.6f},{r['err_cd']:.6e},"
                              f"{r['cl_fvm']:.6f},{r['cl_theory']:.6f},{r['err_cl']:.6e},"
                              f"{r['rms_plateau']:.6e},{r['rms_full']:.6e},{mp},"
                              f"{r['mach_max_fvm']:.6f},{r['mach_max_theory']:.6f},"
                              f"{r['err_mach_max']:.6e},{be}")
        plot_wall_convergence(label, rows, out_dir / f"wall_convergence_{label}.png")

        for key, tag in (("err_cd", "cd"), ("err_cl", "cl"),
                          ("rms_plateau", "cp_plateau"), ("rms_full", "cp_profil_complet"),
                          ("rms_mach_plateau", "mach_plateau"), ("err_mach_max", "mach_max"),
                          ("err_beta", "beta_choc")):
            p_order, r2, n = fit_order(rows, key)
            if p_order is None:
                print(f"    [{tag}] pas assez de points (h<=0.1) pour la regression")
                continue
            print(f"    [{tag}] ordre ajuste (regression log-log, h<=0.1, n={n}) : "
                  f"p={p_order:.3f}  R2={r2:.4f}")
            summary_lines.append(f"{label},{mach},{aoa},{tag},{p_order:.6f},{r2:.6f},{n}")

    (out_dir / "wall_profile_errors.csv").write_text("\n".join(csv_lines) + "\n")
    (out_dir / "wall_profile_order_fit.csv").write_text("\n".join(summary_lines) + "\n")
    print(f"\nCSV -> {out_dir / 'wall_profile_errors.csv'}")
    print(f"CSV -> {out_dir / 'wall_profile_order_fit.csv'}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", default="diamond", choices=["diamond", "diamond_sym"])
    parser.add_argument("--campaign", default="both", choices=["theory", "wall", "both"],
                        help="theory=Cd/Cl/Cp/beta vs theorie (HR seul) ; "
                             "wall=convergence du profil de paroi (toute la chaine)")
    args = parser.parse_args()
    if args.campaign in ("theory", "both"):
        run_theory_validation(args.case)
    if args.campaign in ("wall", "both"):
        run_wall_convergence(args.case)


if __name__ == "__main__":
    main()
