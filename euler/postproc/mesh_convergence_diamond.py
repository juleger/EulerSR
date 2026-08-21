"""Campagne convergence de maillage sur le cas diamant.

Reconstruit Cd, Cl et le profil de Mach de paroi a partir des bundles FVM
bruts (data/raw/diamond/h<h>/AOA<aoa>/*.npz), pour un petit nombre de points
(Mach, AoA) representatifs, sur la chaine de maillages
h in {0.3, 0.2, 0.15, 0.1, 0.05, 0.025, 0.0175, 0.0125}.

NB : la creation d'entropie globale (deltaS) a ete retiree de cette campagne
-- c'est une quantite integree tres bruitee (cf. mouvement du choc de cellule
en cellule d'un maillage a l'autre), pas exploitable pour un ordre de
convergence. Pour des quantites physiques qui convergent proprement (angle de
choc, Mach max, Cp de paroi), voir theory_validation_diamond.py --campaign
wall, qui compare a la theorie choc-detente plutot qu'au maillage le plus fin.

L'ordre de convergence observe et l'indice GCI (Roache) sont calcules
uniquement sur la sous-chaine a ratio constant r=2 :
    0.2 -> 0.1 -> 0.05 -> 0.025 -> 0.0125
(les niveaux 0.3 et 0.0175 servent seulement au tracé de tendance visuel).

Usage :
    uv run python euler/postproc/mesh_convergence_diamond.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
EULER_DIR = REPO_ROOT / "euler"
for p in (REPO_ROOT, EULER_DIR, EULER_DIR / "meshing"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from jax_fvm.src.mesh import Mesh
from jax_fvm.src import helper

GAMMA = 1.4
RHO_INF = 1.0
P_INF = 1.0

# Chaine complete (tendance visuelle) et sous-chaine a ratio constant r=2,
# dyadique (puissances de 2) jusqu'a 0.0125 (calcul GCI / ordre observe).
H_ALL = [0.4, 0.3, 0.2, 0.15, 0.1, 0.05, 0.025, 0.0175, 0.0125]
H_GCI = [0.4, 0.2, 0.1, 0.05, 0.025, 0.0125]

# (label, Mach, AoA)
TEST_POINTS = [
    ("P1_transsonique", 0.90, 0.0),
    ("P2_supersonique_marginal", 1.05, 0.0),
    ("P3_supersonique_propre", 1.50, 0.0),
    ("P4_supersonique_fort", 2.50, 0.0),
    ("P5_incidence", 2.00, 3.0),
    ("P6_mach2_symetrique", 2.00, 0.0),
    ("P7_incidence_forte", 2.50, 5.0),
]

MESH_DIR = REPO_ROOT / "data" / "meshes"

# Parametres par cas (mis a jour par main() selon --case) : "diamond" (maillage
# Delaunay generique avec trou) ou "diamond_sym" (maillage miroir, symetrique
# par construction). Cf. euler/meshing/diamond.py::build_mesh(symmetric=...).
CASE = "diamond"
RAW_DIR = REPO_ROOT / "data" / "raw" / CASE
OUT_DIR = REPO_ROOT / "data" / "validation" / f"mesh_convergence_{CASE}"

_FNAME_RE = re.compile(r'AOA([+-]?[0-9.]+)_M([0-9.]+)_.*_t([0-9.]+)\.npz$')


def _format_h(h: float) -> str:
    return f"h{h:.4f}".rstrip("0").rstrip(".")


def _find_bundle(h: float, mach: float, aoa: float) -> Path | None:
    aoa_dir = RAW_DIR / _format_h(h) / f"AOA{aoa:.2f}"
    if not aoa_dir.exists():
        return None
    best_path, best_t = None, -1.0
    for f in aoa_dir.glob(f"AOA{aoa:.2f}_M{mach:.2f}_*.npz"):
        m = _FNAME_RE.search(f.name)
        if not m:
            continue
        t = float(m.group(3))
        if t > best_t:
            best_t, best_path = t, f
    return best_path


_MESH_CACHE: dict[float, Mesh] = {}


def _load_mesh(h: float) -> Mesh:
    if h not in _MESH_CACHE:
        mesh = Mesh()
        mesh.load_mesh(str(MESH_DIR / f"{CASE}_h{h}.npy"))
        _MESH_CACHE[h] = mesh
    return _MESH_CACHE[h]


def _relative_profile_error(ref, cur, n_samples=400):
    if ref is None or cur is None or ref["s"].size < 2 or cur["s"].size < 2:
        return None
    s_min = max(ref["s"].min(), cur["s"].min())
    s_max = min(ref["s"].max(), cur["s"].max())
    if s_max <= s_min:
        return None
    grid = np.linspace(s_min, s_max, n_samples)
    ref_i = np.interp(grid, ref["s"], ref["mach"])
    cur_i = np.interp(grid, cur["s"], cur["mach"])
    denom = np.linalg.norm(ref_i)
    if denom <= 1e-16:
        return None
    return float(np.linalg.norm(cur_i - ref_i) / denom)


def _compute_point_row(h: float, mach: float, aoa: float, ref_profile) -> dict | None:
    bundle_path = _find_bundle(h, mach, aoa)
    if bundle_path is None:
        print(f"    [WARN] pas de bundle pour h={h} Mach={mach} AoA={aoa}")
        return None
    d = np.load(bundle_path)
    mesh = _load_mesh(h)
    W = np.asarray(d["conservatives"], dtype=np.float64)

    U_inf = mach * np.sqrt(GAMMA * P_INF / RHO_INF)
    L_ref = float(mesh.metadata["obstacle_length"])
    Cd = float(helper.get_drag_coefficient(W=W, mesh=mesh, rho_inf=RHO_INF, U_inf=U_inf, L_ref=L_ref))
    Cl = float(helper.get_lift_coefficient(W=W, mesh=mesh, rho_inf=RHO_INF, U_inf=U_inf, L_ref=L_ref))

    wall = helper.get_wall_profile(W, mesh, gamma=GAMMA, M=1.0)
    wall_err = _relative_profile_error(ref_profile, wall) if ref_profile is not None else None

    return dict(h=h, mach=mach, aoa=aoa, n_cells=int(mesh.tris.shape[0]),
                cd=Cd, cl=Cl, wall_profile=wall, wall_profile_rel_err=wall_err)


def richardson_gci(values_by_h: dict[float, float], chain=H_GCI, safety=1.25):
    """Ordre observe et GCI_fine sur les triplets glissants de `chain` (ratio r=2)."""
    r = chain[0] / chain[1]  # =2
    out = []
    for i in range(len(chain) - 2):
        h1, h2, h3 = chain[i], chain[i + 1], chain[i + 2]  # coarse, mid, fine
        f1, f2, f3 = values_by_h.get(h1), values_by_h.get(h2), values_by_h.get(h3)
        if f1 is None or f2 is None or f3 is None:
            continue
        num, den = f1 - f2, f2 - f3
        if abs(den) < 1e-14 or (num / den) <= 0:
            out.append(dict(h_coarse=h1, h_mid=h2, h_fine=h3, p_obs=None, extrap=None, gci_fine_pct=None))
            continue
        p_obs = np.log(num / den) / np.log(r)
        extrap = f3 + (f3 - f2) / (r**p_obs - 1.0)
        eps = abs((f3 - f2) / f3) if f3 != 0 else float("nan")
        gci_fine = safety * eps / (r**p_obs - 1.0) * 100.0
        out.append(dict(h_coarse=h1, h_mid=h2, h_fine=h3, p_obs=float(p_obs),
                         extrap=float(extrap), gci_fine_pct=float(gci_fine)))
    return out


def run_point(label: str, mach: float, aoa: float) -> list[dict]:
    print(f"--- {label} : Mach={mach} AoA={aoa} ---")
    # profil de reference = maillage le plus fin disponible (h=0.0125)
    ref_bundle = _find_bundle(min(H_ALL), mach, aoa)
    ref_profile = None
    if ref_bundle is not None:
        d = np.load(ref_bundle)
        ref_profile = helper.get_wall_profile(np.asarray(d["conservatives"], dtype=np.float64),
                                               _load_mesh(min(H_ALL)), gamma=GAMMA, M=1.0)
    rows = []
    for h in H_ALL:
        row = _compute_point_row(h, mach, aoa, ref_profile)
        if row is not None:
            rows.append(row)
            werr = row["wall_profile_rel_err"]
            werr_s = f"{werr:.3e}" if werr is not None else "n/a"
            print(f"    h={h:<8g} n_cells={row['n_cells']:>7d}  Cd={row['cd']:+.5f}  Cl={row['cl']:+.5f}  "
                  f"err_paroi/finest={werr_s}")
    return rows


def plot_point(label: str, rows: list[dict], out_path: Path):
    if not rows:
        return
    h = np.array([r["h"] for r in rows])
    cd = np.array([r["cd"] for r in rows])
    cl = np.array([r["cl"] for r in rows])
    werr = np.array([r["wall_profile_rel_err"] if r["wall_profile_rel_err"] is not None else np.nan for r in rows])

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 9.0), sharex=True, constrained_layout=True)
    axes[0].plot(h, cd, "o-", color="tab:red")
    axes[0].set_ylabel("$C_D$")
    axes[0].set_title(f"Convergence de maillage -- {CASE} -- {label}")
    axes[1].plot(h, cl, "o-", color="tab:blue")
    axes[1].set_ylabel("$C_L$")
    mask = np.isfinite(werr)
    axes[2].plot(h[mask], werr[mask], "o-", color="black")
    axes[2].set_ylabel("Erreur profil $M_{paroi}$\nvs. h=0.0125")
    axes[2].set_xlabel("h")
    for ax in axes:
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.grid(True, which="both", alpha=0.25)
    if np.any(mask):
        axes[2].set_yscale("log")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    figure -> {out_path}")


def main():
    import argparse
    global CASE, RAW_DIR, OUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="diamond", choices=["diamond", "diamond_sym"])
    args = parser.parse_args()
    CASE = args.case
    RAW_DIR = REPO_ROOT / "data" / "raw" / CASE
    OUT_DIR = REPO_ROOT / "data" / "validation" / f"mesh_convergence_{CASE}"
    _MESH_CACHE.clear()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_lines = ["label,mach,aoa,h_coarse,h_mid,h_fine,quantity,p_obs,extrap,gci_fine_pct"]
    raw_lines = ["label,mach,aoa,h,n_cells,cd,cl,wall_profile_rel_err"]

    for label, mach, aoa in TEST_POINTS:
        rows = run_point(label, mach, aoa)
        for row in rows:
            werr = row["wall_profile_rel_err"]
            raw_lines.append(f"{label},{mach},{aoa},{row['h']},{row['n_cells']},{row['cd']:.6f},"
                              f"{row['cl']:.6f},{'' if werr is None else f'{werr:.6e}'}")
        plot_point(label, rows, OUT_DIR / f"convergence_{label}.png")

        by_h = {r["h"]: r for r in rows}
        for qty in ("cd", "cl"):
            values_by_h = {h: by_h[h][qty] for h in H_GCI if h in by_h}
            if len(values_by_h) < 3:
                continue
            for triplet in richardson_gci(values_by_h):
                p_s = f"{triplet['p_obs']:.3f}" if triplet["p_obs"] is not None else "n/a"
                gci_s = f"{triplet['gci_fine_pct']:.3f}" if triplet["gci_fine_pct"] is not None else "n/a"
                print(f"    [{qty}] h={triplet['h_coarse']}->{triplet['h_mid']}->{triplet['h_fine']}  "
                      f"p_obs={p_s}  GCI_fine={gci_s}%")
                p_csv = "" if triplet["p_obs"] is None else f"{triplet['p_obs']:.6f}"
                extrap_csv = "" if triplet["extrap"] is None else f"{triplet['extrap']:.6f}"
                gci_csv = "" if triplet["gci_fine_pct"] is None else f"{triplet['gci_fine_pct']:.6f}"
                summary_lines.append(
                    f"{label},{mach},{aoa},{triplet['h_coarse']},{triplet['h_mid']},{triplet['h_fine']},"
                    f"{qty},{p_csv},{extrap_csv},{gci_csv}"
                )

    (OUT_DIR / "raw_values.csv").write_text("\n".join(raw_lines) + "\n")
    (OUT_DIR / "gci_summary.csv").write_text("\n".join(summary_lines) + "\n")
    print(f"\nCSV -> {OUT_DIR / 'raw_values.csv'}")
    print(f"CSV -> {OUT_DIR / 'gci_summary.csv'}")


if __name__ == "__main__":
    main()
