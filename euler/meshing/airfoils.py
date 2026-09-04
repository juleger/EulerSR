from pathlib import Path
import sys
import numpy as np
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # euler/ (jax_fvm)
sys.path.insert(0, str(Path(__file__).parent))                # meshing/ (mesh_utils)

from mesh_utils import MeshSizeParams, build_airfoil_mesh

_DAT_DIR = Path(__file__).parent / "dat"
DEFAULT_SIZE_PARAMS = MeshSizeParams()

AIRFOILS = {
    "naca0012": dict(kind="naca", m=0.0,  p=0.0, t=0.12, symmetric=False),
    "naca2412": dict(kind="naca", m=0.02, p=0.4, t=0.12, symmetric=False),
    "rae2822":  dict(kind="dat", dat="rae2822.dat", fmt="rae",      symmetric=False),
    "oneraD":   dict(kind="dat", dat="oneraD.dat",  fmt="selig",    symmetric=True),
    "oa209":    dict(kind="dat", dat="oa209.dat",   fmt="lednicer", symmetric=False),
    # Variante OOD (translation cx/cy) du profil rae2822 ci-dessus, même .dat :
    # réservée à l'évaluation géométrique hors distribution (cf. eval_case.py).
    "rae2822_txy": dict(kind="dat", dat="rae2822.dat", fmt="rae", symmetric=False),
    # NACA 5 chiffres 23012 (classique aérodynamique)
    "naca23012": dict(kind="dat", dat="naca23012.dat", fmt="selig", symmetric=False),
    # Cylindre (cercle) : cas test classique, courbure constante sur tout le
    # contour, hors distribution des formes vues en entrainement.
    "cylinder": dict(kind="circle", symmetric=True),
}


def _thickness(xc, t):
    xc = np.maximum(xc, 0.0)
    return 5*t*(0.2969*np.sqrt(xc) - 0.1260*xc - 0.3516*xc**2
                + 0.2843*xc**3 - 0.1036*xc**4)


def _camber(xc, m, p):
    if m == 0.0 or p == 0.0:
        return np.zeros_like(xc), np.zeros_like(xc)
    yc  = np.where(xc < p,
                   m/p**2*(2*p*xc - xc**2),
                   m/(1-p)**2*((1-2*p) + 2*p*xc - xc**2))
    dyc = np.where(xc < p,
                   2*m/p**2*(p - xc),
                   2*m/(1-p)**2*(p - xc))
    return yc, dyc


def _naca_contour(m, p, t, n_pts, chord, x0, y0):
    beta = np.linspace(0.0, np.pi, n_pts)
    xc   = 0.5 * (1.0 - np.cos(beta))
    yt   = _thickness(xc, t)
    yc, dyc = _camber(xc, m, p)
    theta   = np.arctan(dyc)
    upper = np.column_stack([(xc - yt*np.sin(theta))*chord + x0,
                             (yc + yt*np.cos(theta))*chord + y0])
    lower = np.column_stack([(xc + yt*np.sin(theta))*chord + x0,
                             (yc - yt*np.cos(theta))*chord + y0])[::-1]
    return np.concatenate([upper, lower[1:-1]])


def _dat_contour(cs_u, cs_l, n_pts, chord, x0, y0):
    beta  = np.linspace(0.0, np.pi, n_pts)
    xc    = 0.5 * (1.0 - np.cos(beta))
    upper = np.column_stack([xc * chord + x0, cs_u(xc) * chord + y0])
    lower = np.column_stack([xc * chord + x0, cs_l(xc) * chord + y0])[::-1]
    return np.concatenate([upper, lower[1:-1]])


def _increasing_x(arr):
    keep = [0]
    for i in range(1, len(arr)):
        if arr[i, 0] > arr[keep[-1], 0] + 1e-12:
            keep.append(i)
    return arr[keep]


def _load_dat(path, fmt):
    if fmt == "rae":
        pts = []
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                x, y = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            if 0.0 <= x <= 1.0:
                pts.append([x, y])
        pts = np.array(pts)
        if len(pts) != 130:
            raise ValueError(f"Attendu 130 points (65+65), obtenu {len(pts)} dans {path}")
        return pts[:65], pts[65:]

    if fmt == "selig":
        pts = []
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                pts.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue
        pts = np.array(pts)
        if len(pts) < 10:
            raise ValueError(f"Trop peu de points ({len(pts)}) dans {path}")
        i_le = int(np.argmin(pts[:, 0]))
        return _increasing_x(pts[:i_le + 1][::-1]), _increasing_x(pts[i_le:])

    if fmt == "lednicer":
        coords, counts = [], None
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                a, b = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            if a > 1.5 or b > 1.5:
                counts = (int(round(a)), int(round(b)))
                continue
            coords.append([a, b])
        coords = np.array(coords)
        if counts is None:
            raise ValueError(f"En-tête de comptage introuvable dans {path}")
        n_up, n_low = counts
        if len(coords) != n_up + n_low:
            raise ValueError(f"{path}: {len(coords)} points lus, {n_up + n_low} attendus")
        return _increasing_x(coords[:n_up]), _increasing_x(coords[n_up:])

    raise ValueError(f"Format .dat inconnu : {fmt}")


def _prepare_naca(spec, chord, x_le, cx, cy):
    m, p, t = spec["m"], spec["p"], spec["t"]
    t_max = t * chord
    dense = _naca_contour(m, p, t, 2000, chord, x_le, cy)
    yc_mid = float(_camber(np.array([0.5]), m, p)[0][0])
    hole_pt = (cx, cy + yc_mid * chord)
    extra = dict(thickness=t, camber=m, camber_pos=p, height=t_max,
                 center={"cx": cx, "cy": cy})
    return dense, t_max, hole_pt, extra


def _prepare_dat(spec, chord, x_le, cx, cy):
    upper, lower = _load_dat(_DAT_DIR / spec["dat"], spec["fmt"])
    cs_u = PchipInterpolator(upper[:, 0], upper[:, 1])
    cs_l = PchipInterpolator(lower[:, 0], lower[:, 1])
    scan = np.linspace(0.0, 1.0, 1000)
    t_max = float(np.max(cs_u(scan) - cs_l(scan))) * chord
    dense = _dat_contour(cs_u, cs_l, 2000, chord, x_le, cy)
    y_camber_mid = 0.5 * (float(cs_u(0.5)) + float(cs_l(0.5)))
    hole_pt = (cx, cy + y_camber_mid * chord)
    extra = dict(thickness=t_max / chord, height=t_max, center={"cx": cx, "cy": cy})
    return dense, t_max, hole_pt, extra


def _circle_contour(chord, x0, y0, n_pts):
    """Contour dense du cercle au même format que _naca_contour/_dat_contour :
    demi-cercle supérieur LE->TE (paramétré en angle, donc espacement uniforme
    en arc, pas seulement en x) suivi du demi-cercle inférieur TE->LE."""
    R = chord / 2.0
    cx = x0 + R
    beta = np.linspace(0.0, np.pi, n_pts)
    upper = np.column_stack([cx - R * np.cos(beta), y0 + R * np.sin(beta)])
    lower = np.column_stack([cx - R * np.cos(beta), y0 - R * np.sin(beta)])[::-1]
    return np.concatenate([upper, lower[1:-1]])


def _prepare_circle(spec, chord, x_le, cx, cy):
    dense = _circle_contour(chord, x_le, cy, 2000)
    # local_size_kdtree utilise ce t_max comme longueur de croissance : la distance
    # à la paroi sur laquelle la maille grossit de growth_rate. Pour les profils c'est
    # leur épaisseur (~0.1 chord) ; le diamètre du cercle (=chord) donnerait une
    # croissance ~8x plus lente et un maillage quasi uniforme, d'où ce calibrage sur
    # le même ordre de grandeur que les autres géométries. Le rayon de courbure reste
    # pris en compte via kappa et pilote seul la finesse collée à la paroi.
    t_max_growth = 0.12 * chord
    hole_pt = (cx, cy)
    extra = dict(thickness=1.0, height=chord, center={"cx": cx, "cy": cy}, radius=chord / 2.0)
    return dense, t_max_growth, hole_pt, extra


_PREPARE = {"naca": _prepare_naca, "dat": _prepare_dat, "circle": _prepare_circle}


def build_mesh(name, Lx=4.0, Ly=4.0, h=0.05, chord=1.0, cx=None, cy=None,
               export_vtk=False, size_params=DEFAULT_SIZE_PARAMS, out_dir=None):
    spec = AIRFOILS[name]
    cx = Lx / 2 if cx is None else cx
    cy = Ly / 2 if cy is None else cy
    x_le = cx - chord / 2.0
    x_te = cx + chord / 2.0
    if x_le < 0 or x_te > Lx:
        raise ValueError(f"Profil (LE={x_le:.2f}, TE={x_te:.2f}) hors du domaine [0, {Lx}].")

    dense, t_max, hole_pt, extra = _PREPARE[spec["kind"]](spec, chord, x_le, cx, cy)
    return build_airfoil_mesh(name, dense, t_max, hole_pt, Lx=Lx, Ly=Ly, h=h,
                              chord=chord, cx=cx, cy=cy, symmetric=spec["symmetric"],
                              size_params=size_params, extra_metadata=extra,
                              out_dir=out_dir, export_vtk=export_vtk)


if __name__ == "__main__":
    for name in AIRFOILS:
        for h in [0.2, 0.1, 0.05, 0.025]:
            mesh, path = build_mesh(name, h=h)
            mesh.plot_mesh(filename=str(path.with_suffix(".png")), dpi=500)
