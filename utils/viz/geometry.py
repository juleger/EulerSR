"""
Visualisation propre des géométries CFD (diamond, NACA0012) sans maillage.
Montre domaine, conditions aux limites et stratégie AoA.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc, FancyArrowPatch

# ── couleurs ────────────────────────────────────────────────────────────────
C_INLET = "#1565C0"  # bleu  – bord gauche
C_OUTLET = "#B71C1C"  # rouge – bords haut/bas/droite
C_WALL = "#263238"  # quasi-noir – obstacle
C_FILL = "#B0BEC5"  # gris clair – remplissage obstacle
C_DOMAIN = "#EEF2FB"  # fond du domaine
C_DIM = "#444444"  # cotes
C_AOA = "#1565C0"  # flèche AoA (= même bleu que inlet)
BW = 2.5  # line-width bords domaine


# ── helpers ─────────────────────────────────────────────────────────────────

def _naca_contour(t=0.12, n=400):
    """Contour NACA 4 chiffres symétrique, chord=1, bord d'attaque en x=0."""
    beta = np.linspace(0, np.pi, n)
    xc = 0.5 * (1 - np.cos(beta))
    yt = 5*t*(0.2969*np.sqrt(np.maximum(xc, 0))
                - 0.1260*xc - 0.3516*xc**2 + 0.2843*xc**3 - 0.1036*xc**4)
    upper = np.column_stack([xc, yt])
    lower = np.column_stack([xc, -yt])[::-1]
    return np.vstack([upper, lower[1:-1], upper[:1]])


def _diamond_verts(cx, cy, hc, ht):
    return np.array([[cx-hc, cy], [cx, cy+ht],
                     [cx+hc, cy], [cx, cy-ht], [cx-hc, cy]])


def _draw_domain(ax, Lx, Ly):
    ax.fill([0, Lx, Lx, 0, 0], [0, 0, Ly, Ly, 0],
            fc=C_DOMAIN, ec="none", zorder=0)
    ax.plot([0, 0], [0, Ly], color=C_INLET, lw=BW, solid_capstyle="round", zorder=5)
    ax.plot([Lx, Lx], [0, Ly], color=C_OUTLET, lw=BW, solid_capstyle="round", zorder=5)
    ax.plot([0, Lx], [0, 0], color=C_OUTLET, lw=BW, solid_capstyle="round", zorder=5)
    ax.plot([0, Lx], [Ly, Ly],color=C_OUTLET, lw=BW, solid_capstyle="round", zorder=5)


def _dim_arrow(ax, p0, p1, label, perp_offset=(0, 0), fontsize=8.5,
               color=C_DIM, ha="center", va="center", rot=0):
    """Flèche bilatérale avec étiquette au milieu."""
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.1,
                                mutation_scale=7),
                annotation_clip=False)
    mid = ((p0[0]+p1[0])/2 + perp_offset[0],
           (p0[1]+p1[1])/2 + perp_offset[1])
    ax.text(*mid, label, ha=ha, va=va, fontsize=fontsize, color=color,
            rotation=rot, clip_on=False)


def _draw_aoa_arrow(ax, x0, y0, aoa_deg, arrow_len=0.65, label=True):
    """
    Flèche U∞ à angle aoa_deg depuis l'horizontale.
    La géométrie reste fixe — c'est le flux qui est incliné.
    """
    aoa_rad = np.radians(aoa_deg)
    dx = arrow_len * np.cos(aoa_rad)
    dy = arrow_len * np.sin(aoa_rad)
    ax.annotate("", xy=(x0+dx, y0+dy), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=C_AOA, lw=2.2,
                                mutation_scale=13),
                annotation_clip=False, zorder=12)
    # référence horizontale en pointillés
    ax.plot([x0, x0+dx], [y0, y0], color=C_AOA, lw=1.0,
            ls="--", alpha=0.5, zorder=11, clip_on=False)
    # arc d'angle
    r_arc = 0.20
    theta = np.linspace(0, aoa_rad, 60)
    ax.plot(x0 + r_arc*np.cos(theta), y0 + r_arc*np.sin(theta),
            color=C_AOA, lw=1.2, zorder=11, clip_on=False)
    if label:
        ax.text(x0 + r_arc*1.6, y0 + 0.02,
                rf"$\alpha$", fontsize=9.5, color=C_AOA,
                ha="left", va="bottom", clip_on=False)
        ax.text(x0 - 0.06, y0 + dy*0.6,
                rf"$U_{{\infty}}$", fontsize=9.5, color=C_AOA,
                ha="right", va="center", clip_on=False)


# ── Diamond ─────────────────────────────────────────────────────────────────

def plot_diamond(ax, Lx=4.0, Ly=4.0, chord=1.0, beta_deg=15.0,
                 cx=1.5, cy=2.0, aoa_deg=2.0):
    """
    beta_deg : demi-angle de la pointe (angle géométrique de l'aile)
    aoa_deg  : incidence du flux libre — flèche, géométrie fixe
    """
    height = chord * np.tan(np.radians(beta_deg))
    hc, ht = chord / 2, height / 2

    _draw_domain(ax, Lx, Ly)

    # obstacle
    verts = _diamond_verts(cx, cy, hc, ht)
    ax.fill(verts[:, 0], verts[:, 1], fc=C_FILL, ec=C_WALL, lw=2.0, zorder=6)

    # étiquette slip wall
    ax.text(cx + hc + 0.08, cy, "slip wall",
            ha="left", va="center", fontsize=7.5,
            color=C_WALL, fontstyle="italic", zorder=10)

    # ─ AoA arrow (flux incliné, pas la géométrie) ─
    _draw_aoa_arrow(ax, x0=-0.05, y0=cy, aoa_deg=aoa_deg)

    # ─ cote chord ─
    y_ch = cy - ht - 0.18
    ax.plot([cx-hc, cx-hc], [cy-ht, y_ch], color=C_DIM, lw=0.7, ls=":", clip_on=False)
    ax.plot([cx+hc, cx+hc], [cy-ht, y_ch], color=C_DIM, lw=0.7, ls=":", clip_on=False)
    _dim_arrow(ax, (cx-hc, y_ch), (cx+hc, y_ch),
               rf"$c = {chord:.1f}$", perp_offset=(0, -0.1))

    # ─ cote hauteur ─
    x_ht = cx + hc + 0.45
    ax.plot([cx+hc, x_ht], [cy+ht, cy+ht], color=C_DIM, lw=0.7, ls=":", clip_on=False)
    ax.plot([cx+hc, x_ht], [cy-ht, cy-ht], color=C_DIM, lw=0.7, ls=":", clip_on=False)
    _dim_arrow(ax, (x_ht, cy-ht), (x_ht, cy+ht),
               rf"$h={height:.2f}c$", perp_offset=(0.12, 0),
               ha="left", va="center")

    # ─ demi-angle de la pointe ─
    face_angle = np.degrees(np.arctan2(ht, hc))
    r_b = 0.18
    theta_b = np.linspace(0, np.radians(face_angle), 50)
    ax.plot(cx - hc + r_b*np.cos(theta_b), cy + r_b*np.sin(theta_b),
            color="#E65100", lw=1.4, zorder=10, clip_on=False)
    ax.text(cx - hc + r_b*1.55, cy + 0.03,
            rf"$\beta={beta_deg:.0f}°$",
            fontsize=8.5, color="#E65100", ha="left", va="bottom")

    # ─ labels LE / TE ─
    ax.text(cx - hc - 0.05, cy, "LE", ha="right", va="center",
            fontsize=7, color=C_WALL, fontstyle="italic")
    ax.text(cx + hc + 0.05, cy + ht + 0.08, "TE", ha="left", va="bottom",
            fontsize=7, color=C_WALL, fontstyle="italic")

    # ─ dimensions domaine ─
    _dim_arrow(ax, (0, -0.38), (Lx, -0.38),
               rf"$L_x = {Lx:.0f}\,c$", perp_offset=(0, -0.12))
    _dim_arrow(ax, (-0.48, 0), (-0.48, Ly),
               rf"$L_y = {Ly:.0f}\,c$", perp_offset=(-0.15, 0),
               rot=90, ha="center", va="center")

    # ─ position LE dans le domaine ─
    _dim_arrow(ax, (0, -0.12), (cx-hc, -0.12),
               rf"$x_{{LE}}={cx-hc:.1f}c$", perp_offset=(0, -0.08),
               fontsize=7.5)

    # ─ labels BC ─
    ax.text(0.06, Ly*0.5, "Inlet\n" + r"$(\rho_\infty,\mathbf{U}_\infty,p_\infty)$",
            ha="left", va="center", fontsize=7, color=C_INLET,
            rotation=90, zorder=8)
    ax.text(Lx*0.65, Ly + 0.08, "Outlet / far-field",
            ha="center", va="bottom", fontsize=7, color=C_OUTLET,
            clip_on=False)

    ax.set_xlim(-0.9, Lx + 0.85)
    ax.set_ylim(-0.75, Ly + 0.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(rf"Diamond  ($\beta={beta_deg:.0f}°$, $\alpha_{{AoA}}={aoa_deg:.0f}°$)",
                 fontsize=10.5, pad=7)


# ── NACA 0012 ────────────────────────────────────────────────────────────────

def plot_naca(ax, Lx=4.0, Ly=4.0, chord=1.0, t=0.12,
              cx=1.5, cy=2.0, aoa_deg=2.0, name="NACA 0012"):
    x_le = cx - chord / 2
    max_t = t * chord

    _draw_domain(ax, Lx, Ly)

    # profil
    contour = _naca_contour(t=t, n=500)
    ax.fill(contour[:, 0]*chord + x_le, contour[:, 1]*chord + cy,
            fc=C_FILL, ec=C_WALL, lw=2.0, zorder=6)

    # slip wall label
    x_lbl = x_le + 0.26*chord
    y_lbl = cy + max_t
    ax.annotate("slip wall",
                xy=(x_lbl, y_lbl), xytext=(x_lbl + 0.28, y_lbl + 0.18),
                fontsize=7.5, color=C_WALL, fontstyle="italic",
                arrowprops=dict(arrowstyle="-", color=C_WALL, lw=0.8),
                annotation_clip=False, zorder=10)

    # AoA arrow
    _draw_aoa_arrow(ax, x0=-0.05, y0=cy, aoa_deg=aoa_deg)

    # ─ cote chord ─
    y_ch = cy - max_t - 0.18
    ax.plot([x_le, x_le], [cy-max_t*0.6, y_ch],
            color=C_DIM, lw=0.7, ls=":", clip_on=False)
    ax.plot([x_le+chord, x_le+chord], [cy, y_ch],
            color=C_DIM, lw=0.7, ls=":", clip_on=False)
    _dim_arrow(ax, (x_le, y_ch), (x_le+chord, y_ch),
               rf"$c = {chord:.1f}$", perp_offset=(0, -0.1))

    # ─ épaisseur max ─
    xt = x_le + 0.30*chord  # ~30% chord : max thickness NACA0012
    yt_top = cy + max_t
    yt_bot = cy - max_t
    x_th = xt + 0.38
    ax.plot([xt, x_th], [yt_top, yt_top], color=C_DIM, lw=0.7, ls=":", clip_on=False)
    ax.plot([xt, x_th], [yt_bot, yt_bot], color=C_DIM, lw=0.7, ls=":", clip_on=False)
    _dim_arrow(ax, (x_th, yt_bot), (x_th, yt_top),
               rf"t/c={t*100:.0f}%", perp_offset=(0.12, 0),
               ha="left", va="center")

    # ─ LE radius (qualitatif) ─
    ax.annotate("bord d'attaque\n(rayon $r_{{LE}}$)",
                xy=(x_le, cy), xytext=(x_le - 0.45, cy + 0.30),
                fontsize=7, color=C_WALL,
                arrowprops=dict(arrowstyle="-|>", color=C_WALL, lw=0.8,
                                mutation_scale=8),
                annotation_clip=False, ha="center", zorder=10)

    # ─ labels LE / TE ─
    ax.text(x_le + chord + 0.05, cy, "TE", ha="left", va="center",
            fontsize=7, color=C_WALL, fontstyle="italic")

    # ─ dimensions domaine ─
    _dim_arrow(ax, (0, -0.38), (Lx, -0.38),
               rf"$L_x = {Lx:.0f}\,c$", perp_offset=(0, -0.12))
    _dim_arrow(ax, (-0.48, 0), (-0.48, Ly),
               rf"$L_y = {Ly:.0f}\,c$", perp_offset=(-0.15, 0),
               rot=90, ha="center", va="center")

    # ─ position LE ─
    _dim_arrow(ax, (0, -0.12), (x_le, -0.12),
               rf"$x_{{LE}}={x_le:.1f}c$", perp_offset=(0, -0.08),
               fontsize=7.5)

    # ─ labels BC ─
    ax.text(0.06, Ly*0.5, "Inlet\n" + r"$(\rho_\infty,\mathbf{U}_\infty,p_\infty)$",
            ha="left", va="center", fontsize=7, color=C_INLET,
            rotation=90, zorder=8)
    ax.text(Lx*0.65, Ly + 0.08, "Outlet / far-field",
            ha="center", va="bottom", fontsize=7, color=C_OUTLET,
            clip_on=False)

    ax.set_xlim(-0.9, Lx + 0.85)
    ax.set_ylim(-0.75, Ly + 0.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(rf"{name}  (t/c={t*100:.0f}%,  $\alpha_{{AoA}}={aoa_deg:.0f}°$)",
                 fontsize=10.5, pad=7)


# ── figure principale ────────────────────────────────────────────────────────

def make_geometry_plots(out_path=None, show=True):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2),
                              facecolor="white", gridspec_kw={"wspace": 0.12})

    plot_diamond(axes[0], Lx=4.0, Ly=4.0, chord=1.0, beta_deg=15.0,
                 cx=1.5, cy=2.0, aoa_deg=2.0)
    plot_naca(axes[1], Lx=4.0, Ly=4.0, chord=1.0, t=0.12,
              cx=1.5, cy=2.0, aoa_deg=2.0)

    legend_handles = [
        mpatches.Patch(fc=C_INLET, label="Inlet  — $\\rho_\\infty, U_\\infty, p_\\infty$ imposés"),
        mpatches.Patch(fc=C_OUTLET, label="Outlet / far-field"),
        mpatches.Patch(fc=C_FILL, ec=C_WALL, lw=1.2, label="Obstacle — slip wall (imperméable, inv.)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=9, framealpha=0.92, edgecolor="#ccc",
               bbox_to_anchor=(0.5, -0.01))

    note = (r"Stratégie AoA : la géométrie reste fixe — c'est $\mathbf{U}_\infty$ "
            r"qui est inclinée de $\alpha$ : $u_\infty = M_\infty c_\infty\cos\alpha$,"
            r"  $v_\infty = M_\infty c_\infty\sin\alpha$")
    fig.text(0.5, -0.06, note, ha="center", va="top",
             fontsize=8.5, color="#555", style="italic")

    fig.suptitle("Géométries CFD — domaines, conditions aux limites et stratégie AoA",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    if out_path is None:
        out_path = Path(__file__).resolve().parents[2] / "figures" / "geometries.pdf"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    print(f"Saved → {out_path}")
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    make_geometry_plots(show=True)
