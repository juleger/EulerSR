import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import tri as mtri
from matplotlib.pyplot import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.animation as animation
import jax
import jax.numpy as jnp
import sys

size = 11
params = {
	'text.usetex': False,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
	'legend.fontsize':size,
    'axes.labelsize' : size,
	'axes.titlesize' : size +3,
    'xtick.labelsize' : size-2,
    'ytick.labelsize' : size-2
}
plt.rcParams.update(params)

color = cm.rainbow(np.linspace(0, 1, 10))

sys.modules.setdefault("jax_fvm.src.mesh.plot", sys.modules[__name__])


def draw_diamond_outline(ax, mesh):
    if getattr(mesh, "metadata", {}).get("case") != "diamond":
        return

    wall_faces = np.asarray(mesh.faces)[np.asarray(mesh.face_markers) == 2]
    for face in wall_faces:
        pts = np.asarray(mesh.points)[face]
        ax.plot(pts[:, 0], pts[:, 1], color='black', lw=1.0, zorder=5)


def plot_mesh(mesh, dpi = 300, filename = 'mesh.png'):
    triang = mtri.Triangulation(mesh.points[:, 0], mesh.points[:, 1], mesh.tris)
    fig, ax = plt.subplots(dpi = dpi, figsize=(6,6))
    ax.triplot(triang, color='black', lw=0.5)
    for bc_marker in jnp.unique(mesh.face_markers)[jnp.where(jnp.unique(mesh.face_markers) > 0)[0]]:
        ids = jnp.where(mesh.face_markers == bc_marker)[0]
        for id in ids:
            ax.plot(mesh.points[mesh.faces[id]][...,0], mesh.points[mesh.faces[id]][...,1], 
                    c=color[bc_marker], lw=1.0, label= "Marker " + str(bc_marker))
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_title('Mesh')
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

def plot_solution(mesh, field_data, labels = r'$\rho$', cmap=None, dpi = 300, filename = 'solution.png', title=None, subtitle=None):
    xmin = mesh.points[:,0].min()
    xmax = mesh.points[:,0].max()
    ymin = mesh.points[:,1].min()
    ymax = mesh.points[:,1].max()

    triang = mtri.Triangulation(mesh.points[:, 0], mesh.points[:, 1], mesh.tris)

    fig, ax = plt.subplots(dpi = dpi)
    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    tpc = ax.tripcolor(triang, facecolors = field_data, cmap=cmap)
    draw_diamond_outline(ax, mesh)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    if title:
        fig.subplots_adjust(top=0.9)
        fig.suptitle(title, y=1.0, fontweight='bold', fontsize=plt.rcParams['axes.titlesize'])
    if subtitle:
        fig.text(0.5, 0.95, subtitle, ha='center', va='top', fontsize=plt.rcParams['axes.labelsize']-2)
    clb = fig.colorbar(tpc, cax = cax)
    # place colorbar label on the right side (not on top)
    clb.ax.yaxis.set_label_position('right')
    clb.ax.yaxis.set_ticks_position('right')
    clb.ax.set_ylabel(labels)
    
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

def plot_contour_solution(mesh, field_data, **kwargs):
    xmin = mesh.points[:,0].min()
    xmax = mesh.points[:,0].max()
    ymin = mesh.points[:,1].min()
    ymax = mesh.points[:,1].max()

    fig, ax = plt.subplots(dpi = 500)
    ax.set_aspect('equal')
    AR = (xmax - xmin) / (ymax - ymin)
    N = kwargs.get('N', 128)
    # Create grid values first.
    xi = np.linspace(xmin, xmax, int(N * AR))
    yi = np.linspace(ymin, ymax, N)

    # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
    triang = mtri.Triangulation(mesh.barycenter[:, 0], mesh.barycenter[:, 1])
    interpolator = mtri.LinearTriInterpolator(triang, field_data)

    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    triang = mtri.Triangulation(mesh.points[:, 0], mesh.points[:, 1], mesh.tris)
    tpc = ax.tripcolor(triang, facecolors = field_data, cmap=kwargs.get('cmap', None))
    draw_diamond_outline(ax, mesh)
    clb = fig.colorbar(tpc, cax = cax)
    # place colorbar label on the right side
    clb.ax.yaxis.set_label_position('right')
    clb.ax.yaxis.set_ticks_position('right')
    clb.ax.set_ylabel(kwargs.get('labels', r'$\rho$'))
    ax.contour(xi, yi, zi, levels=kwargs.get('levels', 20), linewidths=0.5, colors='k')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    title = kwargs.get('title', None)
    subtitle = kwargs.get('subtitle', None)
    if title is not None:
        fig.subplots_adjust(top=0.9)
        fig.suptitle(title, fontweight='bold', y=0.98)
        if subtitle is not None:
            fig.text(0.5, 0.95, subtitle, ha='center', va='top', fontsize=max(8, plt.rcParams['axes.titlesize'] - 5))
    plt.tight_layout()


def animate_solution(mesh, field_sequence, path = "animation.gif", interval=200):
    triang = mtri.Triangulation(mesh.points[:, 0], mesh.points[:, 1], mesh.tris)

    xmin = mesh.points[:,0].min()
    xmax = mesh.points[:,0].max()
    ymin = mesh.points[:,1].min()
    ymax = mesh.points[:,1].max()

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    def update(frame):
        ax.tripcolor(triang, facecolors = field_sequence[frame])

    ani = animation.FuncAnimation(fig, update, frames=len(field_sequence), interval=interval)

    ani.save(path, writer='imagemagick')

def plot(y, labels = r'$E_k$'):
    fig, ax = plt.subplots()
    ax.plot(y)
    ax.grid()
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(labels)


def getSliceinMesh(Prim, mesh, y = 0.5, n = 200):
	x_min, x_max = jnp.min(mesh.barycenter[...,0]), jnp.max(mesh.barycenter[...,0])
	x = jnp.linspace(x_min, x_max, n)
	slice = jnp.zeros((n,2))
	slice = slice.at[:,0].set(x)
	slice = slice.at[:,1].set(y)
	cdist = jnp.linalg.norm(slice[:, None] - mesh.barycenter[None], axis=-1)
	_, id = jax.lax.top_k(-cdist, k=1)
	out = Prim[id].squeeze()
	return out, slice
