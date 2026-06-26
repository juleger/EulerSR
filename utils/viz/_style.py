import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as _mcolors

VAR_NAMES = [r'$\rho$', r'$u$', r'$v$', r'$p$']
VAR_LABELS = ['rho', 'u', 'v', 'p']
_DPI = 500

CMAP_FIELD = 'viridis'  # champs physiques (Mach, primitives)
CMAP_ERR = 'YlOrRd'  # erreurs relatives
CMAP_GRAD = _mcolors.LinearSegmentedColormap.from_list('schlieren', ['#d4d4d4', '#000000'])

# Colormaps par primitive : rho (densité), u (vitesse x), v (vitesse y), p (pression)
CMAP_PRIMS = ['viridis', 'plasma', 'RdBu_r', 'inferno']

_SIZE = 11
plt.rcParams.update({
    'text.usetex': False,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'legend.fontsize': _SIZE,
    'axes.labelsize': _SIZE,
    'axes.titlesize': _SIZE + 3,
    'xtick.labelsize': _SIZE - 2,
    'ytick.labelsize': _SIZE - 2,
})
