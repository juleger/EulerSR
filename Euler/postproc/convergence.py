from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


repo_root = Path(__file__).resolve().parents[2]


def load_records(results_root: Path, case: str, mach_target: float):
    records_by_flux = {}
    results_root = Path(results_root)

    for json_path in sorted((results_root / case).rglob('*.json')):
        if json_path.name == 'config.json':
            continue
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except Exception:
            continue

        mach = data.get('mach')
        h = data.get('h')
        flux = data.get('flux')
        cd = data.get('cd')
        cl = data.get('cl')
        delta_s = data.get('deltaS')

        if mach is None or h is None or flux is None:
            continue
        try:
            mach = float(mach)
            h = float(h)
        except Exception:
            continue

        if abs(mach - float(mach_target)) > 1e-8:
            continue
        if cd is None or cl is None or delta_s is None:
            continue

        flux_key = str(flux).upper()
        records_by_flux.setdefault(flux_key, []).append({'h': h, 'cd': float(cd), 'cl': float(cl), 'deltaS': float(delta_s), 'path': json_path})

    for k, v in records_by_flux.items():
        records_by_flux[k] = sorted(v, key=lambda x: x['h'])

    return records_by_flux


def plot_convergence(records_by_flux, case: str, mach: float, figures_root: Path):
    figures_root = Path(figures_root)
    out_dir = figures_root / case / f'convergence_mach{mach:.2f}'
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for flux, records in sorted(records_by_flux.items()):
        hs = np.array([r['h'] for r in records], dtype=float)
        cds = np.array([r['cd'] for r in records], dtype=float)
        cls = np.array([r['cl'] for r in records], dtype=float)
        dS = np.array([r['deltaS'] for r in records], dtype=float)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        fig.suptitle(f'Convergence en maillage — case={case} Mach={mach} flux={flux}', fontweight='bold')

        ax = axes[0]
        ax.semilogx(hs, cds, marker='o', label=r'$C_D$')
        ax.semilogx(hs, cls, marker='s', label=r'$C_L$')
        ax.set_xlabel(r'Taille de maille $h$')
        ax.set_ylabel('Coefficient')
        ax.grid(True, which='both', linestyle='--', alpha=0.4)
        for x, y in zip(hs, cds):
            ax.annotate(f'{x:.5g}', xy=(x, y), xytext=(0, 6), textcoords='offset points', fontsize=7, ha='center')
        for x, y in zip(hs, cls):
            ax.annotate(f'{x:.5g}', xy=(x, y), xytext=(0, -10), textcoords='offset points', fontsize=7, ha='center')
        ax.legend(frameon=True)
        ax = axes[1]
        ax.semilogx(hs, dS, marker='o', color='tab:green', label=r'$\Delta S$')
        ax.set_xlabel(r'Taille de maille $h$')
        ax.set_ylabel(r'Entropie $\Delta S$')
        dS_max = float(np.max(dS)) if dS.size > 0 else 0.0
  
        if dS_max > 0.0:
            ymax = dS_max * (1.0 + 0.2)
        else:
            ymax = 1e-12
        ax.set_ylim(0, ymax)
        ax.grid(True, which='both', linestyle='--', alpha=0.4)
        ax.legend(frameon=True)
    
        for x, y in zip(hs, dS):
            ax.annotate(f'{x:.5g}', xy=(x, y), xytext=(0, 6), textcoords='offset points', fontsize=7, ha='center')

        save_name = out_dir / f'convergence_mach{mach:.2f}_{flux.lower()}.png'
        fig.savefig(save_name, dpi=300, bbox_inches='tight')
        plt.close(fig)
        saved.append(save_name)
        print('Saved', save_name)

    return saved


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Convergence plots for Euler cases')
    parser.add_argument('--case', default='diamond')
    parser.add_argument('--mach', type=float, default=0.9)
    parser.add_argument('--results-root', default=str(repo_root / 'results'))
    parser.add_argument('--figures-root', default=str(repo_root / 'figures'))
    args = parser.parse_args()

    records = load_records(Path(args.results_root), args.case, args.mach)
    plot_convergence(records, args.case, args.mach, Path(args.figures_root))

if __name__ == '__main__':
    main()
