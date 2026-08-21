"""
Entraînement des modèles de super-résolution CFD (JAX/Flax) : parsing de la
config/CLI, chargement des datasets (single ou multi-géométrie), construction
du modèle (DAM/FAM/SIAM), fit() avec logging CSV et plots de validation
périodiques sur les cas de référence.
"""


import sys
import argparse
import importlib
from datetime import datetime
from pathlib import Path

import numpy as np
from flax import nnx

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / 'euler'))
import euler.jax_fvm.src.mesh  # noqa: F401

from preprocessing.dataset import SRDataset, MultiSRDataset
from models.base import TrainConfig, load_cfg
from utils.viz.training import plot_val_panels
from utils.refs import REFERENCE_CASES, NACA_REFERENCE_CASES, find_ref_file, _res_tag
from utils.layout import DataLayout

_MODELS = {
    'dam': ('models.dam', 'DAM'),
    'fam': ('models.fam', 'FAM'),
    'siam': ('models.siam', 'SIAM'),
}


def _parse_range(spec: str | None):
    if spec is None:
        return None
    parts = spec.split(':')
    return (float(parts[0]), float(parts[1]))


def _per_branch(val, n, default):
    """Étend une valeur de config en liste par branche.
    None -> [default]*n ; scalaire -> répété ; liste -> alignée sur lr_res."""
    if val is None:
        return [default] * n
    if isinstance(val, list):
        if len(val) != n:
            raise ValueError(
                f"datasets: la liste {val} doit avoir {n} éléments (un par lr_res)")
        return list(val)
    return [val] * n


def build_branches(datasets_cfg, data_root, hr_res, lr_res,
                   mach_range, aoa_range, default_train_fraction, aoa_step=None):
    """Construit les branches du mode multi-dataset.

    Une branche = un couple (géométrie, résolution_LR).
    Les branches d'une même géométrie partagent le même geom_id (embedding), la
    résolution étant conditionnée séparément.
    """
    branches, geom_to_id = [], {}
    for ds_cfg in datasets_cfg:
        geom = ds_cfg['name']
        gid = geom_to_id.setdefault(geom, len(geom_to_id))
        lr_list = ds_cfg.get('lr_res', [lr_res])
        if not isinstance(lr_list, list):
            lr_list = [lr_list]
        mr = tuple(ds_cfg['mach_range']) if ds_cfg.get('mach_range') else mach_range
        ar = tuple(ds_cfg['aoa_range']) if ds_cfg.get('aoa_range') else aoa_range
        as_ = ds_cfg['aoa_step'] if ds_cfg.get('aoa_step') is not None else aoa_step
        w_list = _per_branch(ds_cfg.get('weight'), len(lr_list), 1.0)
        tf_list = _per_branch(ds_cfg.get('train_fraction'), len(lr_list), default_train_fraction)
        for r, w, tf in zip(lr_list, w_list, tf_list):
            name = geom if len(lr_list) == 1 else f'{geom}_lr{_res_tag(r)}'
            branches.append({
                'name': name, 'geom': geom, 'geom_id': gid, 'lr_res': r,
                'layout': DataLayout.from_root(data_root, geom, r, hr_res),
                'weight': float(w), 'mach_range': mr, 'aoa_range': ar, 'aoa_step': as_,
                'train_fraction': float(tf),
            })
    return branches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    parser.add_argument('--model', default='dam')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--wd', type=float, default=None)
    parser.add_argument('--val_every', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--loss', default=None, choices=['mse', 'rel_mse', 'shock_weighted_mse', 'shock_weighted_rel_mse', 'w2'])
    parser.add_argument('--schedule', default=None, choices=['cosine', 'constant'])
    parser.add_argument('--warmup_epochs', type=int, default=None)
    parser.add_argument('--grad_clip', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--run_name', default=None)
    parser.add_argument('--init_from', default=None, help='Checkpoint .pkl pour fine-tuning')
    parser.add_argument('--no_save_best', action='store_true')
    parser.add_argument('--data', default='data/', help='Racine des données (ex: data/)')
    parser.add_argument('--geometry', default=None, help='Géométrie pour mode single-dataset (ex: diamond)')
    parser.add_argument('--mach', default=None, help='start:stop')
    parser.add_argument('--aoa', default=None, help='start:stop')
    parser.add_argument('--train_fraction', type=float, default=None,
                        help='Fraction du train set à utiliser (ex: 0.25). Seed fixé pour reproductibilité.')
    parser.add_argument('--aoa_step', type=float, default=None, help='Pas AoA pour le train set (ex: 1.0)')
    args = parser.parse_args()

    cfg = load_cfg(args.config) if args.config else {}
    model_name = cfg.get('model', args.model)
    args.train_fraction = args.train_fraction if args.train_fraction is not None else cfg.get('training', {}).get('train_fraction', 1.0)
    args.aoa_step = args.aoa_step if args.aoa_step is not None else cfg.get('training', {}).get('aoa_step')

    if model_name not in _MODELS:
        raise ValueError(f"Unknown model: {model_name!r}. Options: {list(_MODELS)}")
    mod_path, cls_name = _MODELS[model_name]
    cls = getattr(importlib.import_module(mod_path), cls_name)

    tr = cfg.get('training', {})

    def _get(cli_val, key, default):
        return cli_val if cli_val is not None else tr.get(key, default)

    train_cfg = TrainConfig(
        epochs = _get(args.epochs, 'epochs', 200),
        lr = _get(args.lr, 'lr', 3e-4),
        weight_decay = _get(args.wd, 'weight_decay', 1e-5),
        val_every = _get(args.val_every, 'val_every', 10),
        seed = _get(args.seed, 'seed', 42),
        loss = _get(args.loss, 'loss', 'rel_mse'),
        lambda_phys = tr.get('lambda_phys', 0.0),
        lambda_enthalpy = tr.get('lambda_enthalpy', 0.0),
        lambda_endpoint = tr.get('lambda_endpoint', 0.0),
        lambda_endpoint_warmup_epochs = tr.get('lambda_endpoint_warmup_epochs', 0),
        schedule = _get(args.schedule, 'schedule', 'cosine'),
        warmup_epochs = _get(args.warmup_epochs, 'warmup_epochs', 0),
        grad_clip = _get(args.grad_clip, 'grad_clip', 0.0),
        batch_size = _get(args.batch_size, 'batch_size', 1),
        save_best = (not args.no_save_best) and tr.get('save_best', True),
        ema_decay = tr.get('ema_decay', 0.0),
    )

    res_cfg = cfg.get('resolution', {})
    lr_res = res_cfg.get('lr', 0.1)
    hr_res = res_cfg.get('hr', 0.025)
    mach_range = _parse_range(args.mach)
    aoa_range = _parse_range(args.aoa)
    use_lr_grad = cfg.get('architecture', {}).get('use_lr_grad', False)
    coord_norm = cfg.get('architecture', {}).get('coord_norm', 'domain')
    preload = tr.get('preload', True)
    sw_factor = tr.get('shock_weight_factor', 1.0)

    data_root = Path(args.data).resolve()

    # Dataset multi geometrie ou single dataset
    if 'datasets' in cfg:
        branches = build_branches(cfg['datasets'], data_root, hr_res, lr_res,
                                  mach_range, aoa_range, args.train_fraction, args.aoa_step)
        # n_geoms = nombre de géométries distinctes (taille de l'embedding)
        cfg.setdefault('architecture', {})['n_geoms'] = len({b['geom_id'] for b in branches})
        print(f"  Branches ({len(branches)}) :")
        for b in branches:
            print(f"    [{b['name']:<16}] geom_id={b['geom_id']}  lr={b['lr_res']}"
                  f"  weight={b['weight']}  train_fraction={b['train_fraction']}")

        train_list, val_list, names, weights, knns = [], [], [], [], {}
        for b in branches:
            train_list.append(SRDataset(b['layout'], 'train', use_lr_grad=use_lr_grad,
                mach_range=b['mach_range'], aoa_range=b['aoa_range'], aoa_step=b['aoa_step'], preload=preload,
                shock_weight_factor=sw_factor, geom_id=b['geom_id'], train_fraction=b['train_fraction'],
                coord_norm=coord_norm))
            val_list.append(SRDataset(b['layout'], 'val', use_lr_grad=use_lr_grad,
                mach_range=b['mach_range'], aoa_range=b['aoa_range'], preload=preload,
                shock_weight_factor=sw_factor, geom_id=b['geom_id'], coord_norm=coord_norm))
            names.append(b['name'])
            weights.append(b['weight'])
            knns[b['name']] = cls.load_knn(b['layout'], cfg)
        train_ds = MultiSRDataset(train_list, names, weights)
        val_ds = MultiSRDataset(val_list, names, weights)
        # Ressources primaires (première branche)
        primary_layout = branches[0]['layout']
        knn = knns
        stats = np.load(primary_layout.stats_path)

        # Ressources secondaires pour plots par branche
        secondary_plot_data = []
        for b in branches[1:]:
            layout_s = b['layout']
            name_s = b['name']
            stats_s = np.load(layout_s.stats_path)
            try:
                mesh_hr_s = np.load(layout_s.mesh_path(hr_res), allow_pickle=True).item()
            except FileNotFoundError as e:
                print(f"  mesh_hr introuvable pour {name_s} — plots désactivés ({e})")
                continue
            try:
                mesh_lr_s = np.load(layout_s.mesh_path(b['lr_res']), allow_pickle=True).item()
            except FileNotFoundError:
                mesh_lr_s = None
            tdir_s = layout_s.proc_dir() / 'test'
            af_s = [str(f) for f in sorted(tdir_s.glob('aoa*.npz'))] if tdir_s.exists() else []
            rc_s = []
            ref_pool = NACA_REFERENCE_CASES if 'naca' in b['geom'].lower() else REFERENCE_CASES
            for mt, at, lbl in ref_pool:
                p = find_ref_file(af_s, mt, at)
                if p:
                    rc_s.append((p, lbl))
            if not rc_s and af_s:
                n_af = len(af_s)
                for idx_af in sorted({0, n_af // 2, n_af - 1}):
                    rc_s.append((af_s[idx_af], Path(af_s[idx_af]).stem))
            if rc_s:
                print(f"  Cas référence [{name_s}] : {[l for _, l in rc_s]}")
            secondary_plot_data.append({
                'name': name_s, 'knn': knns[name_s],
                'stats': stats_s, 'mesh_hr': mesh_hr_s, 'mesh_lr': mesh_lr_s,
                'ref_cases': rc_s,
            })
    else:
        secondary_plot_data = []
        # Mode single-dataset
        geom = args.geometry or cfg.get('geometry')
        if not geom:
            raise ValueError(
                "Géométrie non spécifiée : utiliser --geometry diamond "
                "ou ajouter 'geometry: diamond' dans le config YAML")
        primary_layout = DataLayout.from_root(data_root, geom, lr_res, hr_res)
        knn = cls.load_knn(primary_layout, cfg)
        train_ds = SRDataset(primary_layout, 'train', use_lr_grad=use_lr_grad, mach_range=mach_range,
            aoa_range=aoa_range, aoa_step=args.aoa_step, preload=preload, shock_weight_factor=sw_factor,
            train_fraction=args.train_fraction, coord_norm=coord_norm)
        val_ds = SRDataset(primary_layout, 'val', use_lr_grad=use_lr_grad, mach_range=mach_range,
            aoa_range=aoa_range, preload=preload, shock_weight_factor=sw_factor, coord_norm=coord_norm)
        stats = np.load(primary_layout.stats_path)

    cfg.setdefault('architecture', {})['use_lr_grad'] = (
        train_ds.datasets[0]._has_lr_grad if isinstance(train_ds, MultiSRDataset)
        else train_ds._has_lr_grad)
    cfg.setdefault('architecture', {})['coord_norm'] = coord_norm

    mesh_hr = np.load(primary_layout.mesh_path(hr_res), allow_pickle=True).item()
    mesh_lr = np.load(primary_layout.mesh_path(primary_layout.lr_res), allow_pickle=True).item()

    test_dir = primary_layout.proc_dir() / 'test'
    train_dir = primary_layout.proc_dir() / 'train'
    all_files = [str(f) for f in sorted(test_dir.glob('aoa*.npz'))] if test_dir.exists() else []
    if train_dir.exists():
        all_files += [str(f) for f in sorted(train_dir.glob('aoa*.npz'))]

    run_label = args.run_name or f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    pred_dir = Path('results') / 'predictions' / run_label

    ref_cases = []
    for mach_t, aoa_t, label in REFERENCE_CASES:
        path = find_ref_file(all_files, mach_t, aoa_t)
        if path is not None:
            ref_cases.append((path, label))
    if not ref_cases:
        print(f"  Aucun cas de référence trouvé dans {test_dir} — "
              f"pred_callback désactivé. Vérifier le pattern aoa*.npz et la structure de dossiers.")
    else:
        print(f"  Cas de référence pour pred_callback : {[l for _, l in ref_cases]}")

    pred_dir.mkdir(parents=True, exist_ok=True)

    def pred_callback(m, epoch, knn_):
        try:
            plot_val_panels(m, knn_, ref_cases, stats, mesh_hr, mesh_lr,
                            pred_dir / f'ep{epoch:04d}.png',
                            title=f'{run_label} — epoch {epoch}', coord_norm=coord_norm)
            # Plots par géométrie secondaire
            for sec in secondary_plot_data:
                if not sec['ref_cases']:
                    continue
                plot_val_panels(m, sec['knn'], sec['ref_cases'], sec['stats'], sec['mesh_hr'], sec['mesh_lr'],
                    pred_dir / f'ep{epoch:04d}_{sec["name"]}.png', title=f'{run_label} [{sec["name"]}] — epoch {epoch}',
                    coord_norm=coord_norm)
        except Exception as e:
            print(f"  pred_callback epoch {epoch} : {e}")

    # knn primaire pour les callbacks visuels (plot_val_panels attend un seul dict)
    primary_knn = (list(knn.values())[0] if isinstance(knn, dict) and
                   all(isinstance(v, dict) for v in knn.values()) else knn)

    model = cls(nnx.Rngs(train_cfg.seed), cfg)
    if args.init_from:
        print(f"\n── Initialisation depuis {args.init_from} (fine-tuning) ──────────")
        pretrained = cls.load(args.init_from)
        nnx.update(model, nnx.state(pretrained))
        print("  Params + buffers (res_scale, mu_train/sig_train, ...) repris du checkpoint.")

    trained = model.fit(train_ds, val_ds, knn, train_cfg, out_dir='results/checkpoints',
        model_cfg=cfg, run_name=run_label, pred_callback=pred_callback)

    plot_val_panels(trained, primary_knn, ref_cases, stats, mesh_hr, mesh_lr,
                    pred_dir / 'best.png', title=f'{run_label} — best', coord_norm=coord_norm)
    for sec in secondary_plot_data:
        if sec['ref_cases']:
            plot_val_panels(trained, sec['knn'], sec['ref_cases'], sec['stats'], sec['mesh_hr'], sec['mesh_lr'],
                    pred_dir / f'best_{sec["name"]}.png', title=f'{run_label} [{sec["name"]}] — best',
                    coord_norm=coord_norm)
    print(f"Prediction plots: {pred_dir}/")


if __name__ == '__main__':
    main()
