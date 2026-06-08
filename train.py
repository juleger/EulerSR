import sys
import argparse
import importlib
from pathlib import Path

import numpy as np
import jax
from flax import nnx

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / 'euler'))
import euler.jax_fvm.src.mesh  # noqa: F401

from preprocessing.dataset import SRDataset
from models.base import TrainConfig, load_cfg
from utils.viz.training import plot_prediction
from utils.refs import REFERENCE_CASES, find_ref_file

_MODELS = {
    'mlp_corrector': ('models.deterministic.mlp_corrector', 'MlpCorrector'),
    'localnet':      ('models.deterministic.localnet',      'LocalNet'),
}


def _parse_range(spec: str | None):
    if spec is None:
        return None
    parts = spec.split(':')
    return (float(parts[0]), float(parts[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    parser.add_argument('--model', default='mlp_corrector')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--wd', type=float, default=None)
    parser.add_argument('--val_every', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--loss', default=None, choices=['mse', 'rel_mse', 'shock_weighted_mse', 'shock_weighted_rel_mse'])
    parser.add_argument('--schedule', default=None, choices=['cosine', 'constant'])
    parser.add_argument('--grad_clip', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--run_name', default=None)
    parser.add_argument('--no_save_best', action='store_true')
    parser.add_argument('--data_dir', default='data/diamond/')
    parser.add_argument('--stats', default='data/diamond/stats.npz')
    parser.add_argument('--mach', default=None, help='start:stop')
    parser.add_argument('--aoa', default=None, help='start:stop')
    args = parser.parse_args()

    cfg = load_cfg(args.config) if args.config else {}
    model_name = cfg.get('model', args.model)

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
        schedule = _get(args.schedule, 'schedule', 'cosine'),
        grad_clip = _get(args.grad_clip, 'grad_clip', 0.0),
        batch_size = _get(args.batch_size, 'batch_size', 1),
        save_best = (not args.no_save_best) and tr.get('save_best', True),
    )

    data_dir = Path(args.data_dir).resolve()
    lr_res = cfg.get('resolution', {}).get('lr', 0.1)
    knn = cls.load_knn(data_dir, cfg)
    mach_range = _parse_range(args.mach)
    aoa_range = _parse_range(args.aoa)

    use_lr_grad = cfg.get('architecture', {}).get('use_lr_grad', False)
    train_ds = SRDataset(data_dir, args.stats, 'train', lr_res=lr_res, use_lr_grad=use_lr_grad,
                    mach_range=mach_range, aoa_range=aoa_range)
    val_ds = SRDataset(data_dir, args.stats, 'val', lr_res=lr_res, use_lr_grad=use_lr_grad,
                    mach_range=mach_range, aoa_range=aoa_range)
    
    cfg.setdefault('architecture', {})['use_lr_grad'] = train_ds._has_lr_grad

    stats   = np.load(args.stats)
    res_cfg = cfg.get('resolution', {})
    mesh_hr = np.load(data_dir / f'mesh_{res_cfg.get("hr", 0.025)}.npy', allow_pickle=True).item()
    mesh_lr = np.load(data_dir / f'mesh_{res_cfg.get("lr", 0.1)}.npy', allow_pickle=True).item()

    from preprocessing.dataset import _res_tag
    test_dir = data_dir / 'processed' / f'lr{_res_tag(lr_res)}' / 'test'
    if not test_dir.exists():
        test_dir = data_dir / 'processed' / 'test'
    all_files = [str(f) for f in sorted(test_dir.glob('aoa*.npz'))]
    train_dir = test_dir.parent / 'train'
    if train_dir.exists():
        all_files += [str(f) for f in sorted(train_dir.glob('aoa*.npz'))]

    run_label = args.run_name or f"{model_name}_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}"
    pred_dir  = Path('results') / 'predictions' / run_label

    def _plot_ref_cases(m, knn_, dest):
        for mach_t, aoa_t, label in REFERENCE_CASES:
            path = find_ref_file(all_files, mach_t, aoa_t)
            if path is None:
                continue
            plot_prediction(m, knn_, path, stats, mesh_hr, dest, tag=f'_{label}',
                            mesh_lr=mesh_lr)

    def pred_callback(m, epoch, knn_):
        ep_dir = pred_dir / f'ep{epoch:04d}'
        _plot_ref_cases(m, knn_, ep_dir)

    model = cls(nnx.Rngs(train_cfg.seed), cfg)
    trained = model.fit(train_ds, val_ds, knn, train_cfg, out_dir='results/checkpoints',
        model_cfg=cfg, run_name=run_label, pred_callback=pred_callback)

    _plot_ref_cases(trained, knn, pred_dir / 'best')
    print(f"Prediction plots: {pred_dir}/")


if __name__ == '__main__':
    main()
