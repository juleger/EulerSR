"""Entraînement FAMWall/DAMWall (reconstruction depuis observations de bord
uniquement), script SÉPARÉ de train.py (pipeline DAM/FAM/SIAM legacy), jamais
touché. Réutilise WallSRDataset (preprocessing/dataset.py) + TrainConfig/load_cfg
(models/base.py).

Mode multi-géométrie disponible via une clé 'datasets:' dans le YAML (une
entrée = une géométrie, cf. build_wall_branches) -- même mécanisme que
train.py (MultiSRDataset, model.fit() générique déjà multi-branche), adapté
au fait que la géométrie est la SEULE dimension de branche pertinente pour
FAMWall (pas de lr_res volumique, resolution.lr = proxy wd_exp).
Nécessite architecture.use_geom_cond=true dans le YAML (sinon geom_id est
ignoré par le modèle) et data/processed/{geometry}_wall/ prétraité pour
chaque géométrie (preprocessing/preprocess_wall.py).

Usage :
  python train_wall.py --config configs/fam_wall_sd.yaml
  python train_wall.py --config configs/fam_wall_2geo.yaml
"""
import sys
import argparse
import importlib
from datetime import datetime
from pathlib import Path

from flax import nnx

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / 'euler'))
import euler.jax_fvm.src.mesh  # noqa: F401

from preprocessing.dataset import WallSRDataset, MultiSRDataset
from models.base import TrainConfig, load_cfg
from utils.layout import DataLayout

_MODELS = {
    'dam_wall': ('models.fam_wall', 'DAMWall'),
    'fam_wall': ('models.fam_wall', 'FAMWall'),
}


def build_wall_branches(datasets_cfg, data_root, hr_res, lr_res,
                        mach_range, aoa_range, default_train_fraction, aoa_step=None):
    """Construit les branches du mode multi-géométrie FAMWall.

    Une branche = une géométrie (pas de sweep lr_res : lr_res n'est qu'un
    proxy géométrique fixe pour FAMWall, cf. resolution.lr). Équivalent
    simplifié de train.py:build_branches (une seule branche par géométrie,
    pas de couple géométrie x lr_res)."""
    branches = []
    for gid, ds_cfg in enumerate(datasets_cfg):
        geom = ds_cfg['name']
        mr = tuple(ds_cfg['mach_range']) if ds_cfg.get('mach_range') else mach_range
        ar = tuple(ds_cfg['aoa_range']) if ds_cfg.get('aoa_range') else aoa_range
        as_ = ds_cfg['aoa_step'] if ds_cfg.get('aoa_step') is not None else aoa_step
        branches.append({
            'name': geom, 'geom': geom, 'geom_id': gid,
            'layout': DataLayout.from_root(data_root, geom, lr_res, hr_res),
            'weight': float(ds_cfg.get('weight', 1.0)),
            'mach_range': mr, 'aoa_range': ar, 'aoa_step': as_,
            'train_fraction': float(ds_cfg.get('train_fraction', default_train_fraction)),
        })
    return branches


def main():
    parser = argparse.ArgumentParser(description="Entraînement FAMWall/DAMWall (bord uniquement)")
    parser.add_argument('--config', required=True)
    parser.add_argument('--data', default=None, help="Surcharge data_root du YAML")
    parser.add_argument('--geometry', default=None, help="Surcharge geometry du YAML")
    parser.add_argument('--model', default=None, help="Surcharge model du YAML (dam_wall/fam_wall)")
    parser.add_argument('--epochs', type=int, default=None, help="Surcharge training.epochs du YAML")
    parser.add_argument('--run_name', default=None)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    data_root = Path(args.data or cfg.get('data_root', 'data/')).resolve()
    geometry = args.geometry or cfg.get('geometry')  # optionnel si cfg['datasets'] (mode multi)
    model_name = args.model or cfg.get('model', 'fam_wall')
    module_name, cls_name = _MODELS[model_name]
    cls = getattr(importlib.import_module(module_name), cls_name)

    res = cfg.get('resolution', {})
    hr_res = res.get('hr', 0.025)
    # lr_res : proxy de longueur de décroissance wd_exp (cf. models/fam_wall.py
    # load_hierarchical_knn_wall)
    lr_res = res.get('lr', 0.1)

    # Convention legacy (cf. train.py) : shock_weight_factor/preload/train_fraction/
    train_cfg = cfg.get('training', {})
    preload = train_cfg.get('preload', True)
    sw_factor = train_cfg.get('shock_weight_factor', 1.0)
    default_train_fraction = train_cfg.get('train_fraction', 1.0)
    mach_range = tuple(train_cfg['mach_range']) if 'mach_range' in train_cfg else None
    aoa_range = tuple(train_cfg['aoa_range']) if 'aoa_range' in train_cfg else None
    aoa_step = train_cfg.get('aoa_step')
    wall_k_range = tuple(train_cfg['wall_k_range']) if 'wall_k_range' in train_cfg else None

    # (mid, scale) du conditionnement Mach pour CE run, meme convention que train.py.
    # Persiste dans cfg pour que _build_backbone calibre mach_mid/mach_scale des la
    # construction du modele.
    _mn_range = mach_range if mach_range is not None else (0.7, 3.0)
    mach_norm = ((_mn_range[0] + _mn_range[1]) / 2, (_mn_range[1] - _mn_range[0]) / 2)
    cfg['mach_norm'] = list(_mn_range)

    rngs = nnx.Rngs(train_cfg.get('seed', 42))

    if 'datasets' in cfg:
        branches = build_wall_branches(cfg['datasets'], data_root, hr_res, lr_res,
                                       mach_range, aoa_range, default_train_fraction, aoa_step)
        cfg.setdefault('architecture', {})['n_geoms'] = len(branches)
        if not cfg['architecture'].get('use_geom_cond', False):
            print("  ATTENTION : mode multi-géométrie sans architecture.use_geom_cond=true "
                  "-- geom_id sera ignoré par le modèle (branches non distinguées).")
        print(f"  Branches ({len(branches)}) :")
        for b in branches:
            print(f"    [{b['name']:<16}] geom_id={b['geom_id']}  "
                  f"weight={b['weight']}  train_fraction={b['train_fraction']}")

        train_list, val_list, names, weights, knns = [], [], [], [], {}
        for b in branches:
            common = dict(mach_range=b['mach_range'], aoa_range=b['aoa_range'],
                          aoa_step=b['aoa_step'], shock_weight_factor=sw_factor,
                          geom_id=b['geom_id'], coord_norm='object', mach_norm=mach_norm)
            train_list.append(WallSRDataset(b['layout'], split='train', preload=preload,
                              train_fraction=b['train_fraction'], **common))
            val_list.append(WallSRDataset(b['layout'], split='val', preload=preload, **common))
            names.append(b['name'])
            weights.append(b['weight'])
            knns[b['name']] = cls.load_knn(b['layout'], cfg)
        train_ds = MultiSRDataset(train_list, names, weights)
        val_ds = MultiSRDataset(val_list, names, weights)
        knn = knns
        layout = branches[0]['layout']  # ressources primaires (cf. train.py)
        geometry = 'multi(' + ','.join(names) + ')'
    else:
        layout = DataLayout.from_root(data_root, geometry, lr_res, hr_res)
        common = dict(mach_range=mach_range, aoa_range=aoa_range, aoa_step=aoa_step,
                      shock_weight_factor=sw_factor, geom_id=0, coord_norm='object',
                      mach_norm=mach_norm)
        train_ds = WallSRDataset(layout, split='train', preload=preload,
                                 train_fraction=default_train_fraction, **common)
        val_ds = WallSRDataset(layout, split='val', preload=preload, **common)
        knn = cls.load_knn(layout, cfg)

    if wall_k_range is not None:
        if not hasattr(train_ds, 'enable_wall_subsample'):
            raise ValueError("training.wall_k_range demande mais le dataset ne supporte pas "
                             "le sous-echantillonnage (WallSRDataset/MultiSRDataset attendu).")
        print(f"  Sous-echantillonnage N_wall active : train_ds tire un N_wall dans "
              f"{wall_k_range} a chaque epoque (val_ds reste a taille pleine).")
        train_ds.enable_wall_subsample(wall_k_range)

    model = cls(rngs, cfg)
    if hasattr(model, 'mach_mid'):
        # Garantie explicite, meme si cfg portait deja une autre valeur de mach_norm.
        import jax.numpy as _jnp
        model.mach_mid.value = _jnp.array(mach_norm[0], _jnp.float32)
        model.mach_scale.value = _jnp.array(mach_norm[1], _jnp.float32)

    valid_fields = TrainConfig.__dataclass_fields__
    tcfg = TrainConfig(**{k: v for k, v in train_cfg.items() if k in valid_fields})
    if args.epochs is not None:
        tcfg.epochs = args.epochs

    run_name = args.run_name or cfg.get('run_name') or f'{cls_name}_{geometry}'

    print(f"[{datetime.now():%H:%M:%S}] Lancement train_wall.py  config={args.config}  "
          f"model={model_name}  geometry={geometry}  run_name={run_name}")
    print(f"  n_train={len(train_ds)}  n_val={len(val_ds)}")

    model.fit(train_ds, val_ds, knn, tcfg, out_dir='results/checkpoints',
             model_cfg=cfg, run_name=run_name)


if __name__ == '__main__':
    main()
