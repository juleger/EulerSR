"""
Module de base pour les réseaux de neurones de super-résolution "déterministe" (prédiction directe de HR à partir de LR)

SRModel : classe de base abstraite pour les modèles de super-résolution. Définit l'interface commune pour la prédiction, le chargement des kNN, l'entraînement et l'évaluation.
MLP : classe utilitaire pour un perceptron multi-couches simple, utilisé dans les modèles de super-résolution.

Hyperparamètres :
- architecture : dict contenant les détails de l'architecture du modèle (ex: dimensions des couches, utilisation des gradients LR, échelles de kNN, etc.)
- training : dict contenant les détails de l'entraînement (ex: nombre d'époques, taux d'apprentissage, poids de la loss physique, etc.)

Entraînement :
- fit() : méthode d'entraînement du modèle sur un dataset de train et de validation, avec évaluation périodique sur la validation, sauvegarde du meilleur modèle, et possibilité de callback pour visualiser les prédictions pendant l'entraînement.
- evaluate() : méthode d'évaluation du modèle sur un dataset donné, avec calcul de la perte et des erreurs relatives L2, et comparaison avec une baseline d'interpolation IDW.

"""

import datetime
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / 'euler') not in sys.path:
    sys.path.append(str(REPO_ROOT / 'euler'))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loss import rel_l2, LOSS_FNS
from utils.viz._style import VAR_LABELS
from utils.viz.training import plot_training_curves, plot_prediction  # noqa: F401


def eval_idw(ds, knn: dict, k: int = 6) -> np.ndarray:
    """
    Evaluation d'une baseline d'interpolation par IDW à partir des kNN : permet de comparer les modèles deep learning avec une approche d'interpolation classique (erreur rel L2 sur les prims)
    """
    k_use = min(k, np.asarray(knn['idx']).shape[1])
    idx_np  = np.asarray(knn['idx'])[:, :k_use].astype(np.int32)
    dist_np = np.asarray(knn['dist'])[:, :k_use].astype(np.float64)
    w = 1.0 / (dist_np ** 2 + 1e-12)
    w /= w.sum(axis=1, keepdims=True)
    w32 = w.astype(np.float32)

    l2_acc = np.zeros(4)
    for i in range(len(ds)):
        _, lr, tg, *_ = ds[i]
        pred = (w32[:, :, None] * lr[:, :4][idx_np]).sum(axis=1)
        l2_acc += np.array(rel_l2(jnp.array(pred), jnp.array(tg)))
    return l2_acc / len(ds)


@dataclass
class TrainConfig:
    epochs: int = 100
    lr: float = 2e-4
    weight_decay: float = 1e-5
    val_every: int = 10
    seed: int = 42
    loss: str = 'rel_mse'
    schedule: str = 'cosine'
    grad_clip: float = 0.0
    save_best: bool = True
    batch_size: int = 1
    lambda_phys: float = 0.0

class MLP(nnx.Module):
    def __init__(self, dims: list[int], rngs: nnx.Rngs):
        self.layers = [nnx.Linear(dims[i], dims[i + 1], rngs=rngs)
                       for i in range(len(dims) - 1)]

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers[:-1]:
            x = jax.nn.silu(layer(x))
        return self.layers[-1](x)

@nnx.jit
def _eval_step(model, hr_feat, lr_feat, knn, target, weights):
    pred = model.predict(hr_feat, lr_feat, knn)
    return rel_l2(pred, target)

class SRModel(nnx.Module):
    def predict(self, hr_feat: jax.Array, lr_feat: jax.Array, knn: dict) -> jax.Array:
        raise NotImplementedError

    @classmethod
    def load_knn(cls, data_dir: Path) -> dict: 
        raise NotImplementedError

    def save(self, path: str | Path, cfg: dict | None = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _, state = nnx.split(self)
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        if cfg is not None:
            import yaml
            with open(path.with_suffix('.yaml'), 'w') as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: str | Path) -> 'SRModel':
        path = Path(path)
        cfg = None
        cfg_path = path.with_suffix('.yaml')
        if cfg_path.exists():
            import yaml
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
        dummy = cls(nnx.Rngs(0), cfg)
        graphdef, _ = nnx.split(dummy)
        with open(path, 'rb') as f:
            state = pickle.load(f)
        return nnx.merge(graphdef, state)

    def evaluate(self, ds, knn: dict) -> dict:
        l2_acc = np.zeros(4)
        t0 = time.time()
        for i in range(len(ds)):
            hr, lr, tg, wt, *_ = ds[i]
            l2 = _eval_step(self,
                             jnp.array(hr), jnp.array(lr), knn,
                             jnp.array(tg), jnp.array(wt))
            l2_acc += np.array(l2)
        elapsed = time.time() - t0
        l2_acc /= len(ds)
        idw_l2 = eval_idw(ds, knn)
        results = {v: float(l2_acc[i]) for i, v in enumerate(VAR_LABELS)}
        results.update({f'idw_{v}': float(idw_l2[i]) for i, v in enumerate(VAR_LABELS)})
        results['_time_total_s'] = elapsed
        results['_time_per_sample_ms'] = elapsed / len(ds) * 1000
        return results

    def fit(self, train_ds, val_ds, knn: dict, cfg: TrainConfig, out_dir: str | Path = 'results/checkpoints',
            model_cfg: dict | None = None, run_name: str | None = None, pred_callback=None) -> 'SRModel':
        
        if run_name is None:
            # Utilisation du temps actuel pour nommer la run
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            run_name = f"{self.__class__.__name__.lower()}_{ts}"

        out_dir = Path(out_dir) / run_name
        out_dir.mkdir(parents=True, exist_ok=True)

        sep = '─' * 52
        print(f'\n{sep}')
        print(f'  Run : {run_name}')
        print(f'  Out : {out_dir}')
        if model_cfg:
            import yaml
            cfg_str = yaml.dump(model_cfg, default_flow_style=False,
                                sort_keys=False).rstrip()
            for line in cfg_str.splitlines():
                print(f'  {line}')
        print(f'  ·  n_train={len(train_ds)}  n_val={len(val_ds)}'
              f'  params={n_params(self):,}')
        print(sep)

        n_steps = cfg.epochs * len(train_ds)
        lr_sched = (optax.cosine_decay_schedule(cfg.lr, n_steps)
                    if cfg.schedule == 'cosine' else cfg.lr)
        
        # Optimiseur AdamW avec éventuellement du clipping de gradients
        tx = optax.adamw(lr_sched, weight_decay=cfg.weight_decay)
        if cfg.grad_clip > 0:
            tx = optax.chain(optax.clip_by_global_norm(cfg.grad_clip), tx)

        optimizer = nnx.Optimizer(self, tx) 
        compute_loss = LOSS_FNS[cfg.loss]
        batch_size = cfg.batch_size
        lambda_phys = cfg.lambda_phys

        @nnx.jit
        def step_single(opt, hr, lr, tg, wt, gp_tg):
            # Run du modèle sur un seul sample
            def lf(m):
                pred = m.predict(hr, lr, knn)
                loss = compute_loss(pred, tg, wt)
                if lambda_phys > 0 and hasattr(m, '_grad_p'):
                    loss = loss + lambda_phys * (
                        (jnp.arcsinh(m._grad_p(hr, lr, knn)) - gp_tg) ** 2).mean()
                return loss
            loss, grads = nnx.value_and_grad(lf)(opt.model)
            opt.update(grads)
            return loss

        @nnx.jit
        def step_batched(opt, hr_b, lr_b, tg_b, wt_b, valid_b, gp_b):
            # Run du modèle sur un batch de samples 
            def lf(m):
                def one(hr, lr, tg, wt, gp):
                    pred = m.predict(hr, lr, knn)
                    loss = compute_loss(pred, tg, wt)
                    if lambda_phys > 0 and hasattr(m, '_grad_p'):
                        loss = loss + lambda_phys * ((m._grad_p(hr, lr, knn) - gp) ** 2).mean()
                    return loss
                losses = jax.vmap(one)(hr_b, lr_b, tg_b, wt_b, gp_b)
                return (losses * valid_b).sum() / valid_b.sum()
            loss, grads = nnx.value_and_grad(lf)(opt.model)
            opt.update(grads)
            return loss

        @nnx.jit
        def eval_step_val(model, hr, lr, tg, wt):
            # Validation sur un sample
            pred = model.predict(hr, lr, knn)
            return compute_loss(pred, tg, wt), rel_l2(pred, tg)

        rng = np.random.default_rng(cfg.seed)
        train_losses, val_losses, val_l2s = [], [], []
        best_val, best_state = float('inf'), None
        t0 = time.time()

        phys_info = f"  lambda_phys={lambda_phys}" if lambda_phys > 0 else ""
        print(f"\nTraining {self.__class__.__name__}  "
              f"epochs={cfg.epochs}  lr={cfg.lr}  loss={cfg.loss}{phys_info}  "
              f"schedule={cfg.schedule}  grad_clip={cfg.grad_clip}  "
              f"batch={batch_size}")

        # Baseline IDW sur le jeu de validation avant entraînement
        idw_ref = eval_idw(val_ds, knn)
        print(f"  IDW baseline (val): "
              + "  ".join(f"{v}={idw_ref[i]:.3f}" for i, v in enumerate(VAR_LABELS)))

        def _delta(model_v, idw_v):
            pct = (model_v - idw_v) / (idw_v + 1e-9) * 100
            sign = '-' if pct <= 0 else '+'
            return f"{model_v:.3f}({sign}{abs(pct):.0f}%)"

        for epoch in range(1, cfg.epochs + 1):
            idx = rng.permutation(len(train_ds))
            ep_loss, n = 0.0, 0

            if batch_size > 1:
                for bi in range(0, len(idx), batch_size):
                    # Construction d'un batch avec les samples et features correspondantes
                    raw = [train_ds[int(i)] for i in idx[bi:bi + batch_size]]
                    B_real = len(raw)
                    pad = [raw[-1]] * (batch_size - B_real)
                    samples = raw + pad
                    valid = np.array([1.0] * B_real + [0.0] * (batch_size - B_real), dtype=np.float32)
                    hr_b = jnp.array(np.stack([s[0] for s in samples]))
                    lr_b = jnp.array(np.stack([s[1] for s in samples]))
                    tg_b = jnp.array(np.stack([s[2] for s in samples]))
                    wt_b = jnp.array(np.stack([s[3] for s in samples]))
                    gp_b = jnp.array(np.stack([s[4] for s in samples]))

                    # Perte sur l'ensemble du batch
                    loss = step_batched(optimizer, hr_b, lr_b, tg_b, wt_b, jnp.array(valid), gp_b)
                    ep_loss += float(loss) * B_real
                    n += B_real
            else:
                for i in idx:
                    hr, lr, tg, wt, gp = train_ds[int(i)]
                    loss = step_single(optimizer, jnp.array(hr), jnp.array(lr),
                        jnp.array(tg), jnp.array(wt), jnp.array(gp))
                    ep_loss += float(loss)
                    n += 1

            train_losses.append(ep_loss / max(n, 1))

            if epoch % cfg.val_every == 0 or epoch == cfg.epochs:
                # Validation sur l'ensemble du ds avec calcul de la loss et des erreurs relatives L2, sauvegarde du modèle/prédictions
                vl, vl2 = 0.0, np.zeros(4)
                for i in range(len(val_ds)):
                    hr, lr, tg, wt, *_ = val_ds[i]
                    l, l2 = eval_step_val(self, jnp.array(hr), jnp.array(lr), jnp.array(tg), jnp.array(wt))
                    vl += float(l); vl2 += np.array(l2)
                vl /= len(val_ds); vl2 /= len(val_ds)
                val_losses.append(vl)
                val_l2s.append(vl2.tolist())

                if cfg.save_best and vl < best_val:
                    best_val = vl
                    _, best_state = nnx.split(self)

                self.save(out_dir / 'model.pkl', cfg=model_cfg)

                if pred_callback is not None:
                    pred_callback(self, epoch, knn)

                print(f"  ep {epoch:4d}/{cfg.epochs}  "
                    f"tr={train_losses[-1]:.4f}  val={vl:.4f}  "
                    + "  ".join(f"{v}={_delta(vl2[i], idw_ref[i])}"
                                for i, v in enumerate(VAR_LABELS))
                    + f"  {time.time()-t0:.0f}s")

        # Si besoin, sauvegarde le meilleur modèle trouvé pendant l'entraînement (en fonction de la val)
        if cfg.save_best and best_state is not None:
            graphdef, _ = nnx.split(self)
            saved = nnx.merge(graphdef, best_state)
        else:
            saved = self

        total = time.time() - t0
        print(f"    Entraînement terminé  {total:.1f}s  "
              f"({total / cfg.epochs:.2f}s/epoch  "
              f"{total / (cfg.epochs * len(train_ds)) * 1000:.1f}ms/sample)")

        # Sauvegarde du modèle final et des courbes d'entraînement
        ckpt = out_dir / 'model.pkl'
        saved.save(ckpt, cfg=model_cfg)
        print(f"    Checkpoint: {ckpt}  (best_val={best_val:.4f})" if cfg.save_best
              else f"    Checkpoint: {ckpt}")

        plot_training_curves(train_losses, val_losses, val_l2s, out_dir / 'training_curves.png')

        saved._run_dir = out_dir
        return saved


def _res_tag(r: float) -> str:
    return 'h' + f'{r}'.replace('0.', '')


def _knn_path(data_dir: Path, hr_res: float, lr_res: float, k: int) -> Path:
    base = Path(data_dir) / 'knn'
    lr_tag, hr_tag = _res_tag(lr_res), _res_tag(hr_res)
    p = base / f'knn_{lr_tag}_to_{hr_tag}_k{k}.npz'
    if not p.exists() and k == 6:
        p = base / f'knn_{lr_tag}_to_{hr_tag}.npz'
    if not p.exists():
        raise FileNotFoundError(
            f"Fichier kNN non trouvé: {p}\n"
            f"Construction via:\n"
            f"  python preprocessing/preprocess.py "
            f"--data_dir {data_dir} "
            f"--knn_k {k}"
        )
    return p


def load_cfg(path: str | Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)

def n_params(model: nnx.Module) -> int:
    _, state = nnx.split(model)
    leaves = jax.tree_util.tree_leaves(state)
    total = 0
    for leaf in leaves:
        val = getattr(leaf, 'value', leaf)
        if hasattr(val, 'size'):
            total += int(val.size)
    return total
