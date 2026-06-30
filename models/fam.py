"""
FAM : Flow-based Attention Mesh (Model) :
Flow matching conditionnel en espace complet pour la super-résolution CFD.

Au lieu de prédire directement HR (déterministe) ou de diffuser en latent,
on apprend un champ de vitesse v(x_t, t | cond) qui transporte du bruit
gaussien vers le résidu r = HR - IDW(LR) (normalisé par canal).
    x_t = (1-t)*x0 + t*r,  v* = r - x0,  x0 ~ N(0, I)
L'échantillonnage intègre l'ODE dx/dt = v par schéma de Heun (n_steps pas).
La prédiction finale est baseline IDW + r_hat * res_scale.

Avantage sur la diffusion pour ce genre de problèmes :
- Moins de pas d'intégration (typiquement 1000 pour DDPM a 50/100 pour DDIM) : inférence plus rapide
- L'approche résiduelle rend la trajectoire très courte et plus stable
- Plus naturel pour conditionner la super-résolution

La diffusion modélise la distribution en prédisant le bruit ajouté, ou bien le score
Sans conditionnement, elle est efficace pour générer des images réalistes, mais pour la super-résolution CFD,
Le flow apprend directement les trajectoires de transport vers la distribution cible, plus efficace pour la SR conditionnelle

Backbone U multi-échelles sur hiérarchie de maillages :
- niveau 0 (HR)       : encodage par noeud + self-attention locale kNN
- descente l -> l+1   : cross-attention locale (requêtes = noeuds grossiers)
- bottleneck          : self-attention complète (champ réceptif global)
- remontée l+1 -> l   : cross-attention locale résiduelle + skip
- décodeur            : LayerNorm + Linear -> vitesse (4 canaux), init à zéro
Conditionnement : FiLM (t, Mach, AoA) dans chaque bloc ; champ LR interpolé
IDW (+ gradients) injecté à chaque niveau
Même architecture que DAM, mais avec variable de temps t et intégration ODE pour l'échantillonnage.
"""

import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from models.base import SRModel, Buffer
from utils.layout import DataLayout
from models.dam import AMNet, build_context, load_hierarchical_knn
from utils.viz._style import VAR_LABELS


class FAM(SRModel):
    """Flow matching conditionnel multi-échelles pour la super-résolution CFD :
    v(x_t, t | cond) sur hiérarchie de maillages, intégration ODE par Heun.
    """

    # Multi echelles de maillages pour le réserau
    DEFAULT_LEVELS = [0.05, 0.1]

    def __init__(self, rngs: nnx.Rngs, cfg: dict | None = None):
        cfg = cfg or {}
        arch = cfg.get('architecture', {})
        flow = cfg.get('flow', {})

        self.levels = list(arch.get('levels', self.DEFAULT_LEVELS))
        self.use_lr_grad = arch.get('use_lr_grad', True)
        self.n_c = 4 + (3 if self.use_lr_grad else 0)

        use_bf16 = arch.get('use_bf16', False)
        compute_dtype = jnp.bfloat16 if use_bf16 else None

        self._use_geom_cond = arch.get('use_geom_cond', False)
        self.net = AMNet(
            n_levels=len(self.levels),
            n_c=self.n_c,
            d=arch.get('embed_dim', 96),
            d_cond=arch.get('cond_dim', 128),
            n_heads=arch.get('n_heads', 4),
            n_global=arch.get('n_global_blocks', 3),
            mlp_ratio=arch.get('mlp_ratio', 2),
            rngs=rngs,
            d_t=arch.get('t_dim', 64), n_x=4,  # flow matching : temps + champ bruité x_t
            use_geom_cond=self._use_geom_cond,
            n_geoms=arch.get('n_geoms', 8),
            n_hr_blocks=arch.get('n_hr_blocks', 1),
            dtype=compute_dtype)

        self.n_steps = flow.get('n_steps', 16)
        self.use_residual = flow.get('use_residual', True)
        self.n_samples = flow.get('n_samples', 1)
        self.n_val_gen = flow.get('n_val_gen', 16)
        self.sample_seed = flow.get('sample_seed', 0)
        self.cfg_prob = float(flow.get('cfg_prob', 0.0))  # prob. dropout cond. (training)
        self.cfg_scale = float(flow.get('cfg_scale', 1.0))  # guidance scale (inference)
        # Distribution d'échantillonnage de t à l'entraînement (flow matching)
        self._t_dist = flow.get('t_dist', 'uniform')
        self._sigma_t = float(flow.get('sigma_t', 1.0))
        self._t_alpha = float(flow.get('t_alpha', 2.0))
        self._t_beta = float(flow.get('t_beta', 2.0))
        # Validation sous-échantillonnée : sampling FAM = intégration ODE (coûteux)
        self.val_max_samples = self.n_val_gen

        # Échelle par canal du résiduel
        self.res_scale = Buffer(jnp.ones((4,), jnp.float32))
        # Stats d'entraînement 
        self.mu_train = Buffer(jnp.zeros((4,), jnp.float32))
        self.sig_train = Buffer(jnp.ones((4,), jnp.float32))

    def velocity(self, x_t: jax.Array, t: jax.Array, ctx: dict, knn: dict,
                 scal: jax.Array | None = None) -> jax.Array:
        s = ctx['scal'] if scal is None else scal
        return self.net(ctx['c'], s, knn['fm'], ctx['wall'], ctx['geom_id'],
                        x_t=x_t, t=t)

    def _guided_velocity(self, x_t: jax.Array, t: jax.Array, ctx: dict, knn: dict) -> jax.Array:
        v_cond = self.velocity(x_t, t, ctx, knn)
        if self.cfg_scale == 1.0:
            return v_cond
        scal_null = jnp.zeros_like(ctx['scal'])
        v_uncond = self.velocity(x_t, t, ctx, knn, scal=scal_null)
        return v_uncond + self.cfg_scale * (v_cond - v_uncond)

    def _integrate(self, x0: jax.Array, ctx: dict, knn: dict) -> jax.Array:
        dt = 1.0 / self.n_steps

        def body(k, x):
            t1 = k * dt
            v1 = self._guided_velocity(x, t1, ctx, knn)
            v2 = self._guided_velocity(x + dt * v1, t1 + dt, ctx, knn)
            return x + 0.5 * dt * (v1 + v2)

        return jax.lax.fori_loop(0, self.n_steps, body, x0)

    def sample(self, hr_feat: jax.Array, lr_feat: jax.Array, knn: dict,
               key: jax.Array | None = None, n_samples: int | None = None) -> jax.Array:
        """Tire n_samples champs HR normalisés [S, N, 4] (ensemble / UQ)."""
        ctx = build_context(self, hr_feat, lr_feat, knn)
        n_s = n_samples if n_samples is not None else self.n_samples
        key = key if key is not None else jax.random.PRNGKey(self.sample_seed)
        keys = jax.random.split(key, n_s)

        def one(k):
            x0 = jax.random.normal(k, (hr_feat.shape[0], 4))
            r = self._integrate(x0, ctx, knn)
            if self.use_residual:
                return ctx['baseline'] + r * self.res_scale.value
            else:
                return r * self.res_scale.value

        return jax.vmap(one)(keys)

    def predict(self, hr_feat: jax.Array, lr_feat: jax.Array, knn: dict) -> jax.Array:
        # Moyenne des n samples pour la préd finale
        return self.sample(hr_feat, lr_feat, knn).mean(axis=0)

    def recalibrate_res_scale(self, lr_feats: list, tg_list: list, knn: dict) -> None:
        """Recalibre res_scale pour une résolution LR différente de l'entraînement.

        lr_feats : liste de tableaux (N_LR, >=4),primitives LR normalisées
        tg_list  : liste de tableaux (N_HR, 4), primitives HR normalisées
        knn      : dict kNN construit avec la nouvelle résolution LR
        """
        w0 = np.asarray(knn['fm']['cond'][0]['w'])
        i0 = np.asarray(knn['fm']['cond'][0]['idx'])
        acc = []
        for lr, tg in zip(lr_feats, tg_list):
            if self.use_residual:
                baseline = (w0[:, :, None] * np.asarray(lr)[:, :4][i0]).sum(axis=1)
                acc.append(np.asarray(tg) - baseline)
            else:
                acc.append(np.asarray(tg))
        res_std = np.maximum(np.concatenate(acc, axis=0).std(0), 1e-6)
        self.res_scale.value = jnp.array(res_std, jnp.float32)
        print('  res_scale recalibrée : '
              + '  '.join(f'{v}={res_std[i]:.4f}' for i, v in enumerate(VAR_LABELS)))

    @classmethod
    def load_knn(cls, layout: DataLayout, cfg: dict | None = None) -> dict:
        return load_hierarchical_knn(layout, cls.DEFAULT_LEVELS, cfg, tag='FM')

    def _sample_t(self, key: jax.Array) -> jax.Array:
        """Tire t ~ distribution configurée (uniform / logit_normal / beta)."""
        if self._t_dist == 'logit_normal':
            return jax.nn.sigmoid(jax.random.normal(key) * self._sigma_t)
        if self._t_dist == 'beta':
            return jax.random.beta(key, self._t_alpha, self._t_beta)
        return jax.random.uniform(key)

    def _pre_fit(self, branch_pairs: list, cfg) -> None:
        """Calibre res_scale (std du résiduel) sur TOUTES les branches"""
        acc = []
        for ds_b, knn_b in branch_pairs:
            w0 = np.asarray(knn_b['fm']['cond'][0]['w'])
            i0 = np.asarray(knn_b['fm']['cond'][0]['idx'])
            sub = np.unique(np.linspace(0, len(ds_b) - 1,
                                        min(256, len(ds_b))).astype(int))
            for i in sub:
                _, lr, tg, *_ = ds_b[int(i)]
                if self.use_residual:
                    baseline = (w0[:, :, None] * lr[:, :4][i0]).sum(axis=1)
                    acc.append(tg - baseline)
                else:
                    acc.append(tg)
        res_std = np.maximum(np.concatenate(acc, 0).std(0), 1e-6)
        self.res_scale.value = jnp.array(res_std, jnp.float32)
        _label = "std résiduel" if self.use_residual else "std cible"
        print(f"  res_scale ({_label}) : "
              + "  ".join(f"{v}={res_std[i]:.4f}" for i, v in enumerate(VAR_LABELS)))

    def _sample_loss(self, hr: jax.Array, lr: jax.Array, tg: jax.Array, wt: jax.Array,
                gp: jax.Array, knn_g: dict, key: jax.Array, aux: dict) -> jax.Array:
        """Loss flow matching : v(x_t, t | cond) doit transporter x0 -> résidu r."""
        ctx = build_context(self, hr, lr, knn_g)
        r = ((tg - ctx['baseline']) / self.res_scale.value if self.use_residual
             else tg / self.res_scale.value)
        kt, kx, kcfg = jax.random.split(key, 3)
        t = self._sample_t(kt)
        x0 = jax.random.normal(kx, r.shape)
        x_t = (1.0 - t) * x0 + t * r
        # Dropout de conditionnement (classifier-free guidance), branche compilée une fois
        if self.cfg_prob > 0:
            drop = jax.random.bernoulli(kcfg, self.cfg_prob)
            scal_used = jnp.where(drop, jnp.zeros_like(ctx['scal']), ctx['scal'])
        else:
            scal_used = ctx['scal']
        v = self.velocity(x_t, t, ctx, knn_g, scal=scal_used)
        loss = aux['loss_fn'](v, r - x0, wt)
        # Soft-loss physique sur l'endpoint prédit en un pas : r_hat = x_t + (1-t)*v
        if aux['lambda_phys'] > 0 or aux['lambda_enthalpy'] > 0:
            r_hat = x_t + (1.0 - t) * v
            field = (ctx['baseline'] + r_hat * self.res_scale.value if self.use_residual
                     else r_hat * self.res_scale.value)
            loss = loss + self._phys_terms(field, hr, gp, knn_g, aux)
        return loss
