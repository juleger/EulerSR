"""
FAM : Flow-based Attention Mesh (Model) : flow matching conditionnel en
espace complet pour la super-résolution CFD.

On apprend v(x_t, t | cond) qui transporte du bruit gaussien vers le
résidu r = HR - IDW(LR) (normalisé par canal) :
    x_t = (1-t)*x0 + t*r,  v* = r - x0,  x0 ~ N(0, I)
L'échantillonnage intègre l'ODE dx/dt = v par schéma de Heun (n_steps pas).
La prédiction finale est baseline IDW + r_hat * res_scale. Plus rapide et
plus stable que la diffusion pour ce problème (trajectoire résiduelle courte).

Backbone U multi-échelles sur hiérarchie de maillages (identique à DAM, avec
temps t + intégration ODE) : self-attention locale HR -> cross-attention
descendante -> self-attention globale au bottleneck -> cross-attention
remontante -> décodeur linéaire (init à zéro). Conditionnement FiLM
(t, Mach, AoA) à chaque bloc ; champ LR IDW (+ gradients) injecté à chaque niveau.

res_scale (échelle du résiduel par canal) : par défaut calibrée sur la vérité
terrain HR (buffer/table par branche, cf. _pre_fit / recalibrate_res_scale) --
indisponible en déploiement réel sur géométrie/résolution inconnue. Option
flow.learned_res_scale=True (opt-in, comportement inchangé sinon) : une petite
tête prédit res_scale à partir du seul conditionnement (Mach, AoA, LR,
géométrie), entraînée par une loss auxiliaire séparée (flow.lambda_res_scale),
sans dépendance à la vérité terrain à l'inférence. Cf. _predicted_res_scale.
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
from models.base import SRModel, Buffer, MLP
from utils.layout import DataLayout
from models.dam import (AMNet, build_context, load_hierarchical_knn, _LR_REF,
                        cfg_drop_geom_id, _N_SCALARS, _N_COND)
from utils.attention import scalar_fourier_embedding
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
        self.cfg_scale = float(flow.get('cfg_scale', 1.0))  # guidance scale Mach/AoA (inference)
        # Guidance CFG sur la geometrie (inference) : null geom_id -> token n_geoms,
        # meme mecanisme que cfg_scale mais sur le conditionnement geometrique.
        # 1.0 = desactive (comportement inchange). Necessite geom_cfg_prob > 0 a
        # l'entrainement pour que le token nul soit reellement entraine.
        self.geom_cfg_scale = float(flow.get('geom_cfg_scale', 1.0))
        # Dropout CFG du conditionnement géométrique (entraînement uniquement)
        self.geom_cfg_prob = float(arch.get('geom_cfg_prob', 0.0))
        self._n_geoms = arch.get('n_geoms', 8)
        # Distribution d'échantillonnage de t à l'entraînement (flow matching)
        self._t_dist = flow.get('t_dist', 'uniform')
        self._sigma_t = float(flow.get('sigma_t', 1.0))
        self._t_alpha = float(flow.get('t_alpha', 2.0))
        self._t_beta = float(flow.get('t_beta', 2.0))
        # Validation sous-échantillonnée : sampling FAM = intégration ODE (coûteux)
        self.val_max_samples = self.n_val_gen

        # Échelle par canal du résiduel : table figée calibrée sur la vérité terrain
        # (std du résiduel HR-IDW par branche, cf. _pre_fit / recalibrate_res_scale).
        # Fallback historique, toujours actif si learned_res_scale=False.
        self.res_scale = Buffer(jnp.ones((4,), jnp.float32))
        self._nominal_lr = float((cfg.get('resolution') or {}).get('lr', _LR_REF))
        # Stats d'entraînement
        self.mu_train = Buffer(jnp.zeros((4,), jnp.float32))
        self.sig_train = Buffer(jnp.ones((4,), jnp.float32))

        # res_scale appris (opt-in, comportement historique inchangé si False) : au lieu
        # d'un lookup calibré sur du HR, une petite tête prédit res_scale à partir du seul
        # conditionnement (Mach, AoA, LR, géométrie), entraînée par une loss auxiliaire
        self._learned_res_scale = bool(flow.get('learned_res_scale', False))
        self.lambda_res_scale = float(flow.get('lambda_res_scale', 0.1))
        # false reproduit le comportement d'avant le fix du token nul (rs_pred = rs_raw,
        # jamais le geom_id droppe), reserve a l'ablation A/B.
        self._res_scale_fix = bool(flow.get('res_scale_fix', True))
        if self._learned_res_scale:
            # Tous les nouveaux attributs (y compris log_scale_bias) doivent être initialisés à zéro
            # pour que la prédiction initiale de res_scale soit identique au buffer calibré sur la vérité terrain (res_scale.value).
            scale_hidden = arch.get('scale_head_dim', 64)
            self.scale_mlp = MLP([_N_COND, scale_hidden, 4], rngs, dtype=compute_dtype)
            self.scale_mlp.layers[-1].kernel.value = jnp.zeros_like(self.scale_mlp.layers[-1].kernel.value)
            self.scale_mlp.layers[-1].bias.value = jnp.zeros_like(self.scale_mlp.layers[-1].bias.value)
            if self._use_geom_cond:
                self.scale_geom_emb = nnx.Embed(self._n_geoms + 1, 4, rngs=rngs)
                self.scale_geom_emb.embedding.value = jnp.zeros_like(self.scale_geom_emb.embedding.value)
            self.log_scale_bias = Buffer(jnp.zeros((4,), jnp.float32))

    def velocity(self, x_t: jax.Array, t: jax.Array, ctx: dict, knn: dict,
                 scal: jax.Array | None = None, geom_id: jax.Array | None = None) -> jax.Array:
        s = ctx['scal'] if scal is None else scal
        g = ctx['geom_id'] if geom_id is None else geom_id
        return self.net(ctx['c'], s, knn['fm'], ctx['wall'], g, x_t=x_t, t=t)

    def _predicted_res_scale(self, ctx: dict) -> jax.Array:
        """res_scale prédite (learned_res_scale=True) à partir du conditionnement
        seul (Mach, AoA, résolution LR, géométrie)"""
        scal = ctx['scal']
        feat = jnp.concatenate([scalar_fourier_embedding(scal[:_N_SCALARS]), scal[_N_SCALARS:]])
        log_scale = self.log_scale_bias.value + self.scale_mlp(feat)
        if self._use_geom_cond:
            log_scale = log_scale + self.scale_geom_emb(ctx['geom_id'])
        return jnp.exp(log_scale)

    def _res_scale(self, knn: dict, ctx: dict | None = None) -> jax.Array:
        """Échelle du résiduel pour cette branche de résolution LR (pour l'inférence)."""
        if self._learned_res_scale and ctx is not None:
            return self._predicted_res_scale(ctx)
        rs = knn.get('res_scale')
        return self.res_scale.value if rs is None else rs

    def _project(self, field: jax.Array) -> jax.Array:
        """Projection dure de positivité (rho, p >= eps) en unités physiques."""
        prim = field * self.sig_train.value + self.mu_train.value
        prim = prim.at[:, 0].set(jnp.maximum(prim[:, 0], 1e-3))
        prim = prim.at[:, 3].set(jnp.maximum(prim[:, 3], 1e-3))
        return (prim - self.mu_train.value) / self.sig_train.value

    def _guided_velocity(self, x_t: jax.Array, t: jax.Array, ctx: dict, knn: dict) -> jax.Array:
        if self.cfg_scale == 1.0 and self.geom_cfg_scale == 1.0:
            return self.velocity(x_t, t, ctx, knn)
        v_cond = self.velocity(x_t, t, ctx, knn)
        v = v_cond
        if self.cfg_scale != 1.0:
            scal_null = jnp.zeros_like(ctx['scal'])
            v_uncond = self.velocity(x_t, t, ctx, knn, scal=scal_null)
            v = v + (self.cfg_scale - 1.0) * (v_cond - v_uncond)
        if self.geom_cfg_scale != 1.0:
            geom_null = jnp.full_like(ctx['geom_id'], self._n_geoms)
            v_uncond_geom = self.velocity(x_t, t, ctx, knn, geom_id=geom_null)
            v = v + (self.geom_cfg_scale - 1.0) * (v_cond - v_uncond_geom)
        return v

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
        rs = jax.lax.stop_gradient(self._res_scale(knn, ctx))
        n_s = n_samples if n_samples is not None else self.n_samples
        key = key if key is not None else jax.random.PRNGKey(self.sample_seed)
        keys = jax.random.split(key, n_s)

        def one(k):
            x0 = jax.random.normal(k, (hr_feat.shape[0], 4))
            r = self._integrate(x0, ctx, knn)
            field = ctx['baseline'] + r * rs if self.use_residual else r * rs
            return self._project(field)

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
        scale = jnp.array(res_std, jnp.float32)
        self.res_scale.value = scale
        knn['res_scale'] = scale  # cohérence buffer / knn pour predict()
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

    def _branch_res_std(self, ds_b, knn_b) -> np.ndarray:
        """Std par canal du résiduel HR - IDW(LR)"""
        w0 = np.asarray(knn_b['fm']['cond'][0]['w'])
        i0 = np.asarray(knn_b['fm']['cond'][0]['idx'])
        n = len(ds_b)
        sub = np.unique(np.linspace(0, n - 1, min(256, n)).astype(int))
        acc = []
        for i in sub:
            _, lr, tg, *_ = ds_b[int(i)]
            lr = np.asarray(lr)
            tg = np.asarray(tg)
            if self.use_residual:
                baseline = (w0[:, :, None] * lr[:, :4][i0]).sum(axis=1)
                acc.append(tg - baseline)
            else:
                acc.append(tg)
        return np.maximum(np.concatenate(acc, 0).std(0), 1e-6)

    @staticmethod
    def _branch_lr(knn_b) -> float:
        """Résolution LR d'une branche, retrouvée depuis res_scalar = log(lr/_LR_REF)."""
        rsc = knn_b.get('res_scalar')
        if rsc is None:
            return _LR_REF
        return float(_LR_REF * np.exp(float(np.asarray(rsc).reshape(-1)[0])))

    def _pre_fit(self, branch_pairs: list, cfg) -> None:
        """Calibre res_scale par branche (résolution/géometire)"""
        _label = "std résiduel" if self.use_residual else "std cible"
        multi = len(branch_pairs) > 1
        best_i, best_gap = 0, float('inf')
        for bi, (ds_b, knn_b) in enumerate(branch_pairs):
            res_std = self._branch_res_std(ds_b, knn_b)
            knn_b['res_scale'] = jnp.array(res_std, jnp.float32)
            lr_b = self._branch_lr(knn_b)
            gap = abs(lr_b - self._nominal_lr)
            if gap < best_gap:
                best_gap, best_i = gap, bi
            tag = f" [lr={lr_b:g}]" if multi else ""
            print(f"  res_scale{tag} ({_label}) : "
                  + "  ".join(f"{v}={res_std[i]:.4f}" for i, v in enumerate(VAR_LABELS)))
        self.res_scale.value = jnp.array(branch_pairs[best_i][1]['res_scale'])
        if multi:
            print(f"  buffer res_scale : branche lr={self._branch_lr(branch_pairs[best_i][1]):g}"
                  f"  (résolution nominale {self._nominal_lr:g})")
        if self._learned_res_scale:
            # Point de départ de la tête apprise = buffer historique 
            self.log_scale_bias.value = jnp.log(jnp.maximum(self.res_scale.value, 1e-6))
            print(f"  res_scale APPRISE (learned_res_scale=True, lambda={self.lambda_res_scale:g}) "
                  "-- tête initialisée sur le buffer ci-dessus, affinée pendant l'entraînement.")

    def _sample_loss(self, hr: jax.Array, lr: jax.Array, tg: jax.Array, wt: jax.Array,
                gp: jax.Array, knn_g: dict, key: jax.Array, aux: dict) -> jax.Array:
        """Loss flow matching : v(x_t, t | cond) doit transporter x0 -> résidu r."""
        ctx = build_context(self, hr, lr, knn_g)
        # rs_raw garde le gradient (utilisé uniquement par la loss auxiliaire ci-dessous) ;
        # rs (stop_gradient) normalise le pont de flow matching.
        rs_raw = self._res_scale(knn_g, ctx)
        rs = jax.lax.stop_gradient(rs_raw)
        r = ((tg - ctx['baseline']) / rs if self.use_residual else tg / rs)
        kt, kx, kcfg, kgeom = jax.random.split(key, 4)
        t = self._sample_t(kt)
        x0 = jax.random.normal(kx, r.shape)
        x_t = (1.0 - t) * x0 + t * r
        # Dropout de conditionnement (classifier-free guidance), branche compilée une fois
        if self.cfg_prob > 0:
            drop = jax.random.bernoulli(kcfg, self.cfg_prob)
            scal_used = jnp.where(drop, jnp.zeros_like(ctx['scal']), ctx['scal'])
        else:
            scal_used = ctx['scal']
        # Dropout CFG du geom_id (conditionnement géométrique), indépendant du précédent
        if self.geom_cfg_prob > 0:
            ctx = dict(ctx)
            ctx['geom_id'] = cfg_drop_geom_id(ctx['geom_id'], self._n_geoms,
                                              self.geom_cfg_prob, kgeom)
        v = self.velocity(x_t, t, ctx, knn_g, scal=scal_used)
        loss = aux['loss_fn'](v, r - x0, wt)
        if self._learned_res_scale and self.lambda_res_scale > 0:
            # Loss auxiliaire : régresse la prédiction de res_scale vers le std réel du
            # résidu de CE cas, cible calculée sur le vrai geom_id. rs_pred reprend le ctx
            # d'APRÈS le dropout CFG, sans quoi la ligne du token nul de scale_geom_emb
            # ne recevrait jamais de gradient alors que c'est elle qui sert en OOD.
            rs_pred = (self._predicted_res_scale(ctx)
                       if self.geom_cfg_prob > 0 and self._res_scale_fix else rs_raw)
            true_std = jnp.maximum((r * rs).std(axis=0), 1e-6)
            loss = loss + self.lambda_res_scale * jnp.mean(
                (jnp.log(rs_pred) - jnp.log(true_std)) ** 2)
        # r_hat = x_t + (1-t)*v : extrapolation Euler un pas vers t=1 (endpoint prédit).
        # Si v == v* = r - x0 exactement, r_hat == r quel que soit t -- sert de base à la
        # fois au soft-loss physique et à la loss d'endpoint direct ci-dessous.
        if aux['lambda_phys'] > 0 or aux['lambda_enthalpy'] > 0 or aux.get('use_endpoint', False):
            r_hat = x_t + (1.0 - t) * v
            # Loss d'endpoint : supervise directement r_hat contre la vraie cible r, à
            # n'importe quel t échantillonné (contrairement à la loss de vitesse qui ne
            # contraint que la direction locale).
            if aux.get('use_endpoint', False):
                loss = loss + aux['lambda_endpoint'] * aux['loss_fn'](r_hat, r, wt)
            if aux['lambda_phys'] > 0 or aux['lambda_enthalpy'] > 0:
                field = (ctx['baseline'] + r_hat * rs if self.use_residual
                         else r_hat * rs)
                loss = loss + self._phys_terms(field, hr, gp, knn_g, aux)
        return loss
