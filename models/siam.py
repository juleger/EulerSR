"""
SIAM : Stochastic Interpolant Attention Mesh (Model)
Interpolant stochastique conditionnel pour la super-resolution CFD.

Au lieu de transporter du bruit gaussien vers le residu (FAM), on construit un
pont direct entre le champ basse resolution et le champ haute resolution :
    x0 = baseline IDW(LR)        (champ deja physique)
    x1 = champ HR cible
    x_t = (1-t)*x0 + t*x1 + sigma*g(t)*z ,   z ~ N(0, I) ,  g(t) = sin(pi t)
Le reseau apprend la derive du pont b(x_t, t | cond) = E[ d/dt x_t | x_t ], et
l'echantillonnage integre l'ODE dx/dt = b de x0 (baseline) a x1 (HR).

Avantages sur le transport bruit -> residu :
- On part d'un champ physique, jamais du bruit pur : trajectoire plus courte,
  plus stable, pas de champ aberrant aux petits t.
- sigma_bridge = 0 : pont deterministe (flow matching baseline -> HR).
  sigma_bridge > 0 : interpolant stochastique -> le bruit z du pont faconne la
  derive apprise b = E[d/dt x_t | x_t]. L'inference reste la PF-ODE
  deterministe dx/dt = b (preserve les marginales), donc predict() est
  deterministe : un HR par (geometrie, Mach, AoA).
- Positivite garantie par une projection dure (rho, p >= eps) en fin
  d'integration ; conservation optionnelle via lambda_enthalpy a l'entrainement.

Backbone : AMNet, identique a DAM/FAM (temps t + champ courant x_t en entree).
Le conditionnement (champ LR interpole, Mach/AoA, geometrie, paroi, resolution)
est le meme que FAM via build_context.
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


class SIAM(SRModel):
    """Pont interpolant stochastique multi-echelles pour la super-resolution CFD :
    b(x_t, t | cond) sur hierarchie de maillages, integration de x0=baseline a x1=HR.
    """

    # Multi echelles de maillages pour le reseau (meme hierarchie que FAM)
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
            d_t=arch.get('t_dim', 64), n_x=4,  # pont : temps + champ courant x_t
            use_geom_cond=self._use_geom_cond,
            n_geoms=arch.get('n_geoms', 8),
            n_hr_blocks=arch.get('n_hr_blocks', 1),
            dtype=compute_dtype)

        # Interpolant : bruit sigma*g(t), g(t) = sin(pi t)
        self.sigma_bridge = float(flow.get('sigma_bridge', 0.0))
        # Distribution d'echantillonnage de t a l'entrainement
        self._t_dist = flow.get('t_dist', 'uniform')
        self._sigma_t = float(flow.get('sigma_t', 1.0))
        self._t_alpha = float(flow.get('t_alpha', 2.0))
        self._t_beta = float(flow.get('t_beta', 2.0))

        # Echantillonnage
        self.n_steps = flow.get('n_steps', 16)
        self.n_samples = flow.get('n_samples', 1)
        self.n_val_gen = flow.get('n_val_gen', 16)
        self.sample_seed = flow.get('sample_seed', 0)
        self.val_max_samples = self.n_val_gen

        # Stats d'entrainement (denormalisation pour la projection de positivite)
        self.mu_train = Buffer(jnp.zeros((4,), jnp.float32))
        self.sig_train = Buffer(jnp.ones((4,), jnp.float32))

    def velocity(self, x_t: jax.Array, t: jax.Array, ctx: dict, knn: dict) -> jax.Array:
        return self.net(ctx['c'], ctx['scal'], knn['fm'], ctx['wall'], ctx['geom_id'], x_t=x_t, t=t)

    @staticmethod
    def _noise_env(t: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Envelope de bruit du pont g(t) = sin(pi t) et sa derivee g'(t)."""
        return jnp.sin(jnp.pi * t), jnp.pi * jnp.cos(jnp.pi * t)

    def _sample_t(self, key: jax.Array) -> jax.Array:
        """Tire t ~ distribution configuree (uniform / logit_normal / beta)."""
        if self._t_dist == 'logit_normal':
            return jax.nn.sigmoid(jax.random.normal(key) * self._sigma_t)
        if self._t_dist == 'beta':
            return jax.random.beta(key, self._t_alpha, self._t_beta)
        return jax.random.uniform(key)
    
    def _project(self, field: jax.Array) -> jax.Array:
        """Projection dure de positivite (rho, p >= eps) en unites physiques."""
        prim = field * self.sig_train.value + self.mu_train.value
        prim = prim.at[:, 0].set(jnp.maximum(prim[:, 0], 1e-3))
        prim = prim.at[:, 3].set(jnp.maximum(prim[:, 3], 1e-3))
        return (prim - self.mu_train.value) / self.sig_train.value

    def _integrate(self, x0: jax.Array, ctx: dict, knn: dict) -> jax.Array:
        """Integre le pont baseline (t=0) -> HR (t=1) par la PF-ODE (Heun).
        Positivite imposee en sortie."""
        dt = 1.0 / self.n_steps

        def body(k, x):
            t1 = k * dt
            v1 = self.velocity(x, t1, ctx, knn)
            v2 = self.velocity(x + dt * v1, t1 + dt, ctx, knn)
            return x + 0.5 * dt * (v1 + v2)

        x1 = jax.lax.fori_loop(0, self.n_steps, body, x0)
        return self._project(x1)

    def sample(self, hr_feat: jax.Array, lr_feat: jax.Array, knn: dict,
               key: jax.Array | None = None, n_samples: int | None = None) -> jax.Array:
        """Integre le pont deterministe baseline -> HR (PF-ODE). Renvoie [S, N, 4].

        Le pont etant deterministe, les S echantillons sont identiques ; l'axe S
        est conserve pour la compatibilite d'interface (ensemble). `key` est ignore.
        """
        ctx = build_context(self, hr_feat, lr_feat, knn)
        n_s = n_samples if n_samples is not None else self.n_samples
        field = self._integrate(ctx['baseline'], ctx, knn)
        return jnp.broadcast_to(field, (n_s,) + field.shape)

    def predict(self, hr_feat: jax.Array, lr_feat: jax.Array, knn: dict) -> jax.Array:
        # Moyenne des n samples pour la prediction finale (champ HR normalise)
        return self.sample(hr_feat, lr_feat, knn).mean(axis=0)

    @classmethod
    def load_knn(cls, layout: DataLayout, cfg: dict | None = None) -> dict:
        return load_hierarchical_knn(layout, cls.DEFAULT_LEVELS, cfg, tag='FM')

    def _pre_fit(self, branch_pairs: list, cfg) -> None:
        """Memorise les stats de la geometrie primaire (denorm pour la positivite)."""
        ds0, _ = branch_pairs[0]
        self.mu_train.value = jnp.array(ds0.mu, jnp.float32)
        self.sig_train.value = jnp.array(ds0.sig, jnp.float32)
        print(f"  SIAM pont baseline -> HR  (sigma_bridge={self.sigma_bridge})")

    def _sample_loss(self, hr: jax.Array, lr: jax.Array, tg: jax.Array, wt: jax.Array,
                     gp: jax.Array, knn_g: dict, key: jax.Array, aux: dict) -> jax.Array:
        """Loss du pont : b(x_t, t | cond) doit matcher la derive x1 - x0 (+ bruit)."""
        ctx = build_context(self, hr, lr, knn_g)
        x0 = ctx['baseline']   # champ LR interpole (normalise)
        x1 = tg                # champ HR cible (normalise)
        kt, kz = jax.random.split(key)
        t = self._sample_t(kt)
        z = jax.random.normal(kz, x1.shape)
        g, gprime = self._noise_env(t)
        x_t = (1.0 - t) * x0 + t * x1 + self.sigma_bridge * g * z
        drift_target = (x1 - x0) + self.sigma_bridge * gprime * z
        v = self.velocity(x_t, t, ctx, knn_g)
        loss = aux['loss_fn'](v, drift_target, wt)
        # Physique soft sur l'endpoint predit, ponderee par t : on ne la pose que
        # la ou l'endpoint x1_hat est fiable (evite la divergence aux petits t).
        if aux['lambda_phys'] > 0 or aux['lambda_enthalpy'] > 0:
            x1_hat = x_t + (1.0 - t) * v
            loss = loss + t * self._phys_terms(x1_hat, hr, gp, knn_g, aux)
        return loss
