"""
SIAM : Stochastic Interpolant Attention Mesh (Model)
Interpolant stochastique conditionnel pour la super-resolution CFD.
Reference : Albergo, Boffi & Vanden-Eijnden, "Stochastic Interpolants:
A Unifying Framework for Flows and Diffusions", arXiv:2303.08797 (2023).

Au lieu de transporter du bruit gaussien vers le residu (FAM), on construit un
pont direct entre le champ basse resolution et le champ haute resolution :
    x0 = LR IDW        (champ LR interpole sur le maillage HR, deja physique)
    x1 = champ HR cible
    x_t = alpha(t)*x0 + beta(t)*x1 + gamma(t)*z ,   z ~ N(0, I)
Les coefficients de l'interpolant (cf. Albergo et al.) sont modulables :
    alpha, beta : le "chemin" LR IDW -> HR   (schema 'linear' ou 'trig')
    gamma       : l'"enveloppe de bruit" (variable latente, schema 'sin2'/'sin'/'sqrt')
avec pour conditions aux bords alpha(0)=beta(1)=1, alpha(1)=beta(0)=0 et
gamma(0)=gamma(1)=0 : les deux extremites du pont sont epinglees (x0 et x1
exacts), le bruit ne gonfle qu'au milieu. Le facteur d'echelle du bruit est
sigma (=sigma_bridge). Le reseau apprend la derive du pont
b(x_t, t | cond) = E[ d/dt x_t | x_t ], et l'echantillonnage integre l'ODE
dx/dt = b de x0 (LR IDW) a x1 (HR).

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
from models.dam import AMNet, build_context, load_hierarchical_knn, _LR_REF


class SIAM(SRModel):
    """Pont interpolant stochastique multi-echelles pour la super-resolution CFD :
    b(x_t, t | cond) sur hierarchie de maillages, integration de x0=LR IDW a x1=HR.
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

        # Tete denoiser : eta = E[z|x_t] (canaux 4:8) pour estimer le score du pont
        # s = -eta/gamma, requis par l'echantillonnage SDE. Gate pour ne pas changer les
        # shapes de checkpoint quand desactive (drift b seul, canaux 0:4).
        self.learn_score = bool(flow.get('learn_score', False))
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
            dtype=compute_dtype,
            n_out=8 if self.learn_score else 4)

        # Interpolant stochastique : x_t = alpha(t) x0 + beta(t) x1 + gamma(t) z.
        #   path  : schema du chemin alpha/beta      ('linear' | 'trig')
        #   gamma : schema de l'enveloppe de bruit   ('sin' | 'sqrt')
        #   sigma_bridge : facteur d'echelle du bruit (sigma)
        self.sigma_bridge = float(flow.get('sigma_bridge', 0.0))
        self._path = flow.get('path', 'linear')
        self._gamma_env = flow.get('gamma', 'sin')

        # Echelle du bruit du pont PAR CANAL
        #   ODE (learn_score=False) : echelle estimee EN LIGNE sur le residu du batch
        #   SDE (learn_score=True)  : buffer global fige une fois dans _pre_fit, pour
        #     la coherence entrainement / echantillonnage du score.

        self.per_channel_noise = bool(flow.get('per_channel_noise', True))
        self.noise_scale = Buffer(jnp.ones((4,), jnp.float32))
        self._nominal_lr = float((cfg.get('resolution') or {}).get('lr', _LR_REF))

        # Distribution d'echantillonnage de t a l'entrainement
        self._t_dist = flow.get('t_dist', 'uniform')
        self._sigma_t = float(flow.get('sigma_t', 1.0))
        self._t_alpha = float(flow.get('t_alpha', 2.0))
        self._t_beta = float(flow.get('t_beta', 2.0))

        # Echantillonnage : PF-ODE deterministe ('ode') ou SDE score-based ('sde', Euler-Maruyama).
        self.sampler = flow.get('sampler', 'ode')          # 'ode' | 'sde'
        self.lambda_score = float(flow.get('lambda_score', 1.0))  # poids perte denoising
        self.eps_schedule = flow.get('eps_schedule', 'gamma')  # 'gamma'|'const'|'linear'
        self.eps0 = float(flow.get('eps0', 1.0))            # echelle du coeff de diffusion
        self.n_sde_steps = flow.get('n_sde_steps', 32)      # pas Euler-Maruyama
        if self.sampler == 'sde' and not (self.learn_score and self.sigma_bridge > 0):
            raise ValueError(
                "sampler='sde' requiert learn_score=true et sigma_bridge>0 "
                "(sans bruit du pont, pas de score a estimer).")

        # Echantillonnage
        self.n_steps = flow.get('n_steps', 16)
        self.n_samples = flow.get('n_samples', 1)
        self.n_val_gen = flow.get('n_val_gen', 16)
        self.sample_seed = flow.get('sample_seed', 0)
        self.val_max_samples = self.n_val_gen

        # Residu type DAM (baseline + delta) : le pont opere en coords residu
        self.use_residual = flow.get('use_residual', True)

        # Dropout de conditionnement (classifier-free guidance) : regularise + guidance
        self.cfg_prob = float(flow.get('cfg_prob', 0.0))   # proba de dropout a l'entrainement
        self.cfg_scale = float(flow.get('cfg_scale', 1.0))  # echelle de guidance a l'inference

        # Stats d'entrainement (denormalisation pour la projection de positivite)
        self.mu_train = Buffer(jnp.zeros((4,), jnp.float32))
        self.sig_train = Buffer(jnp.ones((4,), jnp.float32))

    def _net_out(self, x_t: jax.Array, t: jax.Array, ctx: dict, knn: dict, scal: jax.Array | None = None) -> jax.Array:
        """Sortie brute du reseau : [N, 4] (drift b) ou [N, 8] (drift b + denoiser eta)."""
        s = ctx['scal'] if scal is None else scal
        return self.net(ctx['c'], s, knn['fm'], ctx['wall'], ctx['geom_id'], x_t=x_t, t=t)

    def velocity(self, x_t: jax.Array, t: jax.Array, ctx: dict, knn: dict,
                 scal: jax.Array | None = None) -> jax.Array:
        """Drift du pont b(x_t, t | cond) (canaux 0:4)."""
        return self._net_out(x_t, t, ctx, knn, scal=scal)[..., :4]

    def _guided_velocity(self, x_t: jax.Array, t: jax.Array, ctx: dict, knn: dict) -> jax.Array:
        """Vitesse du pont avec classifier-free guidance (inference)."""
        v_cond = self.velocity(x_t, t, ctx, knn)
        if self.cfg_scale == 1.0:
            return v_cond
        scal_null = jnp.zeros_like(ctx['scal'])
        v_uncond = self.velocity(x_t, t, ctx, knn, scal=scal_null)
        return v_uncond + self.cfg_scale * (v_cond - v_uncond)

    def _eps(self, t: jax.Array, gamma: jax.Array) -> jax.Array:
        """Coefficient de diffusion eps(t) >= 0 du SDE (preserve les marginales)."""
        if self.eps_schedule == 'const':
            return self.eps0 * jnp.ones_like(t)
        if self.eps_schedule == 'linear':
            return self.eps0 * (1.0 - t)
        return self.eps0 * gamma

    def _drift_score(self, x_t: jax.Array, t: jax.Array, ctx: dict, knn: dict) -> tuple:
        """Renvoie (drift guide b, score s) du pont pour l'echantillonnage SDE."""
        out = self._net_out(x_t, t, ctx, knn)
        b_cond, eta = out[..., :4], out[..., 4:8]
        if self.cfg_scale == 1.0:
            b = b_cond
        else:
            b_uncond = self.velocity(x_t, t, ctx, knn, scal=jnp.zeros_like(ctx['scal']))
            b = b_uncond + self.cfg_scale * (b_cond - b_uncond)
        _, _, _, _, gamma, _ = self._interp_coeffs(t)
        # residu normalise : bruit du pont unitaire -> score s = -eta/gamma
        s = -eta / jnp.clip(gamma, 1e-4, None)
        return b, s

    def _interp_coeffs(self, t: jax.Array) -> tuple[jax.Array, ...]:
        """Coefficients de l'interpolant stochastique et leurs derivees temporelles"""
        one = jnp.ones_like(t)
        if self._path == 'trig':    # interpolation trigonométrique
            half_pi = 0.5 * jnp.pi
            alpha, dalpha = jnp.cos(half_pi * t), -half_pi * jnp.sin(half_pi * t)
            beta,  dbeta  = jnp.sin(half_pi * t),  half_pi * jnp.cos(half_pi * t)
        else:       # interpolation linéaire
            alpha, dalpha = 1.0 - t, -one
            beta,  dbeta  = t, one

        # variable latente du pont
        s = self.sigma_bridge
        if self._gamma_env == 'sqrt':                  # gamma = sigma sqrt(t(1-t)) (pont brownien)
            tc = jnp.clip(t, 1e-4, 1.0 - 1e-4)
            root = jnp.sqrt(tc * (1.0 - tc))
            gamma, dgamma = s * root, s * (0.5 - tc) / root
        elif self._gamma_env == 'sin2':                # gamma = sigma sin(pi t)^2 : dgamma=0 aux bords
            sin = jnp.sin(jnp.pi * t)
            gamma, dgamma = s * sin * sin, s * jnp.pi * jnp.sin(2.0 * jnp.pi * t)
        else:                                          # 'sin' : gamma = sigma sin(pi t)
            gamma, dgamma = s * jnp.sin(jnp.pi * t), s * jnp.pi * jnp.cos(jnp.pi * t)
        return alpha, dalpha, beta, dbeta, gamma, dgamma

    def _sample_t(self, key: jax.Array) -> jax.Array:
        """Tire t ~ distribution configuree (uniform / logit_normal / beta)."""
        if self._t_dist == 'logit_normal':
            return jax.nn.sigmoid(jax.random.normal(key) * self._sigma_t)
        if self._t_dist == 'beta':
            return jax.random.beta(key, self._t_alpha, self._t_beta)
        return jax.random.uniform(key)
    
    def _res_scale(self, knn: dict | None = None) -> jax.Array:
        """Echelle par canal du residu HR-IDW (std), qui normalise le pont"""
        if knn is not None:
            rs = knn.get('res_scale')
            if rs is not None:
                return jnp.asarray(rs)[None, :]
        return self.noise_scale.value[None, :]

    @staticmethod
    def _branch_lr(knn_b) -> float:
        """Resolution LR d'une branche, retrouvee depuis res_scalar = log(lr/_LR_REF)."""
        rsc = knn_b.get('res_scalar')
        if rsc is None:
            return _LR_REF
        return float(_LR_REF * np.exp(float(np.asarray(rsc).reshape(-1)[0])))

    def _project(self, field: jax.Array) -> jax.Array:
        """Projection dure de positivite (rho, p >= eps) en unites physiques."""
        prim = field * self.sig_train.value + self.mu_train.value
        prim = prim.at[:, 0].set(jnp.maximum(prim[:, 0], 1e-3))
        prim = prim.at[:, 3].set(jnp.maximum(prim[:, 3], 1e-3))
        return (prim - self.mu_train.value) / self.sig_train.value

    def _integrate(self, ctx: dict, knn: dict) -> jax.Array:
        """Integre le pont (t=0) -> (t=1) par la PF-ODE (Heun)"""
        dt = 1.0 / self.n_steps
        rs = self._res_scale(knn)
        vscale = 1.0 if self.use_residual else rs

        def body(k, x):
            t1 = k * dt
            v1 = self._guided_velocity(x, t1, ctx, knn) * vscale
            v2 = self._guided_velocity(x + dt * v1, t1 + dt, ctx, knn) * vscale
            return x + 0.5 * dt * (v1 + v2)

        x0 = jnp.zeros_like(ctx['baseline']) if self.use_residual else ctx['baseline']
        x1 = jax.lax.fori_loop(0, self.n_steps, body, x0)
        field = ctx['baseline'] + x1 * rs if self.use_residual else x1
        return self._project(field)

    def _integrate_sde(self, ctx: dict, knn: dict, key: jax.Array) -> jax.Array:
        """Integre le pont par le SDE forward (Euler-Maruyama, n_sde_steps pas) :
            dX = [b + eps(t) s] dt + sqrt(2 eps(t)) dW ,   s = -eta/gamma."""
        dt = 1.0 / self.n_sde_steps
        sqrt_dt = jnp.sqrt(dt)

        rs = self._res_scale(knn)
        vscale = 1.0 if self.use_residual else rs   # increment normalise -> unites champ

        def body(k, carry):
            key, x = carry
            key, ksub = jax.random.split(key)
            t1 = k * dt
            b, s = self._drift_score(x, t1, ctx, knn)   # normalise par canal
            _, _, _, _, gamma, _ = self._interp_coeffs(t1)
            eps = self._eps(t1, gamma)             # residu normalise : diffusion unitaire
            xi = jax.random.normal(ksub, x.shape)
            dx = dt * (b + eps * s) + jnp.sqrt(2.0 * eps) * sqrt_dt * xi
            x = x + dx * vscale
            return key, x

        x0 = jnp.zeros_like(ctx['baseline']) if self.use_residual else ctx['baseline']
        _, x1 = jax.lax.fori_loop(0, self.n_sde_steps, body, (key, x0))
        # mode residu : denormalisation finale ; mode champ : x1 est deja le champ HR
        field = ctx['baseline'] + x1 * rs if self.use_residual else x1
        return self._project(field)

    def sample(self, hr_feat: jax.Array, lr_feat: jax.Array, knn: dict,
               key: jax.Array | None = None, n_samples: int | None = None) -> jax.Array:
        """Integre le pont LR IDW -> HR. Renvoie [S, N, 4].

        sampler='ode' : PF-ODE deterministe, les S echantillons sont identiques
        sampler='sde' : Euler-Maruyama stochastique, S trajectoires différentes (score-based sampling).
        """
        ctx = build_context(self, hr_feat, lr_feat, knn)
        n_s = n_samples if n_samples is not None else self.n_samples
        if self.sampler == 'sde':
            if key is None:
                key = jax.random.PRNGKey(self.sample_seed)
            keys = jax.random.split(key, n_s)
            return jax.vmap(lambda k: self._integrate_sde(ctx, knn, k))(keys)
        field = self._integrate(ctx, knn)
        return jnp.broadcast_to(field, (n_s,) + field.shape)

    def predict(self, hr_feat: jax.Array, lr_feat: jax.Array, knn: dict) -> jax.Array:
        # Moyenne des n samples pour la prediction finale (champ HR normalise)
        return self.sample(hr_feat, lr_feat, knn).mean(axis=0)

    @classmethod
    def load_knn(cls, layout: DataLayout, cfg: dict | None = None) -> dict:
        return load_hierarchical_knn(layout, cls.DEFAULT_LEVELS, cfg, tag='FM')

    def _branch_res_std(self, ds_b, knn_b) -> np.ndarray:
        """Std par canal du residu HR - IDW(LR) sur un sous-echantillon de la branche."""
        w0 = np.asarray(knn_b['fm']['cond'][0]['w'])
        i0 = np.asarray(knn_b['fm']['cond'][0]['idx'])
        n = len(ds_b)
        sub = np.unique(np.linspace(0, n - 1, min(256, n)).astype(int))
        acc = []
        for i in sub:
            _, lr, tg, *_ = ds_b[int(i)]
            lr, tg = np.asarray(lr), np.asarray(tg)
            baseline = (w0[:, :, None] * lr[:, :4][i0]).sum(axis=1)
            acc.append(tg - baseline)
        return np.maximum(np.concatenate(acc, 0).std(0), 1e-6)

    def _pre_fit(self, branch_pairs: list, cfg) -> None:
        """Memorise les stats de la geometrie primaire (denorm pour la positivite) et calibre
        l'echelle par canal du residu (res_scale)"""
        ds0, knn0 = branch_pairs[0]
        self.mu_train.value = jnp.array(ds0.mu, jnp.float32)
        self.sig_train.value = jnp.array(ds0.sig, jnp.float32)

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
            print(f"  res_scale{tag} (std residuel par canal) : "
                  + "  ".join(f"{v}={res_std[i]:.4f}"
                              for i, v in enumerate(('rho', 'u', 'v', 'p'))))
        # Buffer global (fallback hors branche connue) = branche a la resolution nominale
        self.noise_scale.value = jnp.array(branch_pairs[best_i][1]['res_scale'])
        if multi:
            print(f"  buffer res_scale : branche lr={self._branch_lr(branch_pairs[best_i][1]):g}"
                  f"  (resolution nominale {self._nominal_lr:g})")

        mode = "residu IDW + delta" if self.use_residual else "champ"
        print(f"  SIAM pont LR IDW -> HR normalise  (path={self._path}, gamma={self._gamma_env}, "
              f"sigma={self.sigma_bridge}, mode={mode})")

    def _sample_loss(self, hr: jax.Array, lr: jax.Array, tg: jax.Array, wt: jax.Array,
                     gp: jax.Array, knn_g: dict, key: jax.Array, aux: dict) -> jax.Array:
        """Loss du pont : b(x_t, t | cond) doit matcher la derive x1 - x0 (+ bruit)."""
        ctx = build_context(self, hr, lr, knn_g)
        rs = self._res_scale(knn_g)              # [1, 4] : std residuel par canal (fixe)
        kt, kz, kcfg = jax.random.split(key, 3)
        t = self._sample_t(kt)
        alpha, dalpha, beta, dbeta, gamma, dgamma = self._interp_coeffs(t)
        z = jax.random.normal(kz, ctx['baseline'].shape)   # bruit unitaire
        if self.use_residual:
            # Pont en coords residu NORMALISE : x0=0 -> x1=(HR-IDW)/rs ; entree ET sortie O(1)
            x0 = jnp.zeros_like(ctx['baseline'])
            x1 = (tg - ctx['baseline']) / rs
            x_t = alpha * x0 + beta * x1 + gamma * z
            drift_target = dalpha * x0 + dbeta * x1 + dgamma * z    # deja normalise
        else:
            # Vrai pont en espace-CHAMP (IDW -> HR) : x0=IDW, x1=HR, entree O(1), sortie O(1)
            x0 = ctx['baseline']
            x1 = tg
            x_t = alpha * x0 + beta * x1 + gamma * (rs * z)
            drift_target = (dalpha * x0 + dbeta * x1 + dgamma * (rs * z)) / rs

        # Dropout de conditionnement (classifier-free guidance), branche compilee une fois
        if self.cfg_prob > 0:
            drop = jax.random.bernoulli(kcfg, self.cfg_prob)
            scal_used = jnp.where(drop, jnp.zeros_like(ctx['scal']), ctx['scal'])
        else:
            scal_used = ctx['scal']
        out = self._net_out(x_t, t, ctx, knn_g, scal=scal_used)
        v = out[..., :4]
        loss = aux['loss_fn'](v, drift_target, wt)

        # Tete denoiser : eta = E[z|x_t]
        if self.learn_score:
            eta = out[..., 4:8]
            loss = loss + self.lambda_score * (wt[:, None] * (eta - z) ** 2).mean()

        # Termes physiques : conservation de la masse et de l'energie (enthalpie)
        if aux['lambda_phys'] > 0 or aux['lambda_enthalpy'] > 0:
            det = alpha * dbeta - beta * dalpha
            if self.use_residual:                # v = vitesse residu normalisee
                x1_hat = (alpha * v - dalpha * x_t) / det
                field = ctx['baseline'] + x1_hat * rs
            else:                                # v normalisee -> vitesse champ = v*rs
                x1_hat = (alpha * (v * rs) - dalpha * x_t) / det
                field = x1_hat
            loss = loss + t * self._phys_terms(field, hr, gp, knn_g, aux)
        return loss
