"""
Attention utilities : fonctions et blocs nnx réutilisables entre modèles.

Primitives (fonctions pures JAX) :
- fourier_pos_enc : encodage positionnel Fourier (positions relatives)
- sinusoidal_embedding : encodage sinusoïdal du temps t dans [0, 1]
- local_cross_attention: cross-attention locale kNN geometry-aware

Blocs nnx conditionnés (partagés FM et AE décodeur) :
- AdaLNBlock : bloc résiduel AdaLN-Zero -> MLP (LN + modulation cond init=0)
- LocalAttnBlock : cross-attention locale + AdaLN sur les requêtes
- GlobalAttnBlock : self-attention complète + AdaLN

PE_DIM = 32  (4 * N_FREQS = 4 * 8)
"""

import math

import jax
import jax.numpy as jnp
from flax import nnx

_HAS_FLASH = hasattr(jax.nn, 'dot_product_attention')

N_FREQS = 8  # nombre de bandes de fréquences par axe spatial
PE_DIM = 4 * N_FREQS  # 32 (sin+cos x x+y x n_freqs)


def fourier_pos_enc(rel_pos: jax.Array) -> jax.Array:
    # Encodage positionnel Fourier : (..., 2) -> (..., PE_DIM), fréquences log-espacées.
    freqs = jnp.exp(
        jnp.linspace(0.0, math.log(100.0), N_FREQS, dtype=jnp.float32)
    )
    args = rel_pos[..., None] * freqs
    enc = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
    return enc.reshape(rel_pos.shape[:-1] + (PE_DIM,))


def local_cross_attention(Q: jax.Array, KV: jax.Array, knn_idx: jax.Array, rel_pos: jax.Array,
            mlp_k, mlp_v, n_heads: int = 1, ) -> jax.Array:
    """ Cross attention locale kNN (geometry aware) : Q (N_q, d), KV (N_kv, d), knn_idx (N_q, k), rel_pos (N_q, k, 2)
    - Q : requêtes (N_q, d)
    - KV : clés/valeurs (N_kv, d)
    - knn_idx : indices des k voisins KV pour chaque requête Q (N_q, k)
    - rel_pos : positions relatives normalisées par l'échelle locale (N_q, k, 2)
    - mlp_k : MLP pour transformer les clés (d+PE_DIM -> d)
    - mlp_v : MLP pour transformer les valeurs (d -> d)
    - n_heads : nombre de têtes d'attention
    """
    N_q, k = knn_idx.shape
    d = Q.shape[-1]
    if d % n_heads:
        raise ValueError(f"d={d} non divisible par n_heads={n_heads}")
    dh = d // n_heads

    pe = fourier_pos_enc(rel_pos)  # (N_q, k, PE_DIM)
    kv_feats = KV[knn_idx]  # (N_q, k, d)

    k_in = jnp.concatenate([kv_feats, pe], axis=-1)
    K = mlp_k(k_in.reshape(N_q * k, -1)).reshape(N_q, k, n_heads, dh)
    V = mlp_v(kv_feats.reshape(N_q * k, d)).reshape(N_q, k, n_heads, dh)

    Qh = Q.reshape(N_q, n_heads, dh)
    attn = jnp.einsum('nhd,nkhd->nhk', Qh, K) / math.sqrt(dh)
    w = jax.nn.softmax(attn, axis=-1)
    return jnp.einsum('nhk,nkhd->nhd', w, V).reshape(N_q, d)


def sinusoidal_embedding(t: jax.Array, dim: int) -> jax.Array:
    # Embedding sinusoïdal du temps t dans [0, 1]
    half = dim // 2
    freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half, dtype=jnp.float32) / half)
    args = 1000.0 * jnp.asarray(t, jnp.float32) * freqs
    return jnp.concatenate([jnp.sin(args), jnp.cos(args)])


def _zeros_linear(din: int, dout: int, rngs: nnx.Rngs, dtype=None) -> nnx.Linear:
    # Linear initialisé à zéro (décodeur / modulation identité au départ)
    lin = nnx.Linear(din, dout, rngs=rngs, dtype=dtype)
    lin.kernel.value = jnp.zeros_like(lin.kernel.value)
    lin.bias.value = jnp.zeros_like(lin.bias.value)
    return lin


class AdaLNBlock(nnx.Module):
    #Bloc résiduel par noeud : AdaLN-Zero -> MLP -> +x
    # Permet de moduler les features par un vecteur de conditionnement (ex : temps, Mach, AoA, etc..)

    def __init__(self, d: int, d_cond: int, rngs: nnx.Rngs, mlp_ratio: int = 2, dtype=None):
        self.norm = nnx.LayerNorm(d, use_scale=False, use_bias=False, rngs=rngs)
        self.gamma = _zeros_linear(d_cond, d, rngs, dtype=dtype)
        self.beta = _zeros_linear(d_cond, d, rngs, dtype=dtype)
        self.fc1 = nnx.Linear(d, mlp_ratio * d, rngs=rngs, dtype=dtype)
        self.fc2 = nnx.Linear(mlp_ratio * d, d, rngs=rngs, dtype=dtype)

    def __call__(self, x: jax.Array, cond: jax.Array) -> jax.Array:
        h = self.norm(x) * (1.0 + self.gamma(cond)) + self.beta(cond)
        return x + self.fc2(jax.nn.silu(self.fc1(h)))


class LocalAttnBlock(nnx.Module):
    """Cross-attention locale kNN + AdaLN sur les requêtes (pooling, unpooling ou self).

    Les positions relatives sont normalisées par l'échelle locale du voisinage
    (précalculé dans load_knn), indépendant de la résolution du maillage.
    """

    def __init__(self, q_dim: int, d: int, d_cond: int, n_heads: int, rngs: nnx.Rngs, dtype=None):
        from models.base import MLP as _MLP
        self.q_norm = nnx.LayerNorm(q_dim, rngs=rngs)
        self.q_proj = nnx.Linear(q_dim, d, rngs=rngs, dtype=dtype)
        self.gamma = _zeros_linear(d_cond, d, rngs, dtype=dtype)
        self.beta = _zeros_linear(d_cond, d, rngs, dtype=dtype)
        self.kv_norm = nnx.LayerNorm(d, rngs=rngs)
        self.k_mlp = _MLP([d + PE_DIM, d], rngs, dtype=dtype)
        self.v_mlp = _MLP([d, d], rngs, dtype=dtype)
        self.out = nnx.Linear(d, d, rngs=rngs, dtype=dtype)
        self.n_heads = n_heads

    def __call__(self, q_in: jax.Array, kv: jax.Array, idx: jax.Array,
                 rel: jax.Array, cond: jax.Array) -> jax.Array:
        q = self.q_proj(self.q_norm(q_in))
        q = q * (1.0 + self.gamma(cond)) + self.beta(cond)
        ctx = local_cross_attention(q, self.kv_norm(kv), idx, rel,
                                    self.k_mlp, self.v_mlp, self.n_heads)
        return self.out(ctx)


class GlobalAttnBlock(nnx.Module):
    """Self-attention complète au niveau le plus grossier, Flash Attention si disponible pour accélérer."""

    def __init__(self, d: int, d_cond: int, n_heads: int, rngs: nnx.Rngs, mlp_ratio: int = 2, dtype=None):
        if d % n_heads:
            raise ValueError(f"d={d} non divisible par n_heads={n_heads}")
        self.dh = d // n_heads
        self.n_heads = n_heads
        self.norm1 = nnx.LayerNorm(d, use_scale=False, use_bias=False, rngs=rngs)
        self.g1 = _zeros_linear(d_cond, d, rngs, dtype=dtype)
        self.b1 = _zeros_linear(d_cond, d, rngs, dtype=dtype)
        self.wq = nnx.Linear(d, d, use_bias=False, rngs=rngs, dtype=dtype)
        self.wk = nnx.Linear(d, d, use_bias=False, rngs=rngs, dtype=dtype)
        self.wv = nnx.Linear(d, d, use_bias=False, rngs=rngs, dtype=dtype)
        self.out_proj = nnx.Linear(d, d, rngs=rngs, dtype=dtype)
        self.norm2 = nnx.LayerNorm(d, use_scale=False, use_bias=False, rngs=rngs)
        self.g2 = _zeros_linear(d_cond, d, rngs, dtype=dtype)
        self.b2 = _zeros_linear(d_cond, d, rngs, dtype=dtype)
        self.fc1 = nnx.Linear(d, mlp_ratio * d, rngs=rngs, dtype=dtype)
        self.fc2 = nnx.Linear(mlp_ratio * d, d, rngs=rngs, dtype=dtype)

    def __call__(self, x: jax.Array, cond: jax.Array) -> jax.Array:
        N, d = x.shape
        h = self.norm1(x) * (1.0 + self.g1(cond)) + self.b1(cond)
        q = self.wq(h).reshape(N, self.n_heads, self.dh)
        k = self.wk(h).reshape(N, self.n_heads, self.dh)
        v = self.wv(h).reshape(N, self.n_heads, self.dh)
        if _HAS_FLASH:
            # (N, n_heads, dh),Flash Attention si CUDA + bf16/f16, sinon XLA
            attn_out = jax.nn.dot_product_attention(q, k, v)
        else:
            # Fallback : softmax attention O(N2) sur CPU / GPU non supporté
            scores = jnp.einsum('ihd,jhd->hij', q, k) / math.sqrt(self.dh)
            w = jax.nn.softmax(scores, axis=-1)
            attn_out = jnp.einsum('hij,jhd->ihd', w, v)
        x = x + self.out_proj(attn_out.reshape(N, d).astype(x.dtype))
        h = self.norm2(x) * (1.0 + self.g2(cond)) + self.b2(cond)
        return x + self.fc2(jax.nn.silu(self.fc1(h)))
