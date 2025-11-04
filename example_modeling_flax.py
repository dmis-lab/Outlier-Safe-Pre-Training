from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Literal

import flax.linen as nn
import flax.linen.initializers as init
import jax
import jax.numpy as jnp
from chex import Array
from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel import (
    make_masked_mha_reference,
)
from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_mask import (
    CausalMask,
    MultiHeadMask,
)
from jax.sharding import PartitionSpec as P
from optax import softmax_cross_entropy_with_integer_labels as cross_entropy


@dataclass
class LlamaBase:
    layers: int = 24
    dim: int = 2048
    heads: int = 16
    vocab: int = 49152
    rope_theta: float = 10000.0

    rmsnorm_scale: Literal["single", "multi"] = "multi"
    rotated_embed: bool = False

    @property
    def kwargs(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(LlamaBase)}

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads

    @property
    def ffn_dim(self) -> int:
        return (8 * self.dim // 3 + 127) // 128 * 128

    @property
    def rmsnorm_feature_axes(self) -> int | tuple[int, ...]:
        return -1 if self.rmsnorm_scale == "multi" else ()


class LlamaAttention(LlamaBase, nn.Module):
    def setup(self):
        self.wq = nn.DenseGeneral((self.heads, self.head_dim))
        self.wk = nn.DenseGeneral((self.heads, self.head_dim))
        self.wv = nn.DenseGeneral((self.heads, self.head_dim))
        self.wo = nn.DenseGeneral(self.dim, axis=(-2, -1))

    def __call__(self, x: Array) -> Array:
        q = apply_rotary_emb(self.wq(x), rope_theta=self.rope_theta)
        k = apply_rotary_emb(self.wk(x), rope_theta=self.rope_theta)
        return self.wo(causal_attention(q, k, self.wv(x)))


class LlamaFeedForward(LlamaBase, nn.Module):
    def setup(self):
        self.w1 = nn.DenseGeneral(self.ffn_dim)
        self.w2 = nn.DenseGeneral(self.ffn_dim)
        self.w3 = nn.DenseGeneral(self.dim)

    def __call__(self, x: Array) -> Array:
        x = jax.lax.with_sharding_constraint(x, P(("dp", "op", "fsdp")))
        return self.w3(self.w1(x) * nn.silu(self.w2(x)))


class LlamaLayer(LlamaBase, nn.Module):
    def setup(self):
        self.attn = LlamaAttention(**self.kwargs)
        self.ffn = LlamaFeedForward(**self.kwargs)
        self.norm1 = nn.RMSNorm(feature_axes=self.rmsnorm_feature_axes)
        self.norm2 = nn.RMSNorm(feature_axes=self.rmsnorm_feature_axes)

    def __call__(self, x: Array) -> Array:
        x = jax.lax.with_sharding_constraint(x, P(("dp", "op", "fsdp")))
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Llama(LlamaBase, nn.Module):
    def setup(self):
        self.wte = nn.Embed(self.vocab, self.dim)
        self.layer = [LlamaLayer(**self.kwargs) for _ in range(self.layers)]
        self.norm = nn.RMSNorm(feature_axes=self.rmsnorm_feature_axes)
        self.head = nn.DenseGeneral(self.vocab)

        self.rot1 = self.rot2 = lambda x: x
        if self.rotated_embed:
            self.rot1 = nn.DenseGeneral(self.dim, kernel_init=init.orthogonal())
            self.rot2 = nn.DenseGeneral(self.dim, kernel_init=init.orthogonal())

    def __call__(self, tokens: Array) -> Array:
        x = self.rot1(self.wte(tokens))
        for layer in self.layer:
            x = layer(x)
        logits = self.head(self.rot2(self.norm(x))).astype(jnp.float32)
        return cross_entropy(logits[:, :-1], tokens[:, 1:]).mean()


def apply_rotary_emb(x: Array, rope_theta: float = 10000.0) -> Array:
    angle = jnp.arange(0, x.shape[-1], 2) / x.shape[-1]
    angle = jnp.arange(x.shape[1])[None, :, None, None] / rope_theta**angle
    sin, cos = jnp.sin(angle), jnp.cos(angle)

    # Split the array into real and imaginary part like complex number and manually
    # perform complex multiplication with the above cis.
    real, imag = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    real, imag = real * cos - imag * sin, imag * cos + real * sin
    return jnp.concatenate((real, imag), axis=-1).astype(x.dtype)


def causal_attention(q: Array, k: Array, v: Array) -> Array:
    mask = MultiHeadMask([CausalMask((q.shape[1], k.shape[1]))] * q.shape[2])
    mha_kernel = jax.vmap(make_masked_mha_reference(mask))

    q, k, v = jnp.moveaxis(q, 2, 1), jnp.moveaxis(k, 2, 1), jnp.moveaxis(v, 2, 1)
    return jnp.moveaxis(mha_kernel(q / math.sqrt(q.shape[-1]), k, v), 1, 2)
