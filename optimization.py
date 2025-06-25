# Copyright 2025 Jungwoo Park (affjljoo3581)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections import defaultdict
from functools import partial
from typing import Any, Callable

import jax
import jax.experimental.multihost_utils
import jax.numpy as jnp
import optax
import optax.tree_utils as otu
from chex import Array, ArrayTree
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.interpreters import pxla
from jax.sharding import PartitionSpec as P
from optax._src.base import init_empty_state


def _newtonschulz5(x: Array, steps: int = 5, eps: float = 1e-8) -> Array:
    def body_fn(_: int, x: Array) -> Array:
        return 3.4445 * x + (-4.7750 * (z := x @ x.T) + 2.0315 * z @ z) @ x

    x = (x.T if (tr := x.shape[0] > x.shape[1]) else x) / (jnp.linalg.norm(x) + eps)
    x = jax.lax.fori_loop(0, steps, body_fn, init_val=x.astype(jnp.bfloat16))
    return (x.T if tr else x).astype(jnp.float32)


def scale_by_muon(
    beta: float = 0.95,
    steps: int = 5,
    eps: float = 1e-8,
) -> optax.GradientTransformation:
    ortho_fn = partial(_newtonschulz5, steps=steps, eps=eps)
    op_rank = pxla.thread_resources.env.physical_mesh.shape["op"]

    def batch_ortho_fn(param: Array) -> Array:
        if param.ndim <= 1:
            return param / (jnp.linalg.norm(param) + eps)
        if param.ndim == 2:
            return ortho_fn(param)
        if param.ndim == 3:
            return jax.vmap(ortho_fn, in_axes=0, out_axes=0)(param)
        raise NotImplementedError(f"{param.ndim} shoud be one of {{0, 1, 2, 3}}.")

    def init_fn(params: ArrayTree) -> optax.TraceState:
        return optax.TraceState(trace=otu.tree_zeros_like(params))

    def update_fn(
        updates: ArrayTree, state: optax.TraceState, params: ArrayTree | None = None
    ) -> tuple[ArrayTree, optax.TraceState]:
        del params
        buffers = jax.tree.map(lambda g, t: g + beta * t, updates, state.trace)
        updates = jax.tree.map(lambda g, t: g + beta * t, updates, buffers)

        # Gather gradients with the same shape so they can be orthogonalized together
        # and then reassigned to their corresponding parameters.
        new_updates, grad_groups, name_groups = {}, defaultdict(list), defaultdict(list)
        for name, grad in flatten_dict(updates).items():
            if isinstance(grad, jax.Array):
                grad_groups[grad.shape].append(grad)
                name_groups[grad.shape].append(name)
            else:
                new_updates[name] = grad

        # If gradients can be distributed across devices based on the optimizer
        # parallelism rank, they will be stacked and orthogonalized in a single
        # operation. Otherwise, they will be normalized individually.
        for shape, grads in grad_groups.items():
            print(f"[*] Muon Parallelsim: {len(grads)} x {shape}")
            if len(shape) == 2 and (chunk_size := len(grads) // op_rank * op_rank) > 0:
                chunks, grads = jnp.stack(grads[:chunk_size]), grads[chunk_size:]
                chunks = jax.lax.with_sharding_constraint(chunks, P("op", "fsdp"))
                grads = list(batch_ortho_fn(chunks)) + list(map(batch_ortho_fn, grads))
            else:
                grads = list(map(batch_ortho_fn, grads))
            for name, grad in zip(name_groups[shape], grads):
                if len(grad.shape) > 0:
                    grad = jax.lax.with_sharding_constraint(grad, P("fsdp"))
                new_updates[name] = grad
        return unflatten_dict(new_updates), optax.TraceState(trace=buffers)

    return optax.GradientTransformation(init_fn, update_fn)


def muon(
    learning_rate: optax.ScalarOrSchedule,
    beta: float = 0.95,
    steps: int = 5,
    eps: float = 1e-8,
    weight_decay: float = 1e-2,
    mask: Any | Callable[[optax.Params], Any] | None = None,
) -> optax.GradientTransformation:
    def matching_update_rms_of_adamw_fn(
        updates: ArrayTree, state: optax.EmptyState, params: ArrayTree | None = None
    ) -> tuple[ArrayTree, optax.EmptyState]:
        return jax.tree.map(lambda x: x * max(x.shape or (1,)) ** 0.5, updates), state

    return optax.chain(
        scale_by_muon(beta=beta, steps=steps, eps=eps),
        optax.GradientTransformation(init_empty_state, matching_update_rms_of_adamw_fn),
        optax.scale(0.2),
        optax.add_decayed_weights(weight_decay, mask),
        optax.scale_by_learning_rate(learning_rate),
    )
