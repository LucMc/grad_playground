from collections.abc import Callable
import functools
from typing import Any, Optional, Union
import warnings

import jax
import jax.numpy as jnp
from optax._src import base
from optax._src import clipping
from optax._src import combine
from optax._src import factorized
from optax._src import linesearch as _linesearch
from optax._src import transform
from optax._src import utils
from optax._src import wrappers

AddDecayedWeightsState = base.EmptyState


def add_grad_decayed_weights(
    weight_decay: Union[float, jax.Array] = 0.0,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None
) -> base.GradientTransformation:
  """Add parameter scaled by `weight_decay`.

  Args:
    weight_decay: A scalar weight decay rate.
    mask: A tree with same structure as (or a prefix of) the params PyTree,
      or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the transformation to, and `False` for those you want to skip.

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params):
    del params
    return AddDecayedWeightsState()

  def update_fn(updates, state, params):
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)

    # Hyperparams
    k = 10
    t = 1
    rr = 2e-4 # 2x normal wd auc

    grads_batch_avg = jnp.mean(updates, axis=0)
    grads_avg = jnp.mean(grads_batch_avg, axis=(0,1)) # Scalar
    normed_grads = grads_batch_avg / grads_avg # Mean of 1
    grad_weight_decay = rr * jax.nn.sigmoid(-k * (x - t)) # TODO: Plot some metrics/make tests

    updates = jax.tree_util.tree_map(
        lambda g, p: g + grad_weight_decay * p, updates, params)

    return updates, state

  # If mask is not `None`, apply mask to the gradient transformation.
  # E.g. it is common to skip weight decay on bias units and batch stats.
  if mask is not None:
    return wrappers.masked(
        base.GradientTransformation(init_fn, update_fn), mask)

  return base.GradientTransformation(init_fn, update_fn)


def adam(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
    weight_decay: float = 1e-4,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    *,
    nesterov: bool = False,
) -> base.GradientTransformationExtraArgs:

  return combine.chain(
      transform.scale_by_adam(
          b1=b1,
          b2=b2,
          eps=eps,
          eps_root=eps_root,
          mu_dtype=mu_dtype,
          nesterov=nesterov,
      ),
      transform.scale_by_learning_rate(learning_rate),
  )


def adamw(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
    weight_decay: float = 1e-4,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    *,
    nesterov: bool = False,
) -> base.GradientTransformationExtraArgs:

  return combine.chain(
      transform.scale_by_adam(
          b1=b1,
          b2=b2,
          eps=eps,
          eps_root=eps_root,
          mu_dtype=mu_dtype,
          nesterov=nesterov,
      ),
      transform.add_decayed_weights(weight_decay, mask),
      transform.scale_by_learning_rate(learning_rate),
  )



def adam_gwd(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
    weight_decay: float = 1e-4,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    *,
    nesterov: bool = False,
) -> base.GradientTransformationExtraArgs:

  return combine.chain(
      transform.scale_by_adam(
          b1=b1,
          b2=b2,
          eps=eps,
          eps_root=eps_root,
          mu_dtype=mu_dtype,
          nesterov=nesterov,
      ),
      add_grad_decayed_weights(weight_decay, mask),
      transform.scale_by_learning_rate(learning_rate),
  )

"""
# Normal weight decay
def add_decayed_weights(
    weight_decay: Union[float, jax.Array] = 0.0,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None
) -> base.GradientTransformation:

  def init_fn(params):
    del params
    return AddDecayedWeightsState()

  def update_fn(updates, state, params):
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)
    updates = jax.tree_util.tree_map(
        lambda g, p: g + weight_decay * p, updates, params)
    return updates, state

  # If mask is not `None`, apply mask to the gradient transformation.
  # E.g. it is common to skip weight decay on bias units and batch stats.
  if mask is not None:
    return wrappers.masked(
        base.GradientTransformation(init_fn, update_fn), mask)
  return base.GradientTransformation(init_fn, update_fn)

"""
