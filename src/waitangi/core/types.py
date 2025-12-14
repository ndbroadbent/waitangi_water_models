"""Type definitions for the simulation system.

All array types are designed for GPU compatibility via JAX.
"""

from datetime import datetime
from typing import TypeAlias

import jax.numpy as jnp
import numpy as np
from jax import Array

# Scalar types
Scalar: TypeAlias = float | np.floating | Array

# Array types (GPU-compatible)
FloatArray: TypeAlias = Array | np.ndarray
IntArray: TypeAlias = Array | np.ndarray
ArrayLike: TypeAlias = FloatArray | list[float] | tuple[float, ...]

# Vector types
Vector2D: TypeAlias = tuple[Scalar, Scalar] | Array

# Time types
TimeStamp: TypeAlias = datetime | np.datetime64 | float  # float = seconds since epoch


def to_jax_array(arr: ArrayLike) -> Array:
    """Convert any array-like to a JAX array on the default device."""
    return jnp.asarray(arr)


def to_numpy_array(arr: ArrayLike) -> np.ndarray:
    """Convert any array-like to a NumPy array (CPU)."""
    if isinstance(arr, Array):
        return np.asarray(arr)
    return np.asarray(arr)


def ensure_float32(arr: ArrayLike) -> Array:
    """Ensure array is float32 for GPU efficiency."""
    return jnp.asarray(arr, dtype=jnp.float32)


def ensure_float64(arr: ArrayLike) -> Array:
    """Ensure array is float64 for precision-sensitive calculations."""
    return jnp.asarray(arr, dtype=jnp.float64)
