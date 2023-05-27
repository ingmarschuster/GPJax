import abc
from abc import (
    ABC,
    abstractmethod,
)
from copy import copy
from dataclasses import dataclass
from functools import partial
from typing import Any, Generic

from beartype.typing import Callable, List, Optional, Tuple, Type, TypeVar, Union, Any
import jax.numpy as jnp
import jax.numpy as np
from jaxtyping import (
    Array,
    Float,
    Int,
    Num,
)
import tensorflow_probability.substrates.jax.distributions as tfd
from gpjax.base import param_field, static_field

from gpjax.base import (
    Module,
    param_field,
    static_field,
)
from gpjax.typing import (
    ScalarBool,
    ScalarFloat,
    ScalarInt,
)

from . import reduce as red
from .. import kernels
from .base import AbstractEncoder
from .vector import RkhsVec


@dataclass
class StandardEncoder(AbstractEncoder):
    """Encodes input array into an RKHS vector, where the kernel is applied to each element of the input array.
    In other words, this is the standard mapping of classical kernel methods."""

    k: kernels.AbstractKernel = param_field(kernels.RBF())

    def __call__(self, inp: Any) -> AbstractEncoder:
        """Encodes input array into an RKHS vector, where the kernel is applied to each element of the input array.

        Args:
            inp (Array): Input array.

        Returns:
            FiniteVec: RKHS vector.
        """
        return RkhsVec(k=self.k, insp_pts=inp)
