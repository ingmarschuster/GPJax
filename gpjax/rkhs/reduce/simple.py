import collections.abc
from dataclasses import dataclass


from beartype.typing import (
    Callable,
    List,
    Tuple,
    TypeVar,
    Union,
)
import jax.numpy as np

import jax.scipy as sp
import jax.scipy.stats as stats
from jaxtyping import Array, Float32, Int32, Float, Int
from gpjax.base.param import param_field

from gpjax.typing import (
    ScalarInt,
    ScalarFloat,
)

from .base import (
    LinearizableReduce,
    LinearReduce,
    ChainedReduce,
)

from ..base import (
    ReduceableOrArray,
    NumberOrArray,
    AbstractReduce,
    AbstractReduceable,
    NoReduce,
    ChainedReduce,
)


@dataclass
class Prefactors(LinearizableReduce):
    """Multiply the input array with a set of prefactors"""

    prefactors: Float[Array, "N"] = param_field(np.ones((1,)))

    def __reduce_array__(self, inp: Float[Array, "N ..."]) -> Float[Array, "M ..."]:
        """Multiply the input array with the prefactors.

        Args:
            inp (Array): Input array, typically a gram matrix.

        Returns:
            Array: Input array multiplied with the prefactors.

        Example:
            >>> from jax import numpy as jnp
            >>> from jaxrk.reduce import Prefactors
            >>> p = Prefactors(jnp.array([1,2,3]))
            >>> m = jnp.array([[1,2,3],[4,5,6],[7,8,9]])
            >>> p(m)
            DeviceArray([[ 1,  2,  3],
                         [ 8, 10, 12],
                         [21, 24, 27]], dtype=int32)
            >>> p(m, axis=1)
            DeviceArray([[1,  4,  9],
                         [4, 10, 18],
                         [7, 16, 27]], dtype=int32)
        """
        if self.prefactors.shape[0] != inp.shape[0]:
            raise ValueError(
                f"Prefactors shape {self.prefactors.shape} does not match input shape {inp.shape} along axis 0"
            )
        return inp.astype(self.prefactors.dtype) * np.expand_dims(
            self.prefactors, axis=(0 + 1) % 2
        )

    def __apply_reduce__(self, other_reduce: "AbstractReduce") -> "AbstractReduce":
        """Apply a reduction to the `self` object.

        Args:
            reduce (AbstractReduce): The reduction to apply.

        Returns:
            Any: The result of applying the reduction.
        """
        if isinstance(other_reduce, NoReduce):
            return self
        return ChainedReduce([other_reduce, self])

    def new_len(self, original_len: int) -> int:
        """Compute the new length of the array after reduction.

        Args:
            original_len (int): Original length of the array.

        Returns:
            int: Length of the array after reduction.
        """
        if original_len != len(self.prefactors):
            raise ValueError(
                f"Prefactors shape {self.prefactors.shape} does not match input shape {original_len}"
            )
        return original_len

    def linmap(self, gram_shape: tuple) -> Array:
        """Compute the linear map associated with the prefactors.

        Args:
            gram_shape (tuple): Shape of the gram matrix.
            axis (int, optional): Axis along which to compute the linear map. Defaults to 0.

        Returns:
            Array: Linear map associated with the prefactors.
        """
        return np.diag(self.prefactors)


@dataclass
class Scale(LinearizableReduce):
    """Scale the input array by a constant factor."""

    s: ScalarFloat = param_field(1.0)

    def __reduce_array__(self, inp: Float[Array, "N ..."]) -> Float[Array, "M ..."]:
        """Scale the input array.

        Args:
            inp (ReduceOrArray): Input array, typically a gram matrix.


        Returns:
            ReduceOrArray: Scaled input array.
        """
        return inp * self.s

    def __apply_reduce__(self, other_reduce: "AbstractReduce") -> "AbstractReduce":
        """Apply a reduction to the `self` object.

        Args:
            reduce (AbstractReduce): The reduction to apply.

        Returns:
            Any: The result of applying the reduction.
        """
        if isinstance(other_reduce, NoReduce):
            return self
        return ChainedReduce([other_reduce, self])

    def new_len(self, original_len: int) -> int:
        """Compute the new length of the array after reduction.

        Args:
            original_len (int): Original length of the array.

        Returns:
            int: Length of the array after reduction, i.e. `original_len`.
        """
        return original_len

    def linmap(self, gram_shape: tuple) -> Array:
        """Compute the linear map associated with the scaling.

        Args:
            gram_shape (tuple): Shape of the gram matrix.
            axis (int, optional): Axis along which to compute the linear map. Defaults to 0.

        Returns:
            Array: Linear map associated with the scaling.
        """
        return self.s * np.eye(gram_shape[0])


class Sum(LinearizableReduce):
    """Sum the input array."""

    def __reduce_array__(self, inp: Float[Array, "N ..."]) -> Float[Array, "M ..."]:
        """Sum the input array.

        Args:
            inp (Array): Input array, typically a gram matrix.

        Returns:
            Array: Summed input array.

        Example:
            >>> from jax import numpy as jnp
            >>> from jaxrk.reduce import Sum
            >>> s = Sum()
            >>> m = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            >>> s(m, axis=0)
            DeviceArray([[12, 15, 18]], dtype=int32)
            >>> s(m, axis=1)
            DeviceArray([[ 6],
                         [15],
                         [24]], dtype=int32)
        """
        return inp.sum(axis=0, keepdims=True)

    def __apply_reduce__(self, other_reduce: "AbstractReduce") -> "AbstractReduce":
        """Apply a reduction to the `self` object.

        Args:
            reduce (AbstractReduce): The reduction to apply.

        Returns:
            Any: The result of applying the reduction.
        """
        if isinstance(other_reduce, NoReduce):
            return self
        return ChainedReduce([other_reduce, self])

    def new_len(self, original_len: int) -> int:
        """Compute the new length of the array after reduction.

        Args:
            original_len (int): Original length of the array.

        Returns:
            int: Length of the array after reduction.
        """
        return 1

    def linmap(self, gram_shape: tuple) -> Array:
        """Linear map version of `Sum` reduction.

        Args:
            gram_shape (tuple): Shape of the input matrix.
            axis (int, optional): Axis to apply reduction over. Defaults to 0.

        Returns:
            Array: A linear operator that can be applied to the input matrix to sum over `axis`.
        """
        return np.ones((1, gram_shape[0]))


class Mean(LinearizableReduce):
    """Average the input array."""

    def __reduce_array__(self, inp: Float[Array, "N ..."]) -> Float[Array, "M ..."]:
        """Average the input array.

        Args:
            inp (Array): Input array, typically a gram matrix.

        Returns:
            Array: Reduced input array.
        """
        if isinstance(inp, AbstractReduce):
            return ChainedReduce([self, inp])
        return np.mean(inp, axis=0, keepdims=True)

    def __apply_reduce__(self, other_reduce: "AbstractReduce") -> "AbstractReduce":
        """Apply a reduction to the `self` object.

        Args:
            reduce (AbstractReduce): The reduction to apply.

        Returns:
            Any: The result of applying the reduction.
        """
        if isinstance(other_reduce, NoReduce):
            return self
        return ChainedReduce([other_reduce, self])

    def new_len(self, original_len: int) -> int:
        """Compute the new length of the array after reduction.

        Args:
            original_len (int): Original length of the array.

        Returns:
            int: Length of the array after reduction.
        """
        return 1

    def linmap(self, gram_shape: tuple) -> Array:
        """Linear map version of mean reduction.

        Args:
            gram_shape (tuple): Shape of the input matrix.
            axis (int, optional): Axis to apply reduction over. Defaults to 0.

        Returns:
            Array: A linear operator that can be applied to the input matrix to average over `axis`.
        """
        return np.ones((1, gram_shape[0])) / gram_shape[0]


@dataclass
class BalancedRed(LinearizableReduce):
    """Balanced reduction of the input array. Sums up a number of consecutive elements in the input.

    Args:
        points_per_split (int): Number of points per split, i.e. number of dimensions to sum up to a single result dimension.
        average (bool, optional): If True, average rather than sum up dimensions. Defaults to False.
    """

    points_per_split: int
    average: bool = False

    def __post_init__(self):
        super().__init__()
        if self.points_per_split < 2:
            raise ValueError("points_per_split must be at least 2")
        if self.average:
            self.red = np.mean
            self.factor = 1.0 / self.points_per_split
        else:
            self.red = np.sum
            self.factor = 1.0

    def __reduce_array__(self, inp: Float[Array, "N ..."]) -> Float[Array, "M ..."]:
        """Sums up a fixed number of consecutive elements in the input array (and potentially divide by the number of elements).

        Args:
            inp (Array): Input array, typically a gram matrix.

        Returns:
            Array: Reduced input array.

        Example:
            >>> from jax import numpy as jnp
            >>> from jaxrk.reduce import BalancedRed
            >>> b = BalancedRed(2)
            >>> m = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]])
            >>> b(m)
            DeviceArray([[ 6,  8, 10, 12]], dtype=int32)
            >>> b(m.T).T
            DeviceArray([[ 3,  7],
                         [11, 15]], dtype=int32)
        """
        rval = self.red(
            np.reshape(inp, (-1, self.points_per_split, inp.shape[-1])), axis=1
        )
        return rval

    def __apply_reduce__(self, other_reduce: "AbstractReduce") -> "AbstractReduce":
        """Apply a reduction to the `self` object.

        Args:
            reduce (AbstractReduce): The reduction to apply.

        Returns:
            Any: The result of applying the reduction.
        """
        if isinstance(other_reduce, NoReduce):
            return self
        return ChainedReduce([other_reduce, self])

    def linmap(self, inp_shape: tuple) -> Array:
        """Linear map version of `BalancedRed` reduction.

        Args:
            inp_shape (tuple): Shape of the input matrix.
            axis (int, optional): Axis to apply reduction over. Defaults to 0.

        Returns:
            Array: A linear operator that can be applied to the input matrix and get the same result as the reduction.
        """
        new_len = self.new_len(inp_shape[0])
        rval = np.zeros((new_len, inp_shape[0]))
        for i in range(new_len):
            rval = rval.at[
                i, i * self.points_per_split : (i + 1) * self.points_per_split
            ].set(self.factor)
        return rval

    def new_len(self, original_len: int) -> int:
        """Compute the new length of the array after reduction.

        Args:
            original_len (int): Original length of the array.

        Returns:
            int: Length of the array after reduction.
        """
        assert original_len % self.points_per_split == 0
        return original_len // self.points_per_split


class Center(LinearizableReduce):
    """Center the input array by subtracting the mean."""

    def __reduce_array__(self, inp: Float[Array, "N ..."]) -> Float[Array, "M ..."]:
        """Center input along axis.

        Args:
            inp (Array): Input array, typically a gram matrix.

        Returns:
            Array: Centered input array.
        """
        if isinstance(inp, AbstractReduce):
            return ChainedReduce([self, inp])
        return inp - np.mean(inp, 0, keepdims=True)

    def __apply_reduce__(self, other_reduce: "AbstractReduce") -> "AbstractReduce":
        """Apply a reduction to the `self` object.

        Args:
            reduce (AbstractReduce): The reduction to apply.

        Returns:
            Any: The result of applying the reduction.
        """
        if isinstance(other_reduce, NoReduce):
            return self
        return ChainedReduce([other_reduce, self])

    def linmap(self, gram_shape: tuple) -> Array:
        """Linear map version of `Center` reduction.

        Args:
            gram_shape (tuple): Shape of the input matrix.
            axis (int, optional): Axis to apply reduction over. Defaults to 0.

        Returns:
            Array: A linear operator that can be applied to the input matrix to center over `axis`.
        """
        return (
            np.eye(gram_shape[0])
            - np.ones((gram_shape[0], gram_shape[0])) / gram_shape[0]
        )

    def new_len(self, original_len: int) -> int:
        """Compute the new length of the array after reduction.

        Args:
            original_len (int): Original length of the array.

        Returns:
            int: Length of the array after reduction.
        """
        return original_len
