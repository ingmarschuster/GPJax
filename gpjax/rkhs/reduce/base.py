from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any
from jaxtyping import Float, Array, Int
from beartype.typing import Union, Callable, List, TypeVar, Tuple
import jax.numpy as np
from gpjax.typing import ScalarInt, ScalarFloat, ScalarBool


from ..typing import (
    ReduceableOrArray,
    NumberOrArray,
    AbstractReduce,
    NoReduce,
    ChainedReduce,
)


@dataclass
class LinearReduce(AbstractReduce):
    """Reduction defined by a linear map."""

    linear_map: Float[Array, "N M"]

    def __reduce_array__(self, inp: Float[Array, "N ..."]) -> Float[Array, "M ..."]:
        """Reduce the first axis of the input matrix.

        Args:
            inp (ReduceOrArray): The array to reduce. Typically a gram matrix.

        Returns:
            Array: The array reduced along the first axis.
        """
        if len(inp.shape) != 2:
            raise ValueError(
                "LinearReduce expects a 2D array, got %dD" % len(inp.shape)
            )
        if self.linear_map.shape[-1] != inp.shape[0]:
            raise ValueError(
                "LinearReduce expects a matrix with %d columns, got %d"
                % (self.linear_map.shape[1], inp.shape[0])
            )
        return self.linear_map @ inp

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
        assert (self.linear_map.shape[-1]) == original_len, (
            self.__class__.__name__
            + " expects a gram with %d columns" % self.linear_map.shape[1]
        )
        return self.linear_map.shape[-2]

    @classmethod
    def sum_from_unique(
        cls,
        input: Float[Array, "N ..."],
        mean: ScalarBool = True,
        axis: ScalarInt = None,
    ) -> Tuple[Float[Array, "K ..."], Int[Array, "K"], "LinearReduce"]:
        """Find unique vectors in `input` along `axis`, return the unique data points, their counts and a linear reduction that multiplies the (now unique) vectors by their counts.

        Args:
            input (Float[Array, "N M"]): Input array.
            mean (bool, optional): Average the values if True, sum them if False. Defaults to True.
            axis (int, optional): Axis to find unique vectors along. Defaults to None, in which case the flattened array is used.

        Returns:
            Tuple[Float[Array, "K M"], Float[Array, "K"], "LinearReduce"]: The unique rows, their counts and the linear reduction.

        Example:
            >>> import jax.numpy as np
            >>> from jaxrk.reduce.base import LinearReduce
            >>> input = np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6]])
            >>> unique, counts, reduction = LinearReduce.sum_from_unique(input, mean=False, axis=None)
            >>> print(f"{repr(unique)}\n{repr(counts)}\n{repr(reduction.linear_map)}")
            DeviceArray([1, 2, 3, 4, 5, 6], dtype=int32)
            DeviceArray([2, 2, 2, 1, 1, 1], dtype=int32)
            DeviceArray([[1., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.],
                         [0., 0., 0.],
                         [0., 0., 0.],
                         [0., 0., 0.]], dtype=float32)

            >>> unique, counts, reduction = LinearReduce.sum_from_unique(input, mean=True, axis=0)
            >>> print(f"{repr(unique)}\n{repr(counts)}\n{repr(reduction.linear_map)}")
            DeviceArray([[1, 2, 3],
                         [4, 5, 6]], dtype=int32)
            DeviceArray([2, 1], dtype=int32)
            DeviceArray([[0.5, 0.5, 0. ],
                         [0. , 0. , 1. ]], dtype=float32)

            >>> unique, counts, reduction = LinearReduce.sum_from_unique(input, mean=True, axis=1)
            >>> print(f"{repr(unique)}\n{repr(counts)}\n{repr(reduction.linear_map)}")
            DeviceArray([[1, 2, 3],
                         [1, 2, 3],
                         [4, 5, 6]], dtype=int32)
            DeviceArray([1, 1, 1], dtype=int32)
            DeviceArray([[1., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.]], dtype=float32)

        """
        un, inv_idx, cts = np.unique(
            input, return_inverse=True, return_counts=True, axis=axis
        )

        m = np.zeros((len(un), input.shape[0]))
        for col, row in enumerate(inv_idx):
            m = m.at[row, col].set(1.0 / cts[row].squeeze() if mean else 1.0)

        return un, cts, LinearReduce(m)


class LinearizableReduce(AbstractReduce):
    """Reduction that can be linearized."""

    def linearize(self, gram_shape: tuple, axis: int = 0) -> LinearReduce:
        """Linearize the reduction.

        Args:
            gram_shape (tuple): Shape of the gram matrix.
            axis (int, optional): Axis to apply reduction over. Defaults to 0.

        Returns:
            LinearReduce: The linearized reduction.
        """
        return LinearReduce(self.linmap(gram_shape, axis))

    @abstractmethod
    def linmap(self, gram_shape: tuple, axis: int = 0) -> Array:
        """Linear map equivalent to reduction.

        Args:
            gram_shape (tuple): The gram matrix shape.
            axis (int, optional): Axis to apply reduction over. Defaults to 0.
        """
        pass
