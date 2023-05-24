from dataclasses import dataclass
from abc import ABC, abstractmethod
from jaxtyping import Float, Array, Int
from beartype.typing import Union, Callable, List, TypeVar, Tuple
import jax.numpy as np
from gpjax.typing import ScalarInt, ScalarFloat, ScalarBool

ReduceOrArray = TypeVar("ReduceOrArray", "AbstractReduce", Float[Array, "N M"])

NumberOrArray = TypeVar("NumberOrArray", Float[Array, "N M"], ScalarInt, ScalarFloat)


def tile_view(inp: np.ndarray, reps: int) -> np.ndarray:
    """Tile a view of an array

    Args:
        inp (np.ndarray): Array to tile
        reps (int): Repetitions of the array

    Returns:
        np.ndarray: Tile view of the array

    Example:
        >>> tile_view(np.array([[1, 2], [3, 4]]), 2)
        DeviceArray([[1, 2],
                     [3, 4],
                     [1, 2],
                     [3, 4]], dtype=int32)
    """
    return np.broadcast_to(inp.ravel(), (reps, inp.size)).reshape(
        (reps * inp.shape[0], inp.shape[1])
    )


class AbstractReduce(ABC):
    """The abstract base class for reductions."""

    @abstractmethod
    def __matmul__(self, inp: Array) -> Array:
        """Reduce the first axis of the input matrix.

        Args:
            inp (Array): The array to reduce. Typically a gram matrix.

        Returns:
            Array: The array reduced along the first axis.
        """
        pass

    @abstractmethod
    def new_len(self, original_len: int) -> int:
        """Compute the new length of the array after reduction.

        Args:
            original_len (int): Original length of the array.

        Returns:
            int: Length of the array after reduction.
        """
        pass


@dataclass
class ChainedReduce(AbstractReduce):
    reductions: List[AbstractReduce] = None

    def __post_init__(self):
        # Add kernels to a list, flattening out instances of this class therein, as in GPFlow kernels.
        reductions_list: List[AbstractReduce] = []

        for r in self.reductions:
            if not isinstance(r, AbstractReduce):
                raise TypeError("can only combine Reduce instances")

            if isinstance(r, self.__class__):
                reductions_list.extend(r.reductions)
            else:
                reductions_list.append(r)

        self.reductions = reductions_list

    def __execute_chain(
        self, func: Callable, start_val: NumberOrArray
    ) -> NumberOrArray:
        carry = start_val
        for gr in self.reductions[::-1]:
            carry = func(gr, carry)
        return carry

    def new_len(self, original_len: int) -> int:
        """Return the final length of an array after applying a chain of reductions.

        Args:
            original_len (int): Original length of the array.

        Returns:
            int: Final length of the array after applying the reductions.
        """
        return self.__execute_chain(lambda x, carry: x.new_len(carry), original_len)

    def __matmul__(self, other: ReduceOrArray) -> ReduceOrArray:
        """Apply a list of reductions to an array.

        Args:
            inp (Array): Input array, typically a gram matrix.

        Returns:
            Array: Reduced array.
        """
        if isinstance(other, AbstractReduce):
            return ChainedReduce([self, other])
        if self.reductions is None or len(self.reductions) == 0:
            return other
        else:
            return self.__execute_chain(lambda x, carry: x.__matmul__(carry), other)


@dataclass
class LinearReduce(AbstractReduce):
    """Reduction defined by a linear map."""

    linear_map: Float[Array, "N M"]

    def __matmul__(self, inp: ReduceOrArray) -> ReduceOrArray:
        """Reduce the first axis of the input matrix.

        Args:
            inp (ReduceOrArray): The array to reduce. Typically a gram matrix.

        Returns:
            Array: The array reduced along the first axis.
        """
        if isinstance(inp, AbstractReduce):
            return ChainedReduce([self, inp])
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


class NoReduce(AbstractReduce):
    """No reduction is actually applied."""

    def __matmul__(self, inp: Union[AbstractReduce, Array]):
        """Return the input array.

        Args:
            inp (Array): Input array, typically a gram matrix.

        Returns:
            Array: `inp`
        """
        if isinstance(inp, AbstractReduce):
            return ChainedReduce([self, inp])
        return inp

    def new_len(self, original_len: int) -> int:
        """Return the original length.

        Args:
            original_len (int): Original length of the array.

        Returns:
            int: `original_len`
        """
        return original_len


@dataclass
class Prefactors(AbstractReduce):
    """Multiply the input array with a set of prefactors"""

    prefactors: Float[Array, "N"]

    def __matmul__(self, inp: ReduceOrArray) -> ReduceOrArray:
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
        if isinstance(inp, AbstractReduce):
            return ChainedReduce([self, inp])
        if self.prefactors.shape[0] != inp.shape[axis]:
            raise ValueError(
                f"Prefactors shape {self.prefactors.shape} does not match input shape {inp.shape} along axis {axis}"
            )
        return inp.astype(self.prefactors.dtype) * np.expand_dims(
            self.prefactors, axis=(axis + 1) % 2
        )

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


@dataclass
class Scale(AbstractReduce):
    """Scale the input array by a constant factor."""

    s: ScalarFloat

    def __matmul__(self, inp: ReduceOrArray) -> ReduceOrArray:
        """Scale the input array.

        Args:
            inp (ReduceOrArray): Input array, typically a gram matrix.


        Returns:
            ReduceOrArray: Scaled input array.
        """
        if isinstance(inp, AbstractReduce):
            return ChainedReduce([self, inp])
        return inp * self.s

    def new_len(self, original_len: int) -> int:
        """Compute the new length of the array after reduction.

        Args:
            original_len (int): Original length of the array.

        Returns:
            int: Length of the array after reduction, i.e. `original_len`.
        """
        return original_len


@dataclass
class Repeat(AbstractReduce):
    """Repeat the input array."""

    times: int

    def __post_init__(self, times: int):
        """This reduction will repeat the input array `times` times.

        Args:
            times (int): Number of times to repeat the input array.
        """
        if times <= 1:
            raise ValueError("Repeat times must be greater than 1")

    def __matmul__(self, inp: ReduceOrArray) -> ReduceOrArray:
        """Repeat the input array.

        Args:
            inp (Array): Input array, typically a gram matrix.

        Returns:
            Array: Repeated input array.
        """
        if isinstance(inp, AbstractReduce):
            return ChainedReduce([self, inp])
        return np.repeat(inp, 0)

    def new_len(self, original_len: int) -> int:
        """Compute the new length of the array after reduction.

        Args:
            original_len (int): Original length of the array.

        Returns:
            int: Length of the array after reduction.
        """
        return original_len * self.times


@dataclass
class TileView(LinearizableReduce):
    """Tile the input array. This reduction provides a view on the input array, avoiding data copy."""

    result_len: int

    def __post_init__(self):
        """This reduction will tile the input array `result_len` times."""
        if self.result_len < 2:
            raise ValueError("TileView result_len must be greater than 1")

    def __matmul__(self, inp: ReduceOrArray) -> ReduceOrArray:
        """Reduce the first axis of the input array by tiling it.

        Args:
            inp (ReduceOrArray): Input array, typically a gram matrix.

        Returns:
            ReduceOrArray: Reduced array.

        Example:
            >>> from jax import numpy as jnp
            >>> from jaxrk.reduce import TileView
            >>> t = TileView(6)
            >>> m = jnp.array([[1,2,3],[4,5,6]])
            >>> t(m)
            DeviceArray([[1, 2, 3],
                        [4, 5, 6],
                        [1, 2, 3],
                        [4, 5, 6],
                        [1, 2, 3],
                        [4, 5, 6]], dtype=int32)
            >>> t(m, axis=1)
            DeviceArray([[1, 2, 3, 1, 2, 3],
                        [4, 5, 6, 4, 5, 6]], dtype=int32)
        """
        if isinstance(inp, AbstractReduce):
            return ChainedReduce([self, inp])
        if self.result_len % inp.shape[0] != 0:
            raise ValueError(
                "Input can't be broadcasted to target length %d" % self.result_len
            )
        return tile_view(inp, self.result_len // inp.shape[0])

    def linmap(self, inp_shape: tuple, axis: int = 0) -> Array:
        """Linear map version of reduce_first_ax for the tile view reduction.

        Args:
            inp_shape (tuple): Shape of the input matrix.
            axis (int, optional): Axis to apply reduction over. Defaults to 0.

        Returns:
            Array: A linear operator that can be applied to the input matrix and get a tiled result.
        """
        return tile_view(np.eye(inp_shape[axis]), self.result_len // inp_shape[axis])

    def new_len(self, original_len: int) -> int:
        """Compute the new length of the array after reduction.

        Args:
            original_len (int): Original length of the array.

        Returns:
            int: Length of the array after reduction.
        """
        return self.result_len


class Sum(LinearizableReduce):
    """Sum the input array."""

    def __matmul__(self, inp: ReduceOrArray) -> ReduceOrArray:
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
        if isinstance(inp, AbstractReduce):
            return ChainedReduce([self, inp])
        return inp.sum(axis=0, keepdims=True)

    def new_len(self, original_len: int) -> int:
        """Compute the new length of the array after reduction.

        Args:
            original_len (int): Original length of the array.

        Returns:
            int: Length of the array after reduction.
        """
        return 1

    def linmap(self, gram_shape: tuple, axis: int = 0) -> Array:
        """Linear map version of `Sum` reduction.

        Args:
            gram_shape (tuple): Shape of the input matrix.
            axis (int, optional): Axis to apply reduction over. Defaults to 0.

        Returns:
            Array: A linear operator that can be applied to the input matrix to sum over `axis`.
        """
        return np.ones((1, gram_shape[axis]))


class Mean(LinearizableReduce):
    """Average the input array."""

    def __matmul__(self, inp: ReduceOrArray) -> ReduceOrArray:
        """Average the input array.

        Args:
            inp (Array): Input array, typically a gram matrix.

        Returns:
            Array: Reduced input array.
        """
        if isinstance(inp, AbstractReduce):
            return ChainedReduce([self, inp])
        return np.mean(inp, axis=0, keepdims=True)

    def new_len(self, original_len: int) -> int:
        """Compute the new length of the array after reduction.

        Args:
            original_len (int): Original length of the array.

        Returns:
            int: Length of the array after reduction.
        """
        return 1

    def linmap(self, gram_shape: tuple, axis: int = 0) -> Array:
        """Linear map version of mean reduction.

        Args:
            gram_shape (tuple): Shape of the input matrix.
            axis (int, optional): Axis to apply reduction over. Defaults to 0.

        Returns:
            Array: A linear operator that can be applied to the input matrix to average over `axis`.
        """
        return np.ones((1, gram_shape[axis])) / gram_shape[axis]


@dataclass
class BalancedRed(LinearizableReduce):
    """Balanced reduction of the input array. Sums up a fixed number of consecutive elements in the input array (and potentially divide by the number of elements)."""

    points_per_split: int
    average: bool = False

    def __post_init__(self, points_per_split: int, average=False):
        """Balanced reduction of the input array. Sums up a number of consecutive elements in the input.

        Args:
            points_per_split (int): Number of points per split, i.e. number of dimensions to sum up to a single result dimension.
            average (bool, optional): If True, average rather than sum up dimensions. Defaults to False.
        """
        super().__init__()
        if self.points_per_split < 2:
            raise ValueError("points_per_split must be at least 2")
        if self.average:
            self.red = np.mean
            self.factor = 1.0 / points_per_split
        else:
            self.red = np.sum
            self.factor = 1.0

    def __matmul__(self, inp: ReduceOrArray) -> ReduceOrArray:
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
        if isinstance(inp, AbstractReduce):
            return ChainedReduce([self, inp])
        rval = self.red(
            np.reshape(inp, (-1, self.points_per_split, inp.shape[-1])), axis=1
        )
        return rval

    def linmap(self, inp_shape: tuple, axis: int = 0) -> Array:
        """Linear map version of `BalancedRed` reduction.

        Args:
            inp_shape (tuple): Shape of the input matrix.
            axis (int, optional): Axis to apply reduction over. Defaults to 0.

        Returns:
            Array: A linear operator that can be applied to the input matrix and get the same result as the reduction.
        """
        new_len = self.new_len(inp_shape[axis])
        rval = np.zeros((new_len, inp_shape[axis]))
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


class Center(AbstractReduce):
    """Center the input array by subtracting the mean."""

    def __matmul__(self, inp: ReduceOrArray) -> ReduceOrArray:
        """Center input along axis.

        Args:
            inp (Array): Input array, typically a gram matrix.

        Returns:
            Array: Centered input array.
        """
        if isinstance(inp, AbstractReduce):
            return ChainedReduce([self, inp])
        return inp - np.mean(inp, 0, keepdims=True)

    def new_len(self, original_len: int) -> int:
        """Compute the new length of the array after reduction.

        Args:
            original_len (int): Original length of the array.

        Returns:
            int: Length of the array after reduction.
        """
        return original_len
