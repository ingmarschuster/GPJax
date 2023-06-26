import collections.abc
from dataclasses import dataclass


from beartype.typing import (
    Callable,
    List,
    Tuple,
    TypeVar,
    Union,
)
from jax import jit
import jax.numpy as np

import jax.scipy as sp
import jax.scipy.stats as stats
from jaxtyping import Array, Float32, Int32, Float, Int

from gpjax.typing import (
    ScalarBool,
    ScalarInt,
    ScalarFloat,
)

from ..base import ReduceableOrArray, NumberOrArray, AbstractReduce

from .base import LinearizableReduce, LinearReduce, NoReduce, ChainedReduce

from gpjax.base import param_field, static_field


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


@dataclass
class Repeat(LinearizableReduce):
    """This reduction will repeat the input array `times` times.

    Args:
        times (int): Number of times to repeat the input array.
    """

    times: ScalarInt

    def __post_init__(
        self,
    ):
        if self.times <= 1:
            raise ValueError("Repeat times must be greater than 1")

    def __reduce_array__(self, inp: Float[Array, "N M"]) -> Float[Array, "K M"]:
        """Repeat the input array.

        Args:
            inp (Array): Input array, typically a gram matrix.

        Returns:
            Array: Repeated input array.
        """
        if isinstance(inp, AbstractReduce):
            return ChainedReduce([self, inp])
        return np.repeat(inp, self.times, 0)

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
        return original_len * self.times

    def linmap(self, gram_shape: tuple) -> Array:
        """Compute the linear map of the reduction.

        Args:
            gram_shape (tuple): Shape of the gram matrix.

        Returns:
            Array: Linear map of the reduction.
        """
        return np.repeat(np.eye(gram_shape[0]), self.times, 0)


@dataclass
class TileView(LinearizableReduce):
    """Tile the input array. This reduction provides a view on the input array, avoiding data copy."""

    result_len: int = static_field(None)
    tile_times: int = static_field(None)

    def __post_init__(self):
        """This reduction will tile the input array `result_len` times."""
        # check that either tile_times or result_len is set, but not both
        if (self.tile_times is None and self.result_len is None) or (
            self.tile_times is not None and self.result_len is not None
        ):
            raise ValueError("TileView must have either tile_times or result_len set")

        if self.result_len is not None and self.result_len < 2:
            raise ValueError("TileView result_len must be greater than 1")
        if self.tile_times is not None and self.tile_times < 2:
            raise ValueError("TileView tile_times must be greater than 1")

    def __get_repeats__(self, original_len: int) -> int:
        """Compute the number of times to repeat the input array.

        Args:
            original_len (int): Original length of the array.

        Returns:
            int: Number of times to repeat the input array.
        """
        if self.tile_times is not None:
            return self.tile_times
        if self.result_len % original_len != 0:
            raise ValueError(
                "Input can't be broadcasted to target length %d" % self.result_len
            )
        return self.result_len // original_len

    def __reduce_array__(self, inp: Float[Array, "N M"]) -> Float[Array, "K M"]:
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
        repeats = self.__get_repeats__(inp.shape[0])
        return tile_view(inp, repeats)

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
        """Linear map version of reduce_first_ax for the tile view reduction.

        Args:
            inp_shape (tuple): Shape of the input matrix.
            axis (int, optional): Axis to apply reduction over. Defaults to 0.

        Returns:
            Array: A linear operator that can be applied to the input matrix and get a tiled result.
        """
        return tile_view(np.eye(inp_shape[0]), self.__get_repeats__(inp_shape[0]))

    def new_len(self, original_len: int) -> int:
        """Compute the new length of the array after reduction.

        Args:
            original_len (int): Original length of the array.

        Returns:
            int: Length of the array after reduction.
        """
        if self.result_len is not None:
            return self.result_len
        else:
            return original_len * self.tile_times


@dataclass
class SparseReduce(LinearizableReduce):
    """SparseReduce constructs a Gram matrix by summing/averaging over rows of its input
    Args:
            idcs (List[np.array]): The indices of the rows to sum/average in the desired order. Each list element contains 2d arrays. The number of columns in the array is the number of summed/averaged elements. The number of rows is the number of rows in the output resulting from this list element.
            average (bool): If True average rows, else sum rows.
            max_idx (int, optional): The maximum index in the input. Defaults to None, in which case the maximum index is inferred from the idcs.
    """

    idcs: List[Int[Array, "N M"]]
    average: bool = static_field(True)
    max_idx: int = static_field(None)

    def __post_init__(
        self,
    ) -> None:
        if self.max_idx is None:
            max_list = []
            for i in self.idcs:
                if i.size > 0:
                    max_list.append(np.max(i))
            self.max_idx = int(np.array(max_list).max())
        if self.average:
            self._reduce = np.mean
        else:
            self._reduce = np.sum

    def __reduce_array__(self, inp: Float[Array, "N M"]) -> Float[Array, "K L"]:
        """Reduce the first axis of the input.

        Args:
            inp (np.array): Input to reduce. Typically a gram matrix.

        Returns:
            np.array: Reduced input.

        Examples:
            >>> import jax.numpy as np
            >>> from jaxrk.reduce import SparseReduce
            >>> inp = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            >>> r1 = SparseReduce([np.array([[0, 1]]), np.array([[0, 2]]), np.array([[0, 1, 2]])], True) # average rows 0 and 1, then 0 and 2, then all rows
            >>> r1 @ inp
            DeviceArray([[2.5, 3.5, 4.5],
                         [4. , 5. , 6. ],
                         [4. , 5. , 6. ]], dtype=float32)
            >>> r2 = SparseReduce([np.array([[0, 1], [0, 2]]), np.array([[0, 1, 2]])], True) # average rows 0 and 1, then 0 and 2, then all rows. Same as r1, just different input format, and probably more efficient.
            >>> r2 @ inp
            DeviceArray([[2.5, 3.5, 4.5],
                        [4. , 5. , 6. ],
                        [4. , 5. , 6. ]], dtype=float32)
            >>> r3 = SparseReduce([np.array([0, 0, 1, 1, 2])[:, np.newaxis]], False) # copy row 0 twice, then row 1 twice, then row 2
            >>> r3 @ inp
            DeviceArray([[ 1,  2,  3],
                            [ 1,  2,  3],
                            [ 4,  5,  6],
                            [ 4,  5,  6],
                            [ 7,  8,  9]], dtype=float32)
        """
        if (self.max_idx + 1) > len(inp):
            raise ValueError(
                self.__class__.__name__ + " expects a longer gram to operate on"
            )
        if len(inp.shape) != 2:
            raise ValueError(
                self.__class__.__name__ + " expects a 2d gram to operate on"
            )
        rval = []

        for i in range(len(self.idcs)):
            if self.idcs[i].shape[1] == 0:
                rval.append(np.zeros((self.idcs[i].shape[0], inp.shape[1])))
            else:
                reduced = self._reduce(
                    inp[list(self.idcs[i].flatten()), :].reshape(
                        (-1, self.idcs[i].shape[1], inp.shape[1])
                    ),
                    1,
                )
                rval.append(reduced)
        return np.concatenate(rval, 0)

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

    def new_len(self, original_len: ScalarInt) -> ScalarInt:
        """Get the length of the reduced gram matrix.

        Args:
            original_len (int): The length of the original gram matrix.

        Returns:
            int: The length of the reduced gram matrix.
        """
        assert (self.max_idx + 1) <= original_len, (
            self.__class__.__name__ + " expects a longer gram to operate on"
        )
        return len(self.idcs)

    def linmap(self, inp_shape: Tuple[ScalarInt, ScalarInt]) -> Float[Array, "N M"]:
        """Get the linear map that reduces the first axis of the input.

        Args:
            inp_shape (Tuple[ScalarInt, ScalarInt]): The shape of the input.

        Returns:
            Float[Array, "N M"]: The linear map that reduces the first axis of the input.

        Example:
            >>> import jax.numpy as np
            >>> from jaxrk.reduce.lincomb import SparseReduce
            >>> input = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
            >>> un, cts, sr = SparseReduce.sum_from_unique(input, mean=False)
            >>> print(sr.linmap((9, 1)))
            [[1. 0. 0. 1. 0. 0. 1. 0. 0.]
             [0. 1. 0. 0. 1. 0. 0. 1. 0.]
             [0. 0. 1. 0. 0. 1. 0. 0. 1.]]

        """
        n_in = self.max_idx + 1
        if inp_shape[0] != n_in:
            raise ValueError("Input shape does not match reduction assumptions")
        n_out = int(np.sum(np.array([len(i) for i in self.idcs])))
        offset = 0
        lin_map = np.zeros((n_out, n_in))
        for i in range(len(self.idcs)):
            if self.idcs[i].shape[1] != 0:
                idx1 = np.repeat(
                    np.arange(self.idcs[i].shape[0]) + offset, self.idcs[i].shape[1]
                )
                lin_map = lin_map.at[(idx1, self.idcs[i].flatten())].set(
                    1.0 / self.idcs[i].shape[1] if self.average else 1.0
                )
            offset += self.idcs[i].shape[0]
        return lin_map

    @classmethod
    def sum_from_block_example(cls, l: list[collections.abc.Sized], mean: bool = True):
        """Construct a SparseReduce object from an example list of arrays.
        The arrays in the list are assumed to be of the length of # of elements that should be reduced.

        Args:
            l (list[collections.abc.Sized]): Block example.
            mean (bool, optional): Whether to average the blocks. Defaults to True.
        """

        def collect_block_start_stop(l: list[np.ndarray]) -> np.ndarray:
            """Collect the start and stop indices of the arrays in the list.
            This can be used to reconstruct the original list after stacking its element arrays.

            Args:
                l (list[np.ndarray]): List of arrays, where each array can have different length.

            Returns:
                np.ndarray: Array of shape (len(l), 2) where each row contains the start and stop index of the corresponding array in the list.
            """
            # variable to collect indices of start and stop of
            # each block, e.g. if the elements in l have length 3, 4, 2
            # then this will be [[0, 3], [3, 7], [7, 9]]
            # i.e. the stop index is excluded
            rval = []
            total_len = 0
            for arr in l:
                arr_len = len(arr)
                rval.append((total_len, arr_len + total_len))
                total_len += arr_len
            return np.array(rval)

        def reduce_blocks(block_start_stop: np.ndarray) -> list[np.ndarray]:
            """Get the indices of the elements between block start and stop indices.

            Args:
                block_start_stop (np.ndarray): Array of shape (D, 2) where each row contains the start and stop index corresponding to the size of an array.

            Returns:
                list[np.ndarray]: List of arrays of indices of the elements between block start and stop indices.
            """
            rval = []
            total_len = block_start_stop[-1, 1]
            for start, stop in block_start_stop:
                rval.append(np.arange(start, stop, dtype=np.uint32)[np.newaxis, :])
            return rval

        blocks = collect_block_start_stop(l)
        return cls(reduce_blocks(blocks), average=mean, max_idx=blocks[-1, 1] - 1)

    @classmethod
    def sum_from_unique(
        cls, input: Float[Array, "N ..."], mean: bool = True
    ) -> Tuple[Float[Array, "K ..."], Int[Array, "K"], "SparseReduce"]:
        """Construct a SparseReduce object from a 1d array values by summing/averaging over the indices of the unique values.

        Args:
            input (Float[Array, "N M"]): The input array.
            mean (bool, optional): Average the values if True, sum them if False. Defaults to True.

        Returns:
            Tuple[Float[Array, "K L"], Float[Array, "K"], "SparseReduce"]: The unique values, the counts of the unique values, and the SparseReduce object.

        Example:
            >>> import jax.numpy as np
            >>> from jaxrk.reduce.lincomb import SparseReduce
            >>> input = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
            >>> un, cts, sr = SparseReduce.sum_from_unique(input)
            >>> print(un)
            [1 2 3]
            >>> print(cts)
            [3 3 3]
            >>> print(sr.idcs)
            [DeviceArray([[0, 3, 6],
                         [1, 4, 7],
                         [2, 5, 8]], dtype=int32)]
        """
        un, cts = np.unique(input, return_counts=True)
        un_idx = [np.argwhere(input == un[i]).flatten() for i in range(un.size)]
        l_arr = np.array([i.size for i in un_idx])
        argsort = np.argsort(l_arr)
        un_sorted = un[argsort]
        cts_sorted = cts[argsort]
        un_idx_sorted = [un_idx[i] for i in argsort]

        change = list(
            np.argwhere(l_arr[argsort][:-1] - l_arr[argsort][1:] != 0).flatten() + 1
        )
        change.insert(0, 0)
        change.append(len(l_arr))
        change = np.array(change)

        el = []
        for i in range(len(change) - 1):
            el.append(
                np.array([un_idx_sorted[j] for j in range(change[i], change[i + 1])])
            )

        # assert False
        return un_sorted, cts_sorted, SparseReduce(el, mean)


def MovingAverage(
    k: ScalarInt,
    seq_length: ScalarInt,
    average: ScalarBool,
    step: ScalarInt = 1,
    output_format: str = "linear",
) -> Union[np.ndarray, SparseReduce, LinearReduce]:
    """Generate a moving average/sum matrix for a sequence of length seq_length.
    Either return the binary matrix or a SparseReduce object.

    Args:
        k (ScalarInt): The window size.
        seq_length (ScalarInt): The length of the sequence. You can also specify a max sequence length and The output matrix will have shape (k, seq_length).
        average (ScalarBool): Whether to average the k elements for each k-mer.
        step (ScalarInt, optional): The step size. Defaults to 1.
        output_format (str, optional): Either "linear" or "sparse" to get a Reduce object. Using "matrix" will return the summing/averaging matrix. Using "index" will return the indices of the k-mers. Defaults to "linear".
    Returns:
        Union[np.ndarray, SparseReduce, LinearReduce]: The k-mer binary/index matrix or a Reduce object.
    """
    if seq_length < k or k <= 1 or seq_length <= 1 or step <= 0:
        raise ValueError(
            f"seq_length ({seq_length}) must be greater than k ({k}) and both must be greater than 1. Step ({step}) must be greater than 0."
        )

    num_windows = (seq_length - k) // step + 1
    idx = np.arange(k)[None, :] + step * np.arange(num_windows)[:, None]
    if output_format == "index":
        return idx
    elif output_format == "sparse":
        return SparseReduce([idx], average)
    A = np.zeros((num_windows, seq_length), dtype=np.float32)
    A = A.at[np.arange(num_windows)[:, None], idx].set(1)
    if average:
        A = A / k
    if output_format == "matrix":
        return A
    elif output_format == "linear":
        return LinearReduce(A)
    raise ValueError(
        f"output_format must be either 'linear', 'sparse', 'matrix', or 'index', but got {output_format}."
    )


def Kmer(
    k: int,
    seq_length: int,
    average: bool = True,
    output_format: str = "linear",
) -> Union[np.ndarray, LinearReduce]:
    """Generate a k-mer linear map A to be applied to a matrix B of shape (seq_length x embedding_dim).
    The shape of A @ B would then be (k x embedding_dim).

    Args:
        k (int): The k-mer length.
        seq_length (int): The length of the sequence.
        average (bool, optional): Whether to average the k elements for each k-mer. Defaults to True.
        output_format (str, optional): Either "linear" or "binary" to get a Reduce object. Using "matrix" will return the binary matrix. Using "index" will return the indices of the k-mers. Defaults to "linear".

    Returns:
        Union[np.ndarray, LinearReduce]: A k-mer matrix of shape (k, seq_length) or a LinearReduce with this matrix inside.
    """

    A = np.zeros((k, seq_length))
    for i in range(0, k):
        A = A.at[i, i : i + seq_length - k + 1].set(1)
    if average:
        A = A / k
    if output_format == "matrix":
        return A
    elif output_format == "linear":
        return LinearReduce(A)
    raise ValueError(
        f"output_format must be either 'linear' or 'matrix', but got {output_format}."
    )
