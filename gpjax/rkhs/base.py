import abc
from abc import (
    ABC,
    abstractmethod,
)
from copy import copy
from dataclasses import dataclass
from functools import partial
from typing import Any

from beartype.typing import (
    Callable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)
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


ReduceableOrArray = TypeVar(
    "ReduceableOrArray", "AbstractReduceable", Float[Array, "N ..."]
)

NumberOrArray = TypeVar(
    "NumberOrArray", Int[Array, "N ..."], Float[Array, "N ..."], ScalarInt, ScalarFloat
)


class AbstractReduceable(Module, ABC):
    """The abstract base class for reducable objects."""

    @abstractmethod
    def __apply_reduce__(self, reduce: "AbstractReduce") -> "AbstractReduceable":
        """Apply a reduction to the `self` object.

        Args:
            reduce (AbstractReduce): The reduction to apply.

        Returns:
            Any: The result of applying the reduction.
        """
        pass


class AbstractReduce(AbstractReduceable):
    """The abstract base class for reductions."""

    def __matmul__(self, inp: ReduceableOrArray) -> ReduceableOrArray:
        """Reduce the first axis of the input matrix.

        Args:
            inp (ReduceOrArray): The array to reduce. Typically a gram matrix.

        Returns:
            ReduceOrArray: The array reduced along the first axis.
        """
        if isinstance(inp, AbstractReduceable):
            return inp.__apply_reduce__(self)
        return self.__reduce_array__(inp)

    @abstractmethod
    def __reduce_array__(self, inp: Float[Array, "N ..."]) -> Float[Array, "M ..."]:
        """Reduce the first axis of the input array.

        Args:
            inp (Float[Array, "N ..."]): The array to reduce.

        Returns:
            Float[Array, "M ..."]: The array reduced along the first axis.
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


class LinearizableReduce(AbstractReduce):
    """Reduction that can be linearized."""

    def linearize(self, gram_shape: tuple) -> "LinearReduce":
        """Linearize the reduction.

        Args:
            gram_shape (tuple): Shape of the gram matrix.

        Returns:
            LinearReduce: The linearized reduction.
        """
        return LinearReduce(self.linmap(gram_shape))

    @abstractmethod
    def linmap(self, gram_shape: tuple) -> Array:
        """Linear map equivalent to reduction.

        Args:
            gram_shape (tuple): The gram matrix shape.
        """
        pass


@dataclass
class LinearReduce(AbstractReduce):
    """Reduction defined by a linear map."""

    linear_map: Float[Array, "N M"] = param_field(np.ones((1, 1)))

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

    def linmap(self, gram_shape: tuple) -> Array:
        """Linear map equivalent to reduction.

        Args:
            gram_shape (tuple): The gram matrix shape.
        """
        return self.linear_map

    def linearize(self, gram_shape: tuple) -> "LinearReduce":
        """Linearize the reduction.

        Args:
            gram_shape (tuple): Shape of the gram matrix.

        Returns:
            LinearReduce: The linearized reduction.
        """
        return self

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


class NoReduce(LinearizableReduce):
    """No reduction is actually applied."""

    def __reduce_array__(self, inp: Float[Array, "N ..."]) -> Float[Array, "M ..."]:
        """Return the input array.

        Args:
            inp (Array): Input array, typically a gram matrix.

        Returns:
            Array: `inp`
        """
        return inp

    def __apply_reduce__(self, other: AbstractReduce) -> AbstractReduce:
        """Apply `other` to `self`. Since `self` is the identity, `other` is returned.

        Args:
            inp (AbstractReduce): Input.

        Returns:
            AbstractReduce: `inp`
        """
        return other

    def new_len(self, original_len: int) -> int:
        """Return the original length.

        Args:
            original_len (int): Original length of the array.

        Returns:
            int: `original_len`
        """
        return original_len

    def linmap(self, gram_shape: tuple) -> Array:
        """Return the identity matrix.

        Args:
            gram_shape (tuple): The gram matrix shape.

        Returns:
            Array: Identity matrix.
        """
        return np.eye(gram_shape[0])


@dataclass
class ChainedReduce(LinearizableReduce):
    chain: List[AbstractReduce] = param_field([NoReduce()])

    def __post_init__(self):
        # Add to a list, flattening out instances of this class therein
        # FIXME: Maybe unnecessary given the implementation of __apply_reduce__?
        reductions_list: List[AbstractReduce] = []

        for r in self.chain:
            if not isinstance(r, AbstractReduce):
                raise TypeError("Can only combine Reduce instances")
            if isinstance(r, NoReduce):
                continue
            if isinstance(r, self.__class__):
                reductions_list.extend(r.chain)
            else:
                reductions_list.append(r)
        if len(reductions_list) == 0:
            self.chain = [NoReduce()]
        else:
            self.chain = reductions_list

    def __getitem__(self, idx: int) -> AbstractReduce:
        return self.chain[idx]

    # FIXME: The return type of this function needs to be specified, but currently I don't know how to and not get BearType errors.
    def __execute_chain(self, func: Callable, start_val: NumberOrArray) -> Any:
        carry = start_val
        # We assume that reductions are applied in reverse order.
        # E.g. reduce1 @ reduce2 @ reduce3 @ gram results in the list
        #  self.reductions = [reduce1, reduce2, reduce3]
        # so to reflect the correct math, we need to apply the reductions in reverse order.
        for gr in self.chain[::-1]:
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

    def linmap(self, gram_shape: tuple) -> Array:
        """Return the linear map of a chain of reductions.

        Args:
            gram_shape (tuple): The gram matrix shape.

        Returns:
            Array: Linear map of the chain of reductions.
        """
        return self.__execute_chain(
            lambda x, carry: x.linmap(carry.shape) @ carry, np.eye(gram_shape[0])
        )

    def __reduce_array__(self, inp: Float[Array, "N ..."]) -> Float[Array, "M ..."]:
        """Apply a list of reductions to an array.

        Args:
            inp (Array): Input array, typically a gram matrix.

        Returns:
            Array: Reduced array.
        """
        if self.chain is None or len(self.chain) == 0:
            return inp
        else:
            return self.__execute_chain(lambda x, carry: x.__matmul__(carry), inp)

    def __apply_reduce__(self, other: AbstractReduce) -> "ChainedReduce":
        """Apply the reduction `other` to the chain of reductions. Typically, `other` is prepended to the chain.

        Args:
            other (AbstractReduce): The reduction to combine with.

        Returns:
            ChainedReduce: The combined reduction.
        """
        if isinstance(other, NoReduce):
            return self
        elif isinstance(other, self.__class__):
            # makes sure that the reductions are flattened out
            return ChainedReduce(other.chain + self.chain)
        else:
            return ChainedReduce([other] + self.chain)


@dataclass
class AbstractRkhsVec(AbstractReduceable):
    """The abstract base class for RKHS vectors."""

    reduce: ChainedReduce = param_field(ChainedReduce([NoReduce()]))

    @property
    def is_colvec(self) -> bool:
        return not self.is_rowvec

    def __len__(self):
        if self.is_rowvec:
            return 1
        else:
            return self.size

    def inner_outer(self, other: "AbstractRkhsVec") -> "AbstractRkhsVec":
        """Compute the RKHS vector with one element according to the formula
        Σ_i self[i] x other[i]
        where x is the tensor product. The RKHS of the result is induced by the kernel k((x1, x2), (y1, y2)) = self.k(x1, y1) * other.k(x2, y2), i.e. it is the direct product of the two RKHSs.

        In other words, elements to pair are selected using an inner product like mechanism (pair elements that have the same position in self and other), while taking the tensor product to combine paired elements.

        Args:
            other (AbstractRkhsVec): The other RKHS vector.

        Raises:
            ValueError: If the two RKHS vectors have different sizes.

        Returns:
            AbstractRkhsVec: The RKHS vector with one element in the direct product space.
        """
        if self.size != other.size:
            raise ValueError(
                f"Trying to compute tensor product between RKHS vectors of different sizes ({self.size} and {other.size})"
            )
        return red.Sum() @ ProductVec([self, other])

    def sum(
        self,
    ) -> "AbstractRkhsVec":
        # FIXME: derived classes should return their own type
        # potential fix is to use a protocol or a metaclass or a decorator adding the method
        return red.Sum() @ self

    def mean(
        self,
    ) -> "AbstractRkhsVec":
        return red.Mean() @ self

    def __matmul__(
        self, other: "AbstractRkhsVec"
    ) -> Union[Float[Array, "M N"], "AbstractRkhsVec"]:
        if self.is_rowvec == other.is_rowvec:
            raise ValueError(
                f"Trying to compute matrix product between two row vectors or two column vectors"
            )
        if self.is_colvec and other.is_rowvec:
            # this returns a real matrix with the RKHS inner products between all possible combinations of elements from self and other
            return self.outer_inner(other)
        elif self.is_rowvec and other.is_colvec:
            # this returns a RKHS vector the one element Σ_i self[i] x other[i] (x being the tensor product)
            return self.inner_outer(other)
        else:
            raise ValueError(
                f"Trying to compute matrix product between two RKHS vectors of the same ({self.shape}). This is not supported."
            )

    def __rmatmul__(
        self, other: Union[AbstractReduce, "AbstractRkhsVec"]
    ) -> Union[Float[Array, "M N"], "AbstractRkhsVec"]:
        if isinstance(other, AbstractReduce):
            self.__apply_reduce__(other)
        else:
            return self.__matmul__(other)

    def __add__(self, other: "AbstractRkhsVec") -> "CombinationVec":
        return SumVec([self, other])

    def __mul__(self, other: "AbstractRkhsVec") -> "CombinationVec":
        return ProductVec([self, other])

    # Todo: __getitem__ methods, including slicing, indexing, and boolean indexing.

    @abc.abstractproperty
    def T(self):
        pass

    @abc.abstractproperty
    def is_rowvec(self) -> bool:
        pass

    @abc.abstractproperty
    def shape(self):
        pass

    @abc.abstractproperty
    def size(self):
        pass

    @abc.abstractmethod
    def outer_inner(self, other: "AbstractRkhsVec") -> Float[Array, "N M"]:
        r"""Compute the matrix resulting from taking the RKHS inner product of each element in self with each element in other.
        The result will be a matrix with element <self[i], other[j]> at position (i, j).
        This is exactly the gram matrix if no reduction is applied.

        In other words, elements to pair are selected using an outer product like mechanism (pair each element in self with each element in other), while taking the RKHS inner product to combine paired elements.

        Args:
            other (AbstractRkhsVec): The other RKHS vector. Assumed to have the same kernel.

        Raises:
            TypeError: If the kernels of the two RKHS vectors do not match.

        Returns:
            Float[Array]: A matrix of shape (self.size, other.size) containing the dot products.
        """
        pass

    @abc.abstractmethod
    def __apply_reduce__(self, reduce: AbstractReduce) -> "RkhsVec":
        pass


class AbstractEncoder(abc.ABC, Module):
    """Abstract class for RKHS vector encoders. These are used to encode input space data into RKHS vectors."""

    @abc.abstractmethod
    def __call__(self, inp: Any) -> AbstractRkhsVec:
        """Encodes input data into an RKHS vector.
        Args:
            inp (Any): Input data set.
        Returns:
            FiniteVec: RKHS vector.
        """
        pass
