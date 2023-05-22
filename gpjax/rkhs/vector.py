import abc
from dataclasses import dataclass
from functools import partial

from beartype.typing import (
    Callable,
    List,
    Optional,
    Type,
    Union,
)
import jax.numpy as jnp
from jaxtyping import Float, Num, Array
import tensorflow_probability.substrates.jax.distributions as tfd

from gpjax.base import (
    Module,
    param_field,
    static_field,
)

import reduce as red
from .. import kernels


@dataclass
class RkhsVec(Module):
    insp_pts: Union["RkhsVec", Float[Array, "N M"]]
    kernel: kernels.AbstractKernel = param_field(kernels.RBF())
    reduce: red.Reduce = param_field(red.Identity())
    transpose: bool = False

    @property
    def is_colvec(self) -> bool:
        return not self.transpose

    @property
    def is_rowvec(self) -> bool:
        return self.transpose

    @property
    def shape(self):
        if self.transpose:
            return (1, self.size)
        return (self.size, 1)

    @property
    def size(self):
        return self.reduce.final_len(len(self.insp_pts))

    def __len__(self):
        if self.transpose:
            return 1
        else:
            return self.size

    def __pairwise_dot__(self, other: "RkhsVec") -> Float[Array]:
        """Compute the dot product between all pairs of elements from two RKHS vectors.

        Args:
            other (RkhsVec): The other RKHS vector. Assumed to have the same kernel.

        Raises:
            TypeError: If the kernels of the two RKHS vectors do not match.

        Returns:
            Float[Array]: A matrix of shape (self.size, other.size) containing the dot products.
        """
        if self.kernel != other.kernel:
            raise TypeError(
                f"Trying to compute inner products between elements of different RKHSs (Kernel types do not match)"
            )
        raw_gram = self.kernel.cross_covariance(self.insp_pts, other.insp_pts)
        return self.reduce @ (other.reduce @ raw_gram.T).T

    def __tensor_prod__(self, other: "RkhsVec") -> "RkhsVec":
        if self.size != other.size:
            raise ValueError(
                f"Trying to compute tensor product between RKHS vectors of different sizes ({self.size} and {other.size})"
            )
        return ProductVec([self, other], red.Sum())

    def sum(
        self,
    ) -> "RkhsVec":
        return red.Sum() @ self

    def mean(
        self,
    ) -> "RkhsVec":
        return red.Mean() @ self

    def __matmul__(self, other: "RkhsVec") -> Union[Float[Array], "RkhsVec"]:
        if self.is_rowvec == other.is_rowvec:
            raise ValueError(
                f"Trying to compute matrix product between two row vectors or two column vectors"
            )
        if self.is_rowvec and other.is_colvec:
            return self.__pairwise_dot__(other)
        else:
            return self.__tensor_prod__(other)

    def __rmatmul__(
        self, other: Union[red.AbstractReduce, "RkhsVec"]
    ) -> Union[Float[Array], "RkhsVec"]:
        if isinstance(other, red.AbstractReduce):
            return RkhsVec(
                self.insp_pts, self.kernel, other @ self.reduce, self.transpose
            )
        else:
            return self.__matmul__(other)
