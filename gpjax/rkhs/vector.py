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

from copy import copy

from . import reduce as red
from .. import kernels

from .base import AbstractRkhsVec, AbstractReduce


@dataclass
class RkhsVec(AbstractRkhsVec):
    k: kernels.AbstractKernel = param_field(kernels.RBF())
    insp_pts: Union["RkhsVec", Float[Array, "N M ..."]] = param_field(jnp.ones((1, 1)))

    def __post_init__(self):
        self.transpose: bool = False

    @property
    def T(self):
        rval = self.reduce @ RkhsVec(self.insp_pts, self.k)
        rval.transpose = not self.transpose
        return rval

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
        return self.reduce.new_len(len(self.insp_pts))

    def outer_inner(self, other: "RkhsVec") -> Float[Array, "N M"]:
        r"""Compute the matrix resulting from taking the RKHS inner product of each element in self with each element in other.
        The result will be a matrix with element <self[i], other[j]> at position (i, j).
        This is exactly the gram matrix if no reduction is applied.

        In other words, elements to pair are selected using an outer product like mechanism (pair each element in self with each element in other), while taking the RKHS inner product to combine paired elements.

        Args:
            other (RkhsVec): The other RKHS vector. Assumed to have the same kernel.

        Raises:
            TypeError: If the kernels of the two RKHS vectors do not match.

        Returns:
            Float[Array]: A matrix of shape (self.size, other.size) containing the dot products.
        """
        if self.k != other.k:
            raise TypeError(
                f"Trying to compute inner products between elements of different RKHSs (Kernel types do not match)"
            )
        raw_gram = self.k.cross_covariance(self.insp_pts, other.insp_pts)
        return self.reduce @ (other.reduce @ raw_gram.T).T

    def __apply_reduce__(self, reduce: AbstractReduce) -> "RkhsVec":
        return RkhsVec(
            reduce=reduce @ copy(self.reduce), k=self.k, insp_pts=self.insp_pts
        )


@dataclass
class CombinationVec(AbstractRkhsVec):
    rkhs_vecs: List[RkhsVec] = param_field([])
    operator: Callable = static_field(None)

    def __post_init__(self):
        super().__post_init__()
        orig_size = self.rkhs_vecs[0].size
        for rkhs_vec in self.rkhs_vecs:
            if len(rkhs_vec) != orig_size:
                raise ValueError(
                    f"Trying to combine RKHS vectors of different sizes ({orig_size} and {len(rkhs_vec)})"
                )
        self.__size = self.reduce.new_len(orig_size)
        self.transpose: bool = False

    @property
    def T(self):
        rval = CombinationVec(self.rkhs_vecs, self.operator)
        rval.reduce = self.reduce
        rval.transpose = not self.transpose
        return rval

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
        return self.__size

    def outer_inner(self, other: "CombinationVec") -> Float[Array, "N M"]:
        r"""Compute the matrix resulting from taking the RKHS inner product of each element in self with each element in other.
        The result will be a matrix with element <self[i], other[j]> at position (i, j).
        This is exactly the gram matrix if no reduction is applied.

        In other words, elements to pair are selected using an outer product like mechanism (pair each element in self with each element in other), while taking the RKHS inner product to combine paired elements.

        Args:
            other (CombinationVec): The other RKHS vector. Assumed to have the same kernel.

        Raises:
            TypeError: If the kernels of the two RKHS vectors do not match.

        Returns:
            Float[Array]: A matrix of shape (self.size, other.size) containing the dot products.
        """
        if self.k != other.k:
            raise TypeError(
                f"Trying to compute inner products between elements of different RKHSs (Kernel types do not match)"
            )
        # compute the gram matrix for all pairs of elements in self and other
        raw_grams = jnp.array(
            [
                self.rkhs_vecs[i].outer_inner(other.rkhs_vecs[i])
                for i in range(len(self.rkhs_vecs))
            ]
        )

        # combine the gram matrices
        combined_raw_gram = self.operator(
            raw_grams,
            axis=0,
        )

        # reduce the combined gram matrix
        return self.reduce @ (other.reduce @ combined_raw_gram.T).T

    def __apply_reduce__(self, reduce: AbstractReduce) -> "RkhsVec":
        return CombinationVec(
            reduce=reduce @ copy(self.reduce),
            rkhs_vecs=self.rkhs_vecs,
            operator=self.operator,
        )


SumVec = partial(CombinationVec, operator=jnp.add)
ProductVec = partial(CombinationVec, operator=jnp.multiply)
