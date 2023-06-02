from typing import Union
from .base import AbstractKernel
from jaxtyping import (
    Float,
    Num,
    PyTree,
)
from gpjax.typing import (
    Array,
    ScalarFloat,
    ScalarInt,
)
import jax.numpy as jnp
from gpjax.base import (
    Module,
    param_field,
    static_field,
)
from beartype.typing import (
    Callable,
    List,
    Optional,
    Type,
    Union,
)

from .stationary.rbf import RBF

from dataclasses import dataclass

from tensorflow_probability.substrates.jax import bijectors as tfb
from gpjax.rkhs.reduce import Kmer


@dataclass
class LinearmapKernel(AbstractKernel):
    r"""A kernel that maps the inputs through a linear map and then applies a base kernel."""
    name: str = static_field("LinearmapKernel")
    base_kernel: AbstractKernel = param_field(RBF())
    linear_map: Float[Array, "... N"] = param_field(jnp.ones(1))

    def __call__(self, x1: Num[Array, "N D"], x2: Num[Array, "N D"], **kwargs):
        x1 = self.slice_input(x1)
        x2 = self.slice_input(x2)
        return self.base_kernel(self.linear_map @ x1, self.linear_map @ x2, **kwargs)


@dataclass
class FeatmapKernel(AbstractKernel):
    r"""A kernel that maps the inputs through a linear map and then applies a base kernel."""
    name: str = static_field("LinearmapKernel")
    base_kernel: AbstractKernel = param_field(RBF())
    # Todo: probably should make this a Module
    func: callable = lambda x: x
    features: List[str] = static_field([])

    def __call__(self, x1: PyTree, x2: PyTree, **kwargs):
        x1 = self.slice_input(x1)
        x2 = self.slice_input(x2)
        return self.base_kernel(self.func(x1), self.func(x2), **kwargs)


@dataclass
class AdapterKernel(AbstractKernel):
    name: str = static_field("AdaptorKernel")
    features: List[str] = static_field([])
    base_kernel: AbstractKernel = param_field(RBF())

    def __post_init__(self):
        try:
            self.base_kernel = self.base_kernel.replace_trainable(variance=False)
        except ValueError:
            pass

    def __call__(self, x: PyTree, y: PyTree) -> ScalarFloat:
        if len(self.feature) == 1:
            x = x[self.feature[0]]
            y = y[self.feature[0]]
        self.base_kernel(x, y)


@dataclass
class ConvexcombinationKernel(AbstractKernel):
    r"""A kernel that is a convex combination of other kernels."""
    name: str = static_field("ConvexcombinationKernel")
    variance: ScalarFloat = param_field(jnp.array(1.0), bijector=tfb.Softplus())
    kernels: List[AdapterKernel] = None

    weights: Float[Array, "N"] = param_field(
        jnp.ones(1), bijector=tfb.SoftmaxCentered()
    )

    def __post_init__(self):
        self.features = []
        for i in self.kernels:
            self.features.extend(i.features)
        if self.weights.size != len(self.kernels):
            raise ValueError(
                f"Number of weights ({self.weights.size}) must match number of kernels ({len(self.kernels)})"
            )

    def __call__(self, x1: PyTree, x2: PyTree, **kwargs):
        rval = []
        for w, k in zip(self.weights, self.idcs):
            rval.append(w * self.kernels[k](x1, x2, **kwargs))
        return sum(rval) * self.variance


@dataclass
class Kmer1HotKernel(AbstractKernel):
    r"""A kernel that maps the inputs through a linear map and then applies a base kernel."""
    name: str = static_field("Kmer1hot")
    base_kernel: AbstractKernel = param_field(RBF())
    # Todo: probably should make this a Module
    features: List[str] = static_field(["aa_1hot"])
    lmap: Float[Array, "..."] = param_field(jnp.ones(1), trainable=False)
    # max_seq_len: ScalarInt = static_field(100)

    def __call__(self, x1: PyTree, x2: PyTree, **kwargs):
        return self.base_kernel(
            self.lmap @ x1[self.features[0]].reshape(self.lmap.shape[1], -1),
            self.lmap @ x2[self.features[0]].reshape(self.lmap.shape[1], -1),
            **kwargs,
        )
