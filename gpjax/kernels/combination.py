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
)
import jax.numpy as jnp
from gpjax.base import (
    Module,
    param_field,
    static_field,
)
from .stationary.rbf import RBF

from dataclasses import dataclass

from tensorflow_probability.substrates.jax import bijectors as tfb


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

    def __call__(self, x1: PyTree, x2: PyTree, **kwargs):
        x1 = self.slice_input(x1)
        x2 = self.slice_input(x2)
        return self.base_kernel(self.func(x1), self.func(x2), **kwargs)


@dataclass
class ConvexcombinationKernel(AbstractKernel):
    r"""A kernel that is a convex combination of other kernels."""
    name: str = static_field("ConvexcombinationKernel")
    kernels: Union[dict[Union[str, tuple[str]]], list[AbstractKernel]] = param_field(
        [RBF()]
    )
    weights: Float[Array, "N"] = param_field(
        jnp.ones(1), bijector=tfb.SoftmaxCentered()
    )

    def __post_init__(self):
        self.features = []
        if isinstance(self.kernels, dict):
            self.idcs = self.kernels
        else:
            self.idcs = range(len(self.kernels))
        for i in self.idcs:
            if isinstance(self.kernels, dict):
                if isinstance(i, str):
                    self.features.append(i)
                else:
                    self.features.extend(i)
            try:
                self.kernels[i] = self.kernels[i].replace_trainable(variance=False)
            except ValueError:
                pass
        if self.weights.size != len(self.kernels):
            raise ValueError(
                f"Number of weights ({self.weights.size}) must match number of kernels ({len(self.kernels)})"
            )

    def __call__(self, x1: PyTree, x2: PyTree, **kwargs):
        rval = []
        for w, k in zip(self.weights, self.idcs):
            if isinstance(self.kernels, dict):
                if isinstance(k, tuple):
                    # We assume self.kernels[k] can handle dictionaries as inputs
                    rval.append(w * self.kernels[k](x1, x2, **kwargs))
                else:
                    # we assume self.kernels[k] is a base kernel that cannot handle dictionaries as inputs
                    rval.append(w * self.kernels[k](x1[k], x2[k], **kwargs))
            else:
                # we assume self.kernels[k] is a base kernel
                rval.append(w * self.kernels[k](x1, x2, **kwargs))
        assert False
        return rval
