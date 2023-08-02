# Copyright 2022 The JaxGaussianProcesses Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from dataclasses import dataclass
from typing import NamedTuple
import jax.numpy as jnp
from jaxtyping import (
    Float,
    Int,
    Num,
)
import tensorflow_probability.substrates.jax as tfp

from gpjax.base import (
    param_field,
    static_field,
)
from gpjax.kernels.base import AbstractKernel
from gpjax.kernels.computations import (
    AbstractKernelComputation,
    EigenKernelComputation,
)
from gpjax.kernels.non_euclidean.utils import jax_gather_nd
from gpjax.typing import (
    Array,
    ScalarFloat,
    ScalarInt,
)

tfb = tfp.bijectors

DictKernelParams = NamedTuple(
    "DictKernelParams",
    [("sdev", Float[Array, "N 1"]), ("cholesky_lower", Float[Array, "N*(N-1)//2"])],
)


@dataclass
class DictKernel(AbstractKernel):
    r"""The Dictionary kernel is defined for a fixed number of values.

    It stores an explicit gram matrix and returns the corresponding values when called.

    Args:
        sdev (Float[Array, "N"]): The standard deviation parameters, one for each input space value.
        cholesky_lower (Float[Array, "N*(N-1)//2 N"]): The parameters for the Cholesky factor of the gram matrix.
        inspace_vals (list): The values in the input space this DictKernel works for. Stored for order reference, making clear the indices used for each input space value.

    Raises:
        ValueError: If the number of diagonal variance parameters does not match the number of input space values.
    """

    sdev: Float[Array, "N"] = param_field(jnp.ones((2,)), bijector=tfb.Softplus())
    cholesky_lower: Float[Array, "N N"] = param_field(
        jnp.eye(2), bijector=tfb.CorrelationCholesky()
    )
    inspace_vals: list = static_field(None)
    name: str = "Dictionary Kernel"

    def __post_init__(self):
        if self.inspace_vals is not None and len(self.inspace_vals) != len(self.sdev):
            raise ValueError(
                f"The number of sdev parameters ({len(self.sdev)}) has to match the number of input space values ({len(self.inspace_vals)}), unless inspace_vals is None."
            )
        L = self.sdev.reshape(-1, 1) * self.cholesky_lower

        self.explicit_gram = L @ L.T

    def __call__(  # TODO not consistent with general kernel interface
        self,
        x: ScalarInt,
        y: ScalarInt,
    ):
        r"""Compute the (co)variance between a pair of dictionary indices.

        Args:
            x (ScalarInt): The index of the first dictionary entry.
            y (ScalarInt): The index of the second dictionary entry.

        Returns
        -------
            ScalarFloat: The value of $k(v_i, v_j)$.
        """
        try:
            x = x.squeeze()
            y = y.squeeze()
        except AttributeError:
            pass
        return self.explicit_gram[x, y]

    @classmethod
    def num_cholesky_lower_params(cls, num_inspace_vals: ScalarInt) -> ScalarInt:
        """Compute the number of parameters required to store the lower triangular Cholesky factor of the gram matrix.

        Args:
            num_inspace_vals (ScalarInt): The number of values in the input space.

        Returns:
            ScalarInt: The number of parameters required to store the lower triangle of the Cholesky factor of the gram matrix.
        """
        return num_inspace_vals * (num_inspace_vals - 1) // 2

    @classmethod
    def gram_to_sdev_cholesky_lower(cls, gram: Float[Array, "N N"]) -> DictKernelParams:
        """Compute the standard deviation and lower triangular Cholesky factor of the gram matrix.

        Args:
            gram (Float[Array, "N N"]): The gram matrix.

        Returns:
            tuple[Float[Array, "N"], Float[Array, "N N"]]: The standard deviation and lower triangular Cholesky factor of the gram matrix, where the latter is scaled to result in unit variances.
        """
        sdev = jnp.sqrt(jnp.diag(gram))
        L = jnp.linalg.cholesky(gram) / sdev.reshape(-1, 1)
        return DictKernelParams(sdev, L)
