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

from jax import vmap
from jaxtyping import Float

from gpjax.kernels.computations.base import AbstractKernelComputation
from gpjax.typing import Array
import gpjax as gpx
import datasets as ds


class HfDenseKernelComputation(AbstractKernelComputation):
    r"""Dense kernel computation class. Operations with the kernel assume
    a dense gram matrix structure.
    """

    def cross_covariance(self, x: ds.Dataset, y: ds.Dataset) -> Float[Array, "N M"]:
        r"""Compute the cross-covariance matrix.

        For a given kernel, compute the NxM covariance matrix on a pair of input
        matrices of shape $`NxD`$ and $`MxD`$.

        Args:
            x (ds.Dataset): The input matrix.
            y (ds.Dataset): The input matrix.

        Returns
        -------
            Float[Array, "N M"]: The computed cross-covariance.
        """
        if isinstance(self.kernel, gpx.kernels.EncoderKernel):
            return self.kernel.enc(x).outer_inner(self.kernel.enc(y))
        return vmap(lambda x: vmap(lambda y: self.kernel(x, y))(y))(x)
