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

from dataclasses import is_dataclass
from itertools import product
from typing import List

import jax
from jax.config import config
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb

from gpjax.kernels.base import AbstractKernel
from gpjax.kernels.computations import DenseKernelComputation
from gpjax.kernels.encoderbased import EncoderKernel
from gpjax.linops import LinearOperator
import gpjax as gpx
import datasets as ds

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)
_initialise_key = jr.PRNGKey(123)
_jitter = 1e-6


def test_encoderkernel() -> None:
    inner_kernel = gpx.kernels.RBF()
    k = EncoderKernel(enc=gpx.rkhs.encoder.StandardEncoder(inner_kernel))
    x = jnp.linspace(0, 1, 10)[:, None]
    y = jnp.linspace(4, 6, 10)[:, None]
    assert jnp.allclose(k.gram(x).to_dense(), inner_kernel.gram(x).to_dense())
    assert jnp.allclose(k.cross_covariance(x, y), inner_kernel.cross_covariance(x, y))


def test_encoderkernelhf() -> None:
    inner_kernel = gpx.kernels.RBF()
    k = EncoderKernel(
        enc=gpx.rkhs.encoder.HfStandardEncoder(inner_kernel),
    )
    x = jnp.linspace(0, 1, 10)[:, None]
    y = jnp.linspace(4, 6, 10)[:, None]
    x_hf = ds.Dataset.from_dict({"x": x})
    y_hf = ds.Dataset.from_dict({"x": y})
    assert jnp.allclose(k.gram(x_hf).to_dense(), inner_kernel.gram(x).to_dense())
    assert jnp.allclose(
        k.cross_covariance(x_hf, y_hf), inner_kernel.cross_covariance(x, y)
    )
