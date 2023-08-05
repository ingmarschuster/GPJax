import jax.numpy as jnp
import gpjax as gpx
import gpjax.objectives as gpxobj
from gpjax.typing import ScalarFloat
from gpjax.base import param_field, static_field
from gpjax.dataset import Dataset
from gpjax.rkhs.vector import RkhsVec
from tensorflow_probability.substrates.jax import bijectors as tfb
from dataclasses import dataclass


@dataclass
class KRRjax(gpx.Module):
    kernel: gpx.AbstractKernel = param_field(gpx.RBF())
    regul: ScalarFloat = param_field(1.0, bijector=tfb.Softplus())

    def predict(self, test_inputs, train_data: Dataset):
        Kno = self.kernel.cross_covariance(test_inputs, train_data.X)
        Koo = self.kernel.gram(train_data.X) + self.regul * jnp.eye(train_data.n)
        rval = Kno @ Koo.solve(train_data.y)
        return rval


@dataclass
class KRRind(gpx.Module):
    beta: ScalarFloat = param_field(1.0, bijector=tfb.Softplus())
    num_pos: int = static_field(600)
    k_obs: gpx.AbstractKernel = param_field(gpx.PoweredExponential())

    def predict(self, test_inputs):
        Kno = self.kernel.cross_covariance(test_inputs, train_data.X)
        Koo = self.kernel.gram(train_data.X) + self.beta * jnp.eye(train_data.n)
        rval = Kno @ Koo.solve(train_data.y)
        return rval


class MSEObjective(gpxobj.AbstractObjective):
    def step(self, model: KRRjax, data: Dataset):
        return jnp.mean((model.predict(data.X, data) - data.y) ** 2)
