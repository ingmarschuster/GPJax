import jax.numpy as jnp
import gpjax as gpx
import gpjax.objectives as gpxobj
from gpjax.typing import ScalarFloat
from gpjax.base import param_field
from gpjax.dataset import Dataset
from tensorflow_probability.substrates.jax import bijectors as tfb
from dataclasses import dataclass


@dataclass
class KRRjax(gpx.Module):
    kernel: gpx.AbstractKernel = param_field(gpx.RBF())
    beta: ScalarFloat = param_field(1.0, bijector=tfb.Softplus())

    def predict(self, test_inputs, train_data: Dataset):
        Kno = self.kernel.cross_covariance(test_inputs, train_data.X)
        Koo = self.kernel.gram(train_data.X) + self.beta * jnp.eye(train_data.n)
        rval = Kno @ Koo.solve(train_data.y)
        return rval


class MSEObjective(gpxobj.AbstractObjective):
    def step(self, model: KRRjax, data: Dataset):
        return jnp.mean((model.predict(data.X, data) - data.y) ** 2)
