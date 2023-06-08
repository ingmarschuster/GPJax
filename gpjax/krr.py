import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, EsmModel
import torch
from datasets import load_dataset
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from scipy.sparse import csr_matrix
import networkx as nx
import copy 
from numpy.linalg import matrix_power
from sklearn.metrics.pairwise import euclidean_distances
import math 
import random
from sklearn.cluster import KMeans
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import jax
import jax.numpy as jnp
import pedata as ped
import datasets as ds
import gpjax as gpx
import gpjax.objectives as gpxobj
from gpjax.typing import ScalarFloat
from gpjax.base import param_field, static_field
from gpjax.dataset import Dataset
from tensorflow_probability.substrates.jax import bijectors as tfb
from dataclasses import dataclass
import optax as ox
import gpjax.kernels as gk

import jax.random as jr
import eep


@dataclass
class KRRjax(gpx.Module):
    kernel: gpx.AbstractKernel = param_field(gpx.RBF())
    beta: ScalarFloat = param_field(1., bijector=tfb.Softplus())

    def predict(self, test_inputs, train_data: Dataset):
        Kno = self.kernel.cross_covariance(test_inputs, train_data.X)
        Koo = self.kernel.gram(train_data.X) + self.beta * jnp.eye(train_data.n)
        rval =  Kno @ Koo.solve(train_data.y)
        return rval


class MSEObjective(gpxobj.AbstractObjective):
    def step(self, model:KRRjax, data:Dataset):
        return jnp.mean((model.predict(data.X, data) - data.y)**2)
    