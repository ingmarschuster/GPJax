# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Geometric Kernels
#
# [Geometric kernels](https://github.com/GPflow/GeometricKernels) is a Python package that provides functionality for defining kernel functions on Riemannian manifolds, graphs and meshes. In this notebook, we'll outline how Geometric kernels can be integrated with GPJax.

# %%
import gpjax as gpx
import jax.numpy as jnp
import jax.random as jr
from jax.config import config
from jaxutils import Dataset
from geometric_kernels.kernels import MaternKarhunenLoeveKernel
from geometric_kernels.spaces import Mesh
import jax
import meshzoo

config.update("jax_enable_x64", True)
key = jr.PRNGKey(123)

# %% [markdown]
# ## Kernel Structure
#
# To allow GPJax to interact with Geometric Kernels, we'll need to define a wrapper object that connects the two libraries. This can be achieved using the `GPJaxGeometricKernel` wrapper that is implemented within Geometric Kernels.

# %%
from geometric_kernels.frontends.jax.gpjax import GPJaxGeometricKernel

# %% [markdown]
# Within the above code, a `GeometricComputation` object is first defined that contains the logic required to compute a kernel matrix. This can then be supplied as the `compute_engine` in the `GPJaxGeometricKernel` object. Along with a kernel from Geometric Kernels which we name `base_kernel`, this initialises the kernel. To make the object valid in GPJax, a `__call__` methods which accepts a set of parameters and a pair of arrays for which we'd like to compute a kernel is defined. As Geometric kernels has already defined the kernel functions, all that is required here is to invoke the relevant method from the base kernel. Finally, we must tell GPJax how the object should be initialised. Again, this logic is contained within Geometric kernels, however, to the parameters' type must be coerced into a JAX array before returning.
#
# ## Defining a Geometric Kernel, simulating data
#
# We will consider modelling data on an icosa sphere from the [MeshZoo](https://github.com/meshpro/meshzoo) library. We begin by defining a Matérn kernel on the sphere.

# %%
from gpjax.config import add_parameter, Softplus

# (1) Define icosa sphere
resolution = 40
vertices, faces = meshzoo.icosa_sphere(resolution)
mesh = Mesh(vertices, faces)

# (2) Define geometric kernel in GeometricKernel
truncation_level = 20
base_kernel = MaternKarhunenLoeveKernel(mesh, truncation_level)

# (3) Define geometric kernel in GPJax, using the `GPJaxGeometricKernel` wrapper
geometric_kernel = GPJaxGeometricKernel(base_kernel)

# (4) Add hyperparamter transformations in GPJax
add_parameter("nu", Softplus)

# %% [markdown]
# A single draw is taken from the Gaussian process' prior distribution at a random set of points to give us a response variable.

# %%
prior = gpx.Prior(kernel=geometric_kernel)
params = prior._initialise_params(key)

n = 25
x = jr.randint(key, minval=0, maxval=mesh.num_vertices, shape=(n, 1))
y = prior(params)(x).sample(seed=key)
D = Dataset(X=x, y=y)

x_test = jnp.arange(mesh.num_vertices).reshape(mesh.num_vertices, 1)

# %% [markdown]
# ## Model specification
#
# A model can now be defined. We'll purposefully keep this section brief as the workflow is identical to that of a regular Gaussian process regression workflow that is detailed in [full](https://gpjax.readthedocs.io/en/latest/examples/regression.html).

# %%
likelihood = gpx.likelihoods.Gaussian(num_datapoints=n)
posterior = likelihood * prior

# %% [markdown]
# As with a regular conjugate Gaussian process, the marginal log-likelihood is tractable and can be evaluated using the posterior's `marginal_log_likelihood` method.

# %%
params = posterior._initialise_params(key)
mll = posterior.marginal_log_likelihood(D)(params)

# %% [markdown]
# Derivatives of the marginal log-likelihood can be taken.

# %%
grads = jax.grad(mll)(params)
print(grads)

# %% [markdown]
# Finally, the predictive posterior can be computed for making predictions at unseen points. Evaluating the predictive posterior distribution returns a multivariate Gaussian distribution for which we can compute the posterior mean and variance as follows.

# %%
predictive_posterior = posterior.predict(params, D)(x_test)
mean = predictive_posterior.mean()
variance = predictive_posterior.variance()

# %%
# %reload_ext watermark
# %watermark -n -u -v -iv -w -a 'Thomas Pinder'
