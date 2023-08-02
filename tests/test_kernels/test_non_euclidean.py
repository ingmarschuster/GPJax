# # Copyright 2022 The JaxGaussianProcesses Contributors. All Rights Reserved.
# #
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

from jax.config import config
import jax.numpy as jnp
import networkx as nx

from gpjax.kernels.non_euclidean import GraphKernel, DictKernel
from gpjax.linops import identity
import jax.random as jr

# # Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)


def test_graph_kernel():
    # Create a random graph, G, and verice labels, x,
    n_verticies = 20
    n_edges = 40
    G = nx.gnm_random_graph(n_verticies, n_edges, seed=123)
    x = jnp.arange(n_verticies).reshape(-1, 1)

    # Compute graph laplacian
    L = nx.laplacian_matrix(G).toarray() + jnp.eye(n_verticies) * 1e-12

    # Create graph kernel
    kern = GraphKernel(laplacian=L)
    assert isinstance(kern, GraphKernel)
    assert kern.num_vertex == n_verticies
    assert kern.eigenvalues.shape == (n_verticies, 1)
    assert kern.eigenvectors.shape == (n_verticies, n_verticies)

    # Compute gram matrix
    Kxx = kern.gram(x)
    assert Kxx.shape == (n_verticies, n_verticies)

    # Check positive definiteness
    Kxx += identity(n_verticies) * 1e-6
    eigen_values = jnp.linalg.eigvalsh(Kxx.to_dense())
    assert all(eigen_values > 0)


def test_dict_kernel():
    x = jr.normal(jr.PRNGKey(123), (5000, 3))
    gram = jnp.cov(x.T)
    params = DictKernel.gram_to_sdev_cholesky_lower(gram)
    dk = DictKernel(
        inspace_vals=list(range(len(gram))),
        sdev=params.sdev,
        cholesky_lower=params.cholesky_lower,
    )
    print(dk.gram, gram)
    assert jnp.allclose(dk.gram, gram)
    assert False
