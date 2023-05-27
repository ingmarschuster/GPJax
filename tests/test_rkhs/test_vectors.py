import pytest

import numpy as np
import jax.numpy as jnp

from gpjax.rkhs.vector import RkhsVec
from gpjax.kernels import RBF, Matern12, Matern32, Matern52
from gpjax.rkhs.reduce import Prefactors, LinearReduce, BalancedRed, Sum

rng = np.random.RandomState(1)


kernel_setups = [RBF(lengthscale=1.0), Matern12(), Matern32(), Matern52()]


@pytest.mark.parametrize("D", [1, 5])
@pytest.mark.parametrize("kernel", kernel_setups)
@pytest.mark.parametrize("N", [10])
def test_RkhsVec(D, kernel, N):
    X = rng.randn(N, D)
    pref1 = Prefactors(jnp.ones(len(X)).astype(jnp.float32))
    rv = pref1 @ RkhsVec(k=kernel, insp_pts=X)

    assert rv.reduce[0] == pref1

    pref2 = Prefactors(jnp.ones(N + 1).astype(jnp.float32))
    rv2 = Sum() @ pref2 @ RkhsVec(k=kernel, insp_pts=rng.randn(N + 1, D))
    assert np.allclose(
        rv.outer_inner(rv),
        rv.k.gram(rv.insp_pts).to_dense()
        * np.outer(rv.reduce[0].prefactors, rv.reduce[0].prefactors),
    ), "Simple vector computation not accurate"
    assert np.allclose(
        rv.outer_inner(rv2),
        (
            rv.k.cross_covariance(rv.insp_pts, rv2.insp_pts)
            * np.outer(rv.reduce[0].prefactors, rv2.reduce[-1].prefactors)
        ).sum(1, keepdims=True),
    ), "Simple vector computation not accurate"

    N = 4
    X = rng.randn(N, D)

    rv = (
        BalancedRed(2)
        @ Prefactors(jnp.ones(len(X)) / 2)
        @ RkhsVec(k=kernel, insp_pts=X)
    )
    el = Sum() @ Prefactors(jnp.ones(N)) @ RkhsVec(k=kernel, insp_pts=X)
    gram = el.k.gram(el.insp_pts).to_dense()
    assert np.allclose(el.outer_inner(el), np.sum(gram))
    assert np.allclose(
        np.squeeze(el.outer_inner(rv)), np.sum(gram, 1).reshape(-1, 2).mean(1)
    )

    rv = (
        BalancedRed(2)
        @ Prefactors(jnp.ones(len(X)) / 2)
        @ RkhsVec(k=kernel, insp_pts=X)
    )
    assert np.allclose(
        rv.outer_inner(rv),
        np.array(
            [
                [
                    np.mean(rv.k.gram(X[:2, :]).to_dense()),
                    np.mean(rv.k.cross_covariance(X[:2, :], X[2:, :])),
                ],
                [
                    np.mean(rv.k.cross_covariance(X[:2, :], X[2:, :])),
                    np.mean(rv.k.gram(X[2:, :]).to_dense()),
                ],
            ]
        ),
    ), "Balanced vector computation not accurate"
