import copy

import jax.numpy as np
import numpy as onp
import pytest
from numpy.testing import assert_allclose


from gpjax.rkhs.reduce import LinearReduce, SparseReduce

rng = onp.random.RandomState(1)


def test_SparseReduce():
    gram = np.array(rng.randn(4, 3))
    r1 = SparseReduce(
        [
            np.array([[0, 1]], dtype=np.int32),
            np.array([[0, 3]], dtype=np.int32),
            np.array([[0, 2]], dtype=np.int32),
        ],
        True,
    )
    r2 = r1.linearize(gram.shape, 0)
    for r in [r1, r2]:
        rgr = r @ gram
        assert np.allclose(rgr[0], (gram[0] + gram[1]) / 2)
        assert np.allclose(rgr[1], (gram[0] + gram[3]) / 2)
        assert np.allclose(rgr[2], (gram[0] + gram[2]) / 2)


def test_reduce_from_unique():
    inp = np.array([1, 1, 0, 3, 5, 0], dtype=np.float32)
    un1, cts1, red1 = SparseReduce.sum_from_unique(inp)
    un3, cts3, red3 = LinearReduce.sum_from_unique(inp)

    args = np.argsort(un1)

    i_out = np.outer(inp, inp)
    assert np.all(red1 @ i_out == (red3 @ i_out)[args, :])


def test_LinearReduce():
    gram = np.array(rng.randn(4, 3))
    r = LinearReduce(np.array([(1, 1, 1, 1), (0.5, 0.5, 2, 2)], dtype=np.float32))
    rgr = r @ gram
    assert np.allclose(rgr[0], gram.sum(0))
    assert np.allclose(rgr[1], gram[:2].sum(0) / 2 + gram[2:].sum(0) * 2)
