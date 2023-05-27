import copy

import jax.numpy as np
import numpy as onp
import pytest
from numpy.testing import assert_allclose


from gpjax.rkhs.reduce import (
    LinearReduce,
    SparseReduce,
    Sum,
    Mean,
    Prefactors,
    Scale,
    Repeat,
    TileView,
    NoReduce,
    ChainedReduce,
    BalancedRed,
    Center,
    Kmer,
)

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
    r2 = r1.linearize(gram.shape)
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
    assert np.allclose((r.linmap(gram.shape) @ gram)[0], gram.sum(0))
    assert np.allclose((r.linearize(gram.shape) @ gram)[0], gram.sum(0))
    assert np.allclose(rgr[1], gram[:2].sum(0) / 2 + gram[2:].sum(0) * 2)


def test_Sum_Mean():
    gram = np.array(rng.randn(4, 3))
    for r, op in [(Sum(), np.sum), (Mean(), np.mean)]:
        rgr = r @ gram
        rgr_l = r.linearize(gram.shape) @ gram
        ogr = op(gram, 0, keepdims=True)
        assert np.allclose(rgr, ogr)
        assert np.allclose(rgr_l, ogr)


def test_Prefactors_Scale():
    gram = np.array(rng.randn(4, 3))
    for r, op in [
        (
            Prefactors(np.arange(4).astype(np.float32)),
            lambda x: x * np.arange(4).reshape(-1, 1),
        ),
        (Scale(2.0), lambda x: x * 2.0),
    ]:
        rgr = r @ gram
        rgr_l = r.linearize(gram.shape) @ gram
        ogr = op(gram)
        assert np.allclose(rgr, ogr)
        assert np.allclose(rgr_l, ogr)


#  Repeat, TileView, NoReduce, ChainedReduce


def test_Repeat_Tile():
    gram = np.array(rng.randn(4, 3))
    for r, op in [
        (Repeat(2), lambda x: np.repeat(x, 2, 0)),
        (TileView(8), lambda x: np.tile(x, (2, 1))),
    ]:
        rgr = r @ gram
        rgr_lm = r.linmap(gram.shape) @ gram
        rgr_l = r.linearize(gram.shape) @ gram
        ogr = op(gram)
        assert np.allclose(rgr, ogr)
        assert np.allclose(rgr_lm, ogr)
        assert np.allclose(rgr_l, ogr)


def test_NoReduce():
    gram = np.array(rng.randn(4, 3))
    r = NoReduce()
    rgr = r @ gram
    rgr_l = r.linearize(gram.shape) @ gram
    assert np.allclose(rgr, gram)
    assert np.allclose(rgr_l, gram)


def test_ChainedReduce():
    gram = np.array(rng.randn(4, 3))
    r = ChainedReduce([TileView(16), Repeat(2)])
    rgr = r @ gram
    rgr_l = r.linearize(gram.shape) @ gram
    rgr_lm = r.linmap(gram.shape) @ gram
    ogr = np.tile(np.repeat(gram, 2, 0), (2, 1))
    assert np.allclose(rgr, ogr)
    assert np.allclose(rgr_l, ogr)
    assert np.allclose(rgr_lm, ogr)

    rgr = TileView(16) @ Repeat(2) @ gram
    assert np.allclose(rgr, ogr)


def test_BalancedRed():
    gram = np.array(rng.randn(4, 3))
    red = BalancedRed(2)
    expected_result = np.array([gram[:2].sum(0), gram[2:].sum(0)])
    for r in red, red.linearize(gram.shape):
        assert np.allclose(r @ gram, expected_result)

    red = BalancedRed(2, True)
    expected_result = np.array([gram[:2].sum(0) / 2, gram[2:].sum(0) / 2])
    for r in red, red.linearize(gram.shape):
        assert np.allclose(r @ gram, expected_result)


def test_Center():
    gram = np.array(rng.randn(4, 3))
    red = Center()
    expected_result = gram - gram.mean(0)
    for r in red, red.linearize(gram.shape):
        assert np.allclose(r @ gram, expected_result)


def test_Kmer():
    gram = np.array(rng.randn(4, 3))
    expected_result = np.array(
        [gram[i : i + 2].sum(0, keepdims=True) for i in range(3)]
    ).squeeze()
    for r in (
        Kmer(2, 4, False, "linear"),
        Kmer(2, 4, False, "sparse"),
        Kmer(2, 4, False, "binary"),
    ):
        assert np.allclose(r @ gram, expected_result)

    idx_res = np.array([gram[i].sum(0) for i in Kmer(2, 4, False, "index")])
    assert np.allclose(idx_res, expected_result)

    expected_result = expected_result / 2
    for r in (
        Kmer(2, 4, True, "linear"),
        Kmer(2, 4, True, "sparse"),
        Kmer(2, 4, True, "binary"),
    ):
        assert np.allclose(r @ gram, expected_result)
