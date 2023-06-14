#!/usr/bin/env python3

import jax
import jax.numpy as np
from jax import random
from .lra import PSDLowRank


def cholesky_helper(A, k, alg):
    n = A.shape[0]
    diags = np.diag(A)

    # row ordering, is much faster for large scale problems
    F = np.zeros((k, n))
    rows = np.zeros((k, n))
    rng = random.PRNGKey(0)

    arr_idx = []

    for i in range(k):
        if alg == "rp":
            idx = random.choice(rng, np.arange(n), shape=(), p=diags / np.sum(diags))
        elif alg == "greedy":
            idx = random.choice(rng, np.where(diags == np.max(diags))[0], shape=())
        else:
            raise RuntimeError("Algorithm '{}' not recognized".format(alg))

        arr_idx.append(idx)
        rows = rows.at[i, :].set(A[idx, :])
        F = F.at[i, :].set(
            (rows[i, :] - np.dot(F[:i, idx].T, F[:i, :])) / np.sqrt(diags[idx])
        )
        diags = diags - F[i, :] ** 2
        diags = np.clip(diags, a_min=0)

    return PSDLowRank(F.T, idx=arr_idx, rows=rows)


def block_cholesky_helper(A, k, b, alg):
    diags = np.diag(A)
    n = A.shape[0]

    # row ordering
    F = np.zeros((k, n))
    rows = np.zeros((k, n))

    rng = random.PRNGKey(0)

    arr_idx = []

    cols = 0
    while cols < k:
        block_size = min(k - cols, b)

        if alg == "rp":
            idx = random.choice(
                rng,
                np.arange(n),
                shape=(2 * block_size,),
                p=diags / np.sum(diags),
                replace=False,
            )
            idx = np.unique(idx)[:block_size]
            block_size = len(idx)
        elif alg == "greedy":
            idx = np.argpartition(diags, -block_size)[-block_size:]
        else:
            raise RuntimeError("Algorithm '{}' not recognized".format(alg))

        arr_idx.extend(idx)
        rows = rows.at[cols : cols + block_size, :].set(A[idx, :])
        F = F.at[cols : cols + block_size, :].set(
            rows[cols : cols + block_size, :] - np.dot(F[0:cols, idx].T, F[0:cols, :])
        )
        C = F[cols : cols + block_size, idx]
        L = np.linalg.cholesky(C + 100 * np.finfo(float).eps * np.eye(block_size))
        F = F.at[cols : cols + block_size, :].set(
            np.linalg.solve(L, F[cols : cols + block_size, :])
        )
        diags = diags - np.sum(F[cols : cols + block_size, :] ** 2, axis=0)
        diags = np.clip(diags, a_min=0)

        cols += block_size

    return PSDLowRank(F.T, idx=arr_idx, rows=rows)


def rp_cholesky(A, k):
    return cholesky_helper(A, k, "rp")


def greedy(A, k):
    return cholesky_helper(A, k, "greedy")


def block_rp_cholesky(A, k, b=100):
    return block_cholesky_helper(A, k, b, "rp")


def block_greedy(A, k, b=100):
    return block_cholesky_helper(A, k, b, "greedy")
