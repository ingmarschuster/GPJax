"""Isotonic optimization routines in JAX."""

import warnings
import jax.numpy as np


def isotonic_l2(y):
    n = y.shape[0]
    target = np.arange(n)
    c = np.ones(n)
    sums = np.zeros(n)
    sol = np.zeros_like(y)

    for i in range(n):
        sol = sol.at[i].set(y[i])
        sums = sums.at[i].set(y[i])

    i = 0
    while i < n:
        k = target[i] + 1
        if k == n:
            break
        if sol[i] > sol[k]:
            i = k
            continue
        sum_y = sums[i]
        sum_c = c[i]
        while True:
            prev_y = sol[k]
            sum_y += sums[k]
            sum_c += c[k]
            k = target[k] + 1
            if k == n or prev_y > sol[k]:
                sol = sol.at[i].set(sum_y / sum_c)
                sums = sums.at[i].set(sum_y)
                c = c.at[i].set(sum_c)
                target = target.at[i].set(k - 1)
                target = target.at[k - 1].set(i)
                if i > 0:
                    i = target[i - 1]
                break

    i = 0
    while i < n:
        k = target[i] + 1
        sol = sol.at[i + 1 : k].set(sol[i])
        i = k
    return sol


def _log_add_exp(x, y):
    larger = np.maximum(x, y)
    smaller = np.minimum(x, y)
    return larger + np.log1p(np.exp(smaller - larger))


def isotonic_kl(y, w):
    sol = np.zeros_like(y)
    n = y.shape[0]
    target = np.arange(n)
    lse_y_ = np.zeros(n)
    lse_w_ = np.zeros(n)

    for i in range(n):
        sol = sol.at[i].set(y[i] - w[i])
        lse_y_ = lse_y_.at[i].set(y[i])
        lse_w_ = lse_w_.at[i].set(w[i])

    i = 0
    while i < n:
        k = target[i] + 1
        if k == n:
            break
        if sol[i] > sol[k]:
            i = k
            continue
        lse_y = lse_y_[i]
        lse_w = lse_w_[i]
        while True:
            prev_y = sol[k]
            lse_y = _log_add_exp(lse_y, lse_y_[k])
            lse_w = _log_add_exp(lse_w, lse_w_[k])
            k = target[k] + 1
            if k == n or prev_y > sol[k]:
                sol = sol.at[i].set(lse_y - lse_w)
                lse_y_ = lse_y_.at[i].set(lse_y)
                lse_w_ = lse_w_.at[i].set(lse_w)
                target = target.at[i].set(k - 1)
                target = target.at[k - 1].set(i)
                if i > 0:
                    i = target[i - 1]
                break

    i = 0
    while i < n:
        k = target[i] + 1
        sol = sol.at[i + 1 : k].set(sol[i])
        i = k
    return sol
