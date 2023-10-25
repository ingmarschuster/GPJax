#!/usr/bin/env python3

import jax
import jax.numpy as np
from abc import ABC, abstractmethod

# Low Rank Approximations (LRA) for positive semidefinite matrices


class AbstractPSDLowRank(ABC):
    def __init__(self, **kwargs):
        self.idx = kwargs["idx"] if ("idx" in kwargs) else None
        self.rows = kwargs["rows"] if ("rows" in kwargs) else None

    @abstractmethod
    def trace(self):
        pass

    @abstractmethod
    def __matmul__(self, other):
        pass

    def __rmatmul__(self, other):
        return (self @ other.T).T

    @abstractmethod
    def rank(self):
        pass

    @abstractmethod
    def eigenvalue_decomposition(self):
        pass

    def matrix(self):
        return self @ np.eye(self.shape[0])


class CompactEigenvalueDecomposition(AbstractPSDLowRank):
    def __init__(self, V, Lambda, **kwargs):
        super().__init__(**kwargs)
        self.V = V
        self.Lambda = Lambda
        self.shape = (self.V[0], self.V[0])

    @staticmethod
    def from_F(F, **kwargs):
        Q, R = np.linalg.qr(F, "reduced")
        U, S, _ = np.linalg.svd(R)
        return CompactEigenvalueDecomposition(Q @ U, S**2, **kwargs)

    def trace(self):
        return np.sum(self.Lambda)

    def __matmul__(self, other):
        return self.V @ (self.Lambda * (self.V.T @ other))

    def rank(self):
        return len(self.Lambda)

    def matrix(self):
        return self.V @ (self.Lambda * self.V.T)

    def eigenvalue_decomposition(self):
        return self

    def krr(self, b, lam):
        VTb = self.V.T @ b
        return b / lam + self.V @ (
            np.divide(VTb, self.Lambda.reshape(VTb.shape) + lam) - VTb / lam
        )

    def krr_vec(self, b, lam):
        VTb = self.V.T @ b
        diag = self.Lambda.reshape(VTb.shape[0]) + lam
        return b / lam + self.V @ (np.divide(VTb, diag[:, np.newaxis]) - VTb / lam)


class PSDLowRank(AbstractPSDLowRank):
    def __init__(self, F, **kwargs):
        super().__init__(**kwargs)
        self.F = F
        self.shape = (F.shape[0], F.shape[0])

    def trace(self):
        return np.linalg.norm(self.F, "fro") ** 2

    def __matmul__(self, other):
        return self.F @ (self.F.T @ other)

    def rank(self):
        return self.F.shape[1]

    def matrix(self):
        return self.F @ self.F.T

    def eigenvalue_decomposition(self):
        return CompactEigenvalueDecomposition.from_F(
            self.F, idx=self.idx, rows=self.rows
        )

    def scale(self, scaling):
        return PSDLowRank(scaling[:, np.newaxis] * self.F, idx=self.idx)
