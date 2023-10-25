#!/usr/bin/env python3

import jax
import jax.numpy as np
from abc import ABC, abstractmethod
import numbers
from ..kernels import AbstractKernel
from functools import partial


# symmetric matrices need to provide diag and trace
# _diag_helper and _getitem_helper are used to count queries and evaluate the matrix
# these use _function, _function_vec, _function_mtx to evaluate the matrix in different ways
# corresponding to kernel (_function), jax.vmap(kernel) (_function_vec), kernel.cross_covariance (_function_mtx)


class AbstractPSDMatrix(ABC):
    """Abstract class for positive semidefinite matrices.

    Attributes:
        queries (int): Counter for the number of queries made to the matrix.
        count_queries (bool): Flag to control whether to count queries.
    """

    def __init__(self, **kwargs):
        self.queries = 0
        self.count_queries = (
            kwargs["count_queries"] if ("count_queries" in kwargs) else True
        )

    @abstractmethod
    def _diag_helper(self, *args):
        pass

    @abstractmethod
    def _getitem_helper(self, *args):
        pass

    def __getitem__(self, *args):
        to_return = self._getitem_helper(*args)
        if isinstance(to_return, np.ndarray):
            self.queries += to_return.size
        else:
            self.queries += 1
        return to_return

    def diag(self, *args):
        to_return = self._diag_helper(*args)
        if isinstance(to_return, np.ndarray):
            self.queries += to_return.size
        else:
            self.queries += 1
        return to_return

    def __call__(self, *args):
        return self.__getitem__(*args)

    def trace(self):
        return sum(self.diag())

    def num_queries(self):
        return self.queries

    def reset_queries(self):
        self.queries = 0

    def to_matrix(self):
        return PSDMatrix(self[:, :])


class PSDMatrix(AbstractPSDMatrix):
    def __init__(self, A, **kwargs):
        super().__init__(**kwargs)
        self.matrix = A
        self.shape = A.shape

    def _diag_helper(self, *args):
        if len(args) > 0:
            X = list(args[0])
        else:
            X = range(self.matrix.shape[0])
        return self.matrix[X, X]

    def _getitem_helper(self, *args):
        return self.matrix.__getitem__(*args)


class FunctionMatrix(AbstractPSDMatrix):
    def __init__(self, n, **kwargs):
        self.shape = (n, n)
        super().__init__(**kwargs)

    @abstractmethod
    def _function(self, i, j):
        pass

    @abstractmethod
    def _function_vec(self, vec_i, vec_j):
        pass

    @abstractmethod
    def _function_mtx(self, vec_i, vec_j):
        pass

    def _diag_helper(self, *args):
        if len(args) > 0:
            idx = list(args[0])
        else:
            idx = np.arange(self.shape[0], dtype=np.int32)
        return self._function_vec(idx, idx)

    def _getitem_helper(self, *args):
        idx = args[0]
        if len(idx) == 1:
            idx = [idx, idx]
        else:
            idx = list(idx)

        for j in range(2):
            if isinstance(idx[j], np.ndarray):
                idx[j] = idx[j].ravel().tolist()
            elif isinstance(idx[j], slice):
                idx[j] = list(range(self.shape[0]))[idx[j]]
            elif isinstance(idx[j], list):
                pass
            elif isinstance(idx[j], numbers.Integral):
                idx[j] = [idx[j]]
            else:
                raise RuntimeError(
                    "Indexing not implemented with index of type {}".format(type(idx))
                )

        if len(idx[0]) == 1 and len(idx[1]) == 1:
            return self._function(idx[0], idx[1])
        else:
            mtx = self._function_mtx(idx[0], idx[1])
            if len(idx[0]) == 1:
                return mtx[0, :]
            elif len(idx[1]) == 1:
                return mtx[:, 0]
            else:
                return mtx


class KernelMatrix(FunctionMatrix):
    @staticmethod
    def kernel_from_input(kernel, **kwargs):
        return kernel, jax.vmap(kernel), kernel.cross_covariance

    def __init__(self, X, kernel="gaussian", **kwargs):
        super().__init__(X.shape[0], **kwargs)
        self.data = X
        kernel, kernel_vec, kernel_mtx = KernelMatrix.kernel_from_input(
            kernel, **kwargs
        )
        self.kernel = kernel
        self.kernel_vec = kernel_vec
        self.kernel_mtx = kernel_mtx

    def _function(self, i, j):
        return self.kernel(self.data[i, :], self.data[j, :])

    def _function_vec(self, vec_i, vec_j):
        return self.kernel_vec(self.data[vec_i, :], self.data[vec_j, :])

    def _function_mtx(self, vec_i, vec_j):
        return self.kernel_mtx(self.data[vec_i, :], self.data[vec_j, :])


class NonsymmetricKernelMatrix(object):
    def __init__(self, X, Y, kernel="gaussian", **kwargs):
        self.X = X
        self.Y = Y
        self.shape = (X.shape[0], Y.shape[0])
        self.kernel, self.kernel_vec, self.kernel_mtx = KernelMatrix.kernel_from_input(
            kernel, **kwargs
        )

    def _function(self, i, j):
        return self.kernel(self.X[i, :], self.Y[j, :])

    def _function_vec(self, vec_i, vec_j):
        return self.kernel_vec(self.X[vec_i, :], self.Y[vec_j, :])

    def _function_mtx(self, vec_i, vec_j):
        return self.kernel_mtx(self.X[vec_i, :], self.Y[vec_j, :])

    def _getitem_helper(self, *args):
        idx = args[0]
        if len(idx) == 1:
            idx = [idx, idx]
        else:
            idx = list(idx)

        for j in range(2):
            if isinstance(idx[j], np.ndarray):
                idx[j] = idx[j].ravel().tolist()
            elif isinstance(idx[j], slice):
                idx[j] = list(range(self.shape[j]))[idx[j]]
            elif isinstance(idx[j], list):
                pass
            elif isinstance(idx[j], numbers.Integral):
                idx[j] = [idx[j]]
            else:
                raise RuntimeError(
                    "Indexing not implemented with index of type {}".format(type(idx))
                )

        if len(idx[0]) == 1 and len(idx[1]) == 1:
            return self._function(idx[0], idx[1])
        else:
            mtx = self._function_mtx(idx[0], idx[1])
            if len(idx[0]) == 1:
                return mtx[0, :]
            elif len(idx[1]) == 1:
                return mtx[:, 0]
            else:
                return mtx

    def __getitem__(self, *args):
        return self._getitem_helper(*args)
