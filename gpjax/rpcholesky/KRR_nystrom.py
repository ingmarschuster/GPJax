import jax
import jax.numpy as np
from .matrix import KernelMatrix, NonsymmetricKernelMatrix
from jax.scipy import linalg
import time


class KRR_Nystrom:
    def __init__(self, kernel="gaussian"):
        self.kernel = kernel

    def fit(self, Xtr, Ytr, lamb):
        self.Xtr = Xtr
        ts = time.time()
        self.K = KernelMatrix(Xtr, kernel=self.kernel)
        K_exact = self.K[:, :]
        self.sol = linalg.solve(
            K_exact + lamb * np.shape(K_exact)[0] * np.eye(np.shape(K_exact)[0]),
            Ytr,
            sym_pos=True,
        )
        te = time.time()
        self.linsolve_time = te - ts

    def predict(self, Xts):
        ts = time.time()
        K_pred = NonsymmetricKernelMatrix(Xts, self.Xtr, kernel=self.kernel)
        K_pred_exact = K_pred[:, :]
        preds = K_pred_exact @ self.sol
        te = time.time()
        self.pred_time = te - ts
        return preds

    def fit_Nystrom(
        self,
        Xtr,
        Ytr,
        lamb,
        sample_num,
        sample_method,
    ):
        self.Xtr = Xtr
        self.K = KernelMatrix(Xtr, kernel=self.kernel)
        ts = time.time()
        lra = sample_method(self.K, sample_num)
        arr_idx = lra.idx
        KMn = lra.rows
        te = time.time()
        self.sample_idx = arr_idx
        self.sample_time = te - ts
        self.queries = self.K.num_queries()

        trK = self.K.trace()
        self.reltrace_err = (trK - lra.trace()) / trK

        self.K.reset_queries()
        ts = time.time()
        # closed form solution for Nyström approximation
        KMM = KMn[:, arr_idx]
        KnM = KMn.T
        self.sol = linalg.solve(
            KMn @ KnM
            + KnM.shape[0] * lamb * KMM
            + 100 * KMM.max() * np.finfo(float).eps * np.eye(sample_num),
            KMn @ Ytr,
            sym_pos=True,
        )
        te = time.time()
        self.linsolve_time = te - ts

    def predict_Nystrom(self, Xts):
        ts = time.time()
        K_pred = NonsymmetricKernelMatrix(Xts, self.Xtr, kernel=self.kernel)
        KtM = K_pred[:, self.sample_idx]
        preds = KtM @ self.sol
        te = time.time()
        self.pred_time = te - ts
        return preds
