import gpjax as gpx
import jax, jax.random as jr
import jax.numpy as np


def test_kernelmatrix():
    key = jr.PRNGKey(0)
    X = jr.normal(key, (100, 2))
    kernel = gpx.RBF()
    gram = kernel.gram(X)
    km_obj = gpx.rpcholesky.matrix.KernelMatrix(X, kernel)
    # km_obj._function_vec(*([list(range(km_obj.shape[0]))] * 2))
    # km_obj._function_vec(*([(range(km_obj.shape[0]))] * 2))
    print(np.mean((np.diag(gram.to_dense()) - km_obj.diag()) ** 2))


def test_krr_nystrom():
    key = jr.PRNGKey(0)
    X = jr.normal(key, (10000, 2))

    approx_rank = 10  # int(np.sqrt(X.shape[0]))
    print(f"Approx rank: {approx_rank}, X shape: {X.shape}")
    Y = X**2
    lamb = 1.0e-8
    kernel = gpx.RBF()

    krr_n = gpx.rpcholesky.KRR_Nystrom(kernel)

    krr_n.fit(X, Y, lamb)
    mse = np.mean((krr_n.predict(X) - Y) ** 2)
    print(
        f"MSE Krr: {mse:e}, fit time {krr_n.linsolve_time:e}s, pred time {krr_n.pred_time:e}s"
    )

    krr_n.fit_Nystrom(
        X,
        Y,
        lamb,
        approx_rank,
        gpx.rpcholesky.rp_cholesky,
    )
    mse = np.mean((krr_n.predict_Nystrom(X) - Y) ** 2)
    print(
        f"MSE Krr: {mse:e}, fit time {krr_n.linsolve_time:e}s, pred time {krr_n.pred_time:e}s"
    )
