import jax
import jax.numpy as jnp
import jaxopt as jo

niter = 1000


def projection_non_negative_after0(x: jnp.array, hyperparams=None) -> jnp.array:
    return x.at[1:].set(jnp.where(x[1:] < 0.0, 0.0, x[1:]))


def constrain_param(param):
    return param.at[1:].set(param[0] - jnp.cumsum(param[1:]))


def isotonic_l2(y: jnp.array) -> jnp.array:
    """Solves an isotonic regression problem with L2 loss using projected gradient descent.

    Formally, it solves argmin_{v_1 >= ... >= v_n} 0.5 ||v - y||^2.

    Args:
      y: input to isotonic regression, a 1d-array.
    """

    def loss_fn(param):
        return jnp.sum(jnp.abs(y - constrain_param(param))) / 2

    solver = jo.ProjectedGradient(
        fun=loss_fn, maxiter=niter, projection=projection_non_negative_after0
    )
    sol = solver.run(y.at[1:].set(0.0)).params
    return constrain_param(sol)


def isotonic_kl(y: jnp.array, w: jnp.array) -> jnp.array:
    """Solves an isotonic regression problem with KL divergence using projected gradient descent.

      Formally, it solves argmin_{v_1 >= ... >= v_n} <e^{y-v}, 1> + <e^w, v>.

    Args:
      y: input to isotonic optimization, a 1d-array.
      w: input to isotonic optimization, a 1d-array.
    """

    def loss_fn(param):
        constr = constrain_param(param)
        return jnp.sum(jnp.exp(y - constr)) + jnp.sum(jnp.exp(w) * constr)

    solver = jo.ProjectedGradient(
        fun=loss_fn, maxiter=niter, projection=projection_non_negative_after0
    )
    sol = solver.run(y.at[1:].set(0.0)).params
    return constrain_param(sol)
