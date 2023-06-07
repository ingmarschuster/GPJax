from jax import numpy as jnp
import jax
from .third_party.jaxopt_isotonic import isotonic_l2, isotonic_kl


def isotonic_l2(input_s, input_w=None):
    if input_w is None:
        input_w = jnp.arange(len(input_s))[::-1] + 1
    input_w = input_w.astype(input_s.dtype)
    return isotonic_l2(input_s - input_w)


def isotonic_kl(input_s, input_w=None):
    if input_w is None:
        input_w = jnp.arange(len(input_s))[::-1] + 1
    input_w = input_w.astype(input_s.dtype)
    solution = jnp.zeros(len(input_s)).astype(input_s.dtype)
    return isotonic_kl(input_s, input_w.astype(input_s.dtype))


def _partition(solution, eps=1e-9):
    if len(solution) == 0:
        return []

    sizes = [1]

    for i in range(1, len(solution)):
        if abs(solution[i] - solution[i - 1]) > eps:
            sizes.append(0)
        sizes[-1] += 1

    return sizes


def _check_regularization(regularization):
    if regularization not in ("l2", "kl"):
        raise ValueError(
            "'regularization' should be either 'l2' or 'kl' "
            "but got %s." % str(regularization)
        )


class Isotonic:
    def __init__(self, input_s, input_w, regularization="l2"):
        self.input_s = input_s
        self.input_w = input_w
        _check_regularization(regularization)
        self.regularization = regularization
        self.solution_ = None

    def compute(self):
        if self.regularization == "l2":
            self.solution_ = isotonic_l2(self.input_s, self.input_w)
        else:
            self.solution_ = isotonic_kl(self.input_s, self.input_w)
        return self.solution_


def _inv_permutation(permutation):
    inv_permutation = jnp.zeros(len(permutation), dtype=int)
    inv_permutation = inv_permutation.at[permutation].set(jnp.arange(len(permutation)))
    return inv_permutation


class Projection:
    def __init__(self, input_theta, input_w=None, regularization="l2"):
        if input_w is None:
            input_w = jnp.arange(len(input_theta))[::-1] + 1
        self.input_theta = jnp.asarray(input_theta)
        self.input_w = jnp.asarray(input_w)
        _check_regularization(regularization)
        self.regularization = regularization
        self.isotonic = None

    def compute(self):
        self.permutation = jnp.argsort(self.input_theta)[::-1]
        input_s = self.input_theta[self.permutation]

        self.isotonic_ = Isotonic(input_s, self.input_w, self.regularization)
        dual_sol = self.isotonic_.compute()
        primal_sol = input_s - dual_sol

        self.inv_permutation = _inv_permutation(self.permutation)
        return primal_sol[self.inv_permutation]


def _check_direction(direction):
    if direction not in ("ASCENDING", "DESCENDING"):
        raise ValueError("direction should be either 'ASCENDING' or 'DESCENDING'")


class SoftRank:
    def __init__(
        self,
        values,
        direction="ASCENDING",
        regularization_strength=1.0,
        regularization="l2",
    ):
        self.values = jnp.asarray(values)
        self.input_w = jnp.arange(len(values))[::-1] + 1
        _check_direction(direction)
        sign = 1 if direction == "ASCENDING" else -1
        self.scale = sign / regularization_strength
        _check_regularization(regularization)
        self.regularization = regularization
        self.projection_ = None

    def compute(self):
        if self.regularization == "kl":
            self.projection_ = Projection(
                self.values * self.scale,
                jnp.log(self.input_w),
                regularization=self.regularization,
            )
            self.factor = jnp.exp(self.projection_.compute())
            return self.factor
        else:
            self.projection_ = Projection(
                self.values * self.scale,
                self.input_w,
                regularization=self.regularization,
            )
            self.factor = 1.0
            return self.projection_.compute()


class SoftSort:
    def __init__(
        self,
        values,
        direction="ASCENDING",
        regularization_strength=1.0,
        regularization="l2",
    ):
        self.values = jnp.asarray(values)
        _check_direction(direction)
        self.sign = 1 if direction == "DESCENDING" else -1
        self.regularization_strength = regularization_strength
        _check_regularization(regularization)
        self.regularization = regularization
        self.isotonic_ = None

    def compute(self):
        size = len(self.values)
        input_w = jnp.arange(1, size + 1)[::-1] / self.regularization_strength
        values = self.sign * self.values
        self.permutation_ = jnp.argsort(values)[::-1]
        s = values[self.permutation_]

        self.isotonic_ = Isotonic(input_w, s, regularization=self.regularization)
        res = self.isotonic_.compute()

        # We set s as the first argument as we want the derivatives w.r.t. s.
        self.isotonic_.s = s
        return self.sign * (input_w - res)


def soft_rank(
    values, direction="ASCENDING", regularization_strength=1.0, regularization="l2"
):
    return SoftRank(
        values,
        regularization_strength=regularization_strength,
        direction=direction,
        regularization=regularization,
    ).compute()


def hard_rank(values, direction="ASCENDING"):
    output = jnp.argsort(jnp.argsort(values))
    if direction == "DESCENDING":
        output = output[::-1]
    return output + 1


def soft_sort(
    values, direction="ASCENDING", regularization_strength=1.0, regularization="l2"
):
    return SoftSort(
        values,
        regularization_strength=regularization_strength,
        direction=direction,
        regularization=regularization,
    ).compute()
