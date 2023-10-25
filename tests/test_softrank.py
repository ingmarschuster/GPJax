import jax.numpy as jnp
import numpy as onp
import jax.random as jr

import gpjax.softrank.third_party.jaxopt_isotonic as pji
import gpjax.softrank.third_party.isotonic as iso
import gpjax.softrank.jax_ops as ojo
import gpjax.softrank.pure_jax_ops_try2 as pjo

# import gpjax.softrank.pure_jax_ops as pjo


def test_isotonic():
    values = [
        jnp.array([1.0, 2.0, 3.0]),
        jnp.array([1.0, 2.0, 3.0]),
        jnp.array(
            [
                913335.56,
                895153.1,
                847238.6,
                588461.44,
                533908.2,
                427497.28,
                267232.47,
                167019.47,
                62466.527,
                58543.4,
            ]
        ),
    ]
    weights = [jnp.array([1.0, 1.0, 1.0]), jnp.array([1.0, 1.0, 1.0]) / 2]
    for v in values:
        for w in weights:
            if len(w) != len(v):
                continue
            sol_orig = onp.zeros_like(v)
            iso.isotonic_kl(onp.array(v), onp.array(w), sol_orig)
            sol_modified = pji.isotonic_kl(v, w)
            print("isotonic_kl orig", sol_orig)
            print("isotonic_kl pure jax", sol_modified)
            print("diff", jnp.abs(sol_orig - sol_modified).sum())
            assert jnp.abs(sol_orig - sol_modified).sum() < 1e-3
        sol_orig = onp.zeros_like(v)
        sol_modified = pji.isotonic_l2(v)
        iso.isotonic_l2(onp.array(v), sol_orig)
        print("isotonic_l2 orig", sol_orig)
        print("isotonic_l2 pure jax", sol_modified)
        print("diff", jnp.abs(sol_orig - sol_modified).sum())
        assert jnp.abs(sol_orig - sol_modified).sum() < 1e-3


def test_soft_rank_l2():
    key = jr.split(jr.PRNGKey(0), 20)

    for k in key:
        for regularization_strength in [1e-6, 1e-3, 1e-1, 1e1, 1e3]:
            v = jr.uniform(
                k,
                (
                    1,
                    10,
                ),
            )
            print("value", v)
            ornk = ojo.soft_rank(
                v, regularization_strength=regularization_strength
            ).squeeze()
            prnk = pjo.soft_rank_l2(v, regularization_strength=regularization_strength)
            print("orig", ornk, "\npure jax", prnk)
            assert jnp.abs(ornk - prnk).sum() < 1e-3
