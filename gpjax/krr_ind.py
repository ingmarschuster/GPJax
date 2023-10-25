import jax.numpy as jnp
import gpjax as gpx
import gpjax.objectives as gpxobj
from gpjax.typing import ScalarFloat
from jaxtyping import Array, Float
from gpjax.base import param_field, static_field
from gpjax.dataset import Dataset
from gpjax.rkhs.vector import RkhsVec
from tensorflow_probability.substrates.jax import bijectors as tfb
from dataclasses import dataclass
import einops as eo


import abc
from abc import ABC
from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterator, List, Tuple, Union
import jax.numpy as np

from jax import random
import gpjax as gpx
import gpjax.kernels as gk
import jax
from functools import partial
from pedata.config import alphabets

from gpjax.base import param_field, static_field, Module
import pedata as ped
import tensorflow_probability.substrates.jax as tfp
from .rkhs.encoder import AbstractEncoder
from .kernels import AbstractKernel, CatKernel, PoweredExponential
from .rkhs.vector import RkhsVec, AbstractRkhsVec, ProductVec
from .rkhs import reduce


tfb = tfp.bijectors


@jax.jit
def toeplitz(x):
    if len(x.shape) == 1:
        return toeplitz(x[:, None])
    n = x.shape[-2]
    m = x.shape[-1]
    if m == n:
        return x
    if m > n:
        return x[..., :n]
    r = jnp.roll(x, m, axis=-2)
    return toeplitz(jnp.concatenate([x, r], axis=-1))


@jax.jit
def toeplitz2(x):
    n = x.shape[-1]
    iota = jnp.arange(n)

    def roll(x, i):
        return jnp.roll(x, i, axis=-1)

    return jax.vmap(roll, in_axes=(None, 0), out_axes=1)(x, iota)


@dataclass
class AaEnc(AbstractEncoder):
    """Encoder that encodes AA or DNA sequences into RKHS vectors."""

    aa_vec_scale: float = param_field(1.0, bijector=tfb.Softplus(low=0.9))

    aa_dict_k: AbstractKernel = param_field(
        CatKernel(
            stddev=np.ones(1),
            cholesky_lower=(np.tril(np.ones(2 * [1])) + np.eye(1)) / 2,
        )
    )
    aa_pos_k: AbstractKernel = param_field(
        PoweredExponential(lengthscale=0.4, power=0.5).replace_trainable(
            variance=False  # don't need to train this as we use aa_vec_scale
        )
        # + Linear().replace_trainable(variance=False)
    )

    def __post_init__(self):
        a, b = ped.load_similarity("aa", "BLOSUM62")
        # init_p = CatKernel.gram_to_stddev_cholesky_lower(b @ b.T)
        init_p = CatKernel.gram_to_sdev_cholesky_lower(b + np.eye(len(b)) * 40)
        self.aa_dict_k = CatKernel(
            stddev=init_p.sdev,
            cholesky_lower=init_p.cholesky_lower,
        )

    @property
    def features(self) -> Iterator[str]:
        """Return the features of model needs.

        Yields:
            Iterator[str]: Iterator of used features.
        """
        yield ("aa_1hot")
        yield ("aa_len")

    def __call__(self, inp: dict[str, np.ndarray]) -> RkhsVec:
        """Encode the input dataset sequences into RKHS vectors.

        Args:
            inp (ds.Dataset): Dataset with sequences to encode in huggingface datasets format.

        Returns:
            jaxrk.rkhs.FiniteVec: A finite RKHS vector.
        """
        aa_vec = constr_invec(
            inp,
            self.aa_dict_k.sdev.size,
            "aa",
            self.aa_dict_k,
            self.aa_pos_k,
            seq_len_scale=self.aa_vec_scale,
        )

        return aa_vec


def constr_invec(
    seq_info: dict[str, np.ndarray],
    alphab_size: int,
    obs_prefix: str,
    obs_kern: AbstractKernel,
    idx_kern: AbstractKernel,
    lvl2_k: AbstractKernel = None,
    seq_len_scale: float = 100.0,
    norm_idx_kern: AbstractKernel = None,
) -> AbstractRkhsVec:
    """Compute which positions occur with the same amino acid/nucleotide.

    Args:
        seq_info (SeqInfo): Named tuple with sequence information (length and one hot encoding).
        alphab_size (int): Size of the alphabet.
        obs_prefix (str): Prefix for the observed sequence.
        obs_kern (gpx.kernel.AbstractKernel): Kernel for similarity of AA/nucleotide.
        idx_kern (gpx.kernel.AbstractKernel): Kernel for similarity of positions (usually a symmetric kernel).
        lvl2_k (gpx.kernel.AbstractKernel, optional): Second level/deep kernel. Defaults to None.
        seq_len_scale (float, optional): Scaling factor. Defaults to 100..
        norm_idx_kern (gpx.kernel.AbstractKernel, optional): Kernel for similarity of normalized indices (i.e. indices from 0 to 1). Defaults to None.

    Returns:
        gpx.kernel.AbstractRkhsVec: The constructed input vector.
    """

    # Running example:
    # DNA sequences "CC" and "CG" result in the following one hot encoding:
    #             A  C  G  T
    # For "CC": [[0, 1, 0, 0, 0], [0, 1, 0, 0, 0]]
    # For "CG": [[0, 1, 0, 0, 0], [0, 0, 1, 0, 0]]
    # (actually there are only 4 nucleotides, but we add a fifth one for padding)
    #
    #
    # The RKHS sequence representation for these would be:
    # For "CC": obs_kern(C, .) * idx_kern(0, .) + obs_kern(C, .) * idx_kern(1, .)
    # For "CG": obs_kern(C, .) * idx_kern(0, .) + obs_kern(G, .) * idx_kern(1, .)
    #
    # But, we could save some computation by using
    # obs_kern(C, .) * (idx_kern(0, .) + idx_kern(1, .))
    # (for "CC") which is what is done below.

    stack_seq_idx = []
    norm_seq_idx = []
    one_hot_averaged = []
    tile_len = 0

    ndata = len(seq_info[obs_prefix + "_len"])
    one_hot_averaged = (
        seq_info[obs_prefix + "_1hot"] / (seq_info[obs_prefix + "_len"].reshape(-1, 1))
    ).reshape(
        ndata, alphab_size, -1
    )  # divide to normalize by sequence length, results in
    #                              A  C  G  T
    # For "CC": onehot_CC = [[0, 0.5, 0, 0, 0], [0, 0.5, 0, 0, 0]]
    # For "CG": onehot_CG = [[0, 0.5, 0, 0, 0], [0, 0, 0.5, 0, 0]]

    # the following reduce operations maps position i to the amino acid that occurs at i
    seq_idx_reduce = reduce.LinearReduce(np.vstack(one_hot_averaged)).replace_trainable(
        linear_map=False
    )
    # if we have a gram matrix K of dimension MxN for the position similarity
    # seq_idx_reduce_CG = jaxrk.reduce.LinearReduce(np.vstack(onehot_CG))
    # then
    # seq_idx_reduce_CG.reduce_first_ax(K)
    # results in a 5xN matrix, the first row summing up the features for positions at which an A occurs,
    # the second row summing up the features for positions at which a C occurs, etc.
    # Using `seq_idx_reduce.reduce_first_ax()` (instead of `seq_idx_reduce_CG`) will do the same for all sequences in the batch.

    max_arange = np.arange(one_hot_averaged.shape[-1])
    stack_seq_idx = np.hstack([max_arange]).reshape(-1, 1)
    tile_len = ndata * one_hot_averaged.shape[1]

    # This is the actual RKHS vector for the position similarity
    pos_vec = seq_idx_reduce @ RkhsVec(k=idx_kern, insp_pts=stack_seq_idx)

    # observation similarity, tiling because we have
    # len(sparse_reduce_indices) sequences
    obs_insp_points = np.arange(obs_kern.sdev.size)[:, np.newaxis]
    obs_reduce = reduce.TileView(tile_len)
    # if we have a gram matrix K of dimension MxN for the letter similarity,
    # then obs_reduce(K) results in a tile_lenxN matrix, where the first M rows are
    # the same as K, the second M rows are the same as K, etc. (i.e. the rows are tiled)

    obs_vec = obs_reduce @ RkhsVec(k=obs_kern, insp_pts=obs_insp_points)
    # `jaxrk.rkhs.FiniteVec(obs_kern, obs_insp_points)` is a vector representing the actual letters occuring in the sequences
    # The `reduce` argument is used to tile the vector to the correct dimension,
    # basically repeating the vector for each sequence in the batch.

    # The following reduction will sum together all the individual
    # the products of positional encoding and observation encoding
    # to get a single representation for each sequence.
    invec_reduce = reduce.BalancedRed(
        points_per_split=one_hot_averaged.shape[1],
        average=False,  # we already averaged over the sequence length through one_hot_averaged
    ).linearize(
        (tile_len, tile_len),
    )
    # Now from position vector and observation vector we can construct
    # the input vector representing complete sequences
    # elementwise multiplication of position and observation vectors
    invec = (
        reduce.Scale(seq_len_scale)
        @ invec_reduce
        @ ProductVec(rkhs_vecs=[pos_vec, obs_vec])
    )

    # Without BlancedRed, multiplication of pos_vec and obs_vec would result in a
    # (5*num_seq)xN matrix, containing all the summands of the RKHS representation for each sequence.
    # The BalancedRed reduces this to a num_seqxN matrix, where each row is the sum of the summands for one sequence.

    if lvl2_k is None:
        rval_invec = invec
    else:
        assert False, "Not implemented"
        rval_invec = RkhsVec(lvl2_k, invec)

    return rval_invec


def constr_feat_vec(
    pos: Float[Array, "N 1"],
    obs_kern: AbstractKernel,
    idx_kern: AbstractKernel,
    lvl2_k: AbstractKernel = None,
) -> AbstractRkhsVec:
    alphab_size = obs_kern.sdev.size
    pos = pos.flatten()[:, None].astype(np.float32)
    tile_pos = reduce.TileView(tile_times=alphab_size).linearize(pos.shape)
    pos_vec = tile_pos @ RkhsVec(k=idx_kern, insp_pts=pos)

    repeat_obs = reduce.Repeat(times=len(pos)).linearize((alphab_size, 1))
    obs_vec = repeat_obs @ RkhsVec(
        k=obs_kern, insp_pts=np.arange(alphab_size)[:, np.newaxis]
    )
    return ProductVec(rkhs_vecs=[pos_vec, obs_vec])


@dataclass
class KRRind(gpx.Module):
    pos: Float[Array, "N 1"] = static_field(None)
    beta: Float[Array, "M"] = param_field(None)  # , bijector=tfb.Softplus())

    aa_dict_k: AbstractKernel = param_field(
        CatKernel(
            stddev=np.ones(1),
            cholesky_lower=(np.tril(np.ones(2 * [1])) + np.eye(1)) / 2,
        )
    )
    aa_pos_k: AbstractKernel = param_field(
        PoweredExponential(lengthscale=0.4, power=0.5)
    )

    @property
    def features(self):
        yield "aa_1hot"
        yield "aa_len"

    def __post_init__(self):
        a, b = ped.load_similarity("aa", "BLOSUM62")
        # init_p = CatKernel.gram_to_stddev_cholesky_lower(b @ b.T)
        init_p = CatKernel.gram_to_sdev_cholesky_lower(b + np.eye(len(b)) * 40)
        self.aa_dict_k = gk.CatKernel(
            stddev=init_p.sdev,
            cholesky_lower=init_p.cholesky_lower,
        )
        self.aa_pos_k = self.aa_pos_k
        self.enc = AaEnc(
            1.0,
            self.aa_dict_k,
            self.aa_pos_k,
        )  # .stop_gradient()
        self.feat_vec = constr_feat_vec(
            self.pos,
            self.aa_dict_k,
            self.aa_pos_k,
        )  # .stop_gradient()
        self.beta = np.ones(len(self.feat_vec))

    def predict(self, test_inputs, train_data: Dataset):
        return self.enc(test_inputs).inner_outer(self.feat_vec) @ self.beta


@dataclass
class AbstractInducingPoints(gpx.Module, ABC):
    include_bias: bool = static_field(True)

    @abc.abstractproperty
    def features(self):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass

    def __add_bias__(self, cross_covariance):
        if self.include_bias:
            return np.hstack([cross_covariance, np.ones((len(cross_covariance), 1))])
        else:
            return cross_covariance

    @abc.abstractmethod
    def cross_covariance(self, test_inputs, train_data: Dataset):
        pass


@dataclass
class InducingPoints(AbstractInducingPoints):
    kernel: gpx.AbstractKernel = param_field(gpx.RBF())
    inducing_points: Float[Array, "N M"] = param_field(None)

    @property
    def features(self):
        yield from self.kernel.features

    def __len__(self):
        return len(self.inducing_points) + int(self.include_bias)

    def cross_covariance(self, test_inputs, train_data: Dataset):
        return self.__add_bias__(
            self.kernel.cross_covariance(test_inputs, self.inducing_points)
        )


@dataclass
class PositionwiseInducing(AbstractInducingPoints):
    num_pos: int = static_field(550)
    dict_k: AbstractKernel = param_field(
        CatKernel(
            stddev=np.ones(1),
            cholesky_lower=(np.tril(np.ones(2 * [1])) + np.eye(1)) / 2,
        )
    )
    # onehot: Float[Array, "N M"] = param_field(np.ones((1, 1)), trainable=False)

    def __post_init__(self):
        return
        # make one-hot vector with self.aa_dict_k.sdev.size entries
        inp = np.eye(self.dict_k.sdev.size)

        # repeat it num_pos times in the first dimension
        # without actually copying the data (broadcasting)
        self.onehot = (
            np.broadcast_to(inp.ravel(), (self.num_pos, inp.size))
            .reshape((self.num_pos * inp.shape[0], inp.shape[1]))
            .T
        )

    def __len__(self):
        return self.num_pos * self.dict_k.sdev.size + int(self.include_bias)

    @property
    def features(self):
        yield "aa_1hot"

    def cross_covariance(self, test_inputs, train_data: Dataset):
        resh_inp = test_inputs["aa_1hot"].reshape(
            len(test_inputs["aa_1hot"]), -1, self.dict_k.sdev.size
        )
        # max_len = test_inputs.size // self.dict_size // len(test_inputs)
        feature_per_datapoint = reduce.BalancedRed(resh_inp.shape[1], True)

        # compute all the features for each position in all data points
        rval = resh_inp @ self.dict_k.explicit_gram
        # reshape so that we have features for each AA at each position
        rval1 = rval.reshape(len(test_inputs["aa_1hot"]), -1)
        # average over all positions
        # rval2 = (rval @ self.onehot).mean(1)  # feature_per_datapoint(rval)
        # add bias term if necessary
        # assert False
        return self.__add_bias__(rval1)


@dataclass
class MultiPositionwiseInducing(AbstractInducingPoints):
    num_pos: int = static_field(550)
    dict_k: list[AbstractKernel] = param_field(
        [
            CatKernel(
                stddev=np.ones(1),
                cholesky_lower=(np.tril(np.ones(2 * [1])) + np.eye(1)) / 2,
            )
        ]
    )

    def __len__(self):
        return self.num_pos * self.dict_k[0].sdev.size * len(self.dict_k) + int(
            self.include_bias
        )

    @property
    def features(self):
        yield "aa_1hot"

    def cross_covariance(self, test_inputs, train_data: Dataset):
        resh_inp = test_inputs["aa_1hot"].reshape(
            len(test_inputs["aa_1hot"]), -1, self.dict_k[0].sdev.size
        )

        # compute all the features for each position in all data points
        # reshape so that we have features for each AA at each position
        rval = (resh_inp @ np.hstack([k.explicit_gram for k in self.dict_k])).reshape(
            len(test_inputs["aa_1hot"]), -1
        )

        return self.__add_bias__(rval)


@dataclass
class PosEmbInducing(AbstractInducingPoints):
    num_pos_inducing: int = static_field(550)
    num_pos_max: int = static_field(600)
    dict_k: AbstractKernel = param_field(
        CatKernel(
            stddev=np.ones(1),
            cholesky_lower=(np.tril(np.ones(2 * [1])) + np.eye(1)) / 2,
        )
    )
    pos_k: AbstractKernel = param_field(gpx.PoweredExponential())
    distances_vec: Float[Array, "N"] = static_field(None)

    def __post_init__(self):
        self.distances_vec = np.arange(
            max(self.num_pos_max, self.num_pos_inducing)
        ).squeeze()

    def __len__(self):
        return self.num_pos_inducing * self.dict_k.sdev.size + int(self.include_bias)

    @property
    def features(self):
        yield "aa_1hot"

    @jax.jit
    def cross_covariance(self, test_inputs, train_data: Dataset):
        resh_inp = eo.rearrange(
            test_inputs["aa_1hot"],
            "batch (seqlen dictsize) -> batch seqlen dictsize",
            dictsize=self.dict_k.sdev.size,
        )
        batch, seqlen, dictsize = resh_inp.shape

        G_p = toeplitz(jax.vmap(self.pos_k.rbf)(self.distances_vec))[
            :seqlen, : self.num_pos_inducing
        ]
        # G_p = jnp.repeat(G_p, self.dict_k.sdev.size, -1)
        # G_p = eo.repeat(G_p, "seqlen nind -> seqlen (nind rep)", rep=self.dict_k.sdev.size)

        # compute all the features for each position in all data points
        # G_o = jnp.tile(
        #    resh_inp @ self.dict_k.explicit_gram, (1, 1, self.num_pos_inducing)
        # )
        G_o = resh_inp @ self.dict_k.explicit_gram
        # G_o = eo.repeat(
        #    G_o,
        #    "batch seqlen nind -> batch seqlen (tile nind)",
        #    tile=self.num_pos_inducing,
        # )
        if True:
            rval = (
                eo.einsum(
                    G_o,
                    G_p,
                    "batch seqlen dictsize, seqlen nind -> batch nind dictsize",
                )
                / seqlen
                / dictsize
            )
        else:
            rval = jnp.average(
                G_o * G_p,
                1,
            )
        # diffmax = np.max(np.abs(rval1 - rval2))
        # assert diffmax == 0
        # assert False
        return self.__add_bias__(
            eo.rearrange(rval, "batch nind dictsize -> batch (nind dictsize)")
        )


@dataclass
class MultiPosEmbInducing(AbstractInducingPoints):
    num_pos_inducing: int = static_field(550)
    num_pos_max: int = static_field(600)
    dict_k: list[AbstractKernel] = param_field(
        [
            CatKernel(
                stddev=np.ones(1),
                cholesky_lower=(np.tril(np.ones(2 * [1])) + np.eye(1)) / 2,
            )
        ]
    )
    pos_k: AbstractKernel = param_field(gpx.PoweredExponential())
    distances_vec: Float[Array, "N"] = static_field(None)
    pos_lincomb: Float[Array, "M M"] = param_field(None)
    dict_lincomb: Float[Array, "K K"] = param_field(None)

    def __post_init__(self):
        self.distances_vec = np.arange(
            max(self.num_pos_max, self.num_pos_inducing)
        ).squeeze()
        self.pos_lincomb = np.eye(self.num_pos_inducing)
        self.dict_lincomb = np.eye(len(self.dict_k) * self.dict_k[0].sdev.size)

    def __len__(self):
        return self.num_pos_inducing * self.dict_k[0].sdev.size * len(
            self.dict_k
        ) + int(self.include_bias)

    @property
    def features(self):
        yield "aa_1hot"

    @jax.jit
    def cross_covariance(self, test_inputs, train_data: Dataset):
        resh_inp = eo.rearrange(
            test_inputs["aa_1hot"],
            "batch (seqlen dictsize) -> batch seqlen dictsize",
            dictsize=self.dict_k[0].sdev.size,
        )
        batch, seqlen, dictsize = resh_inp.shape
        stackdictsize = len(self.dict_k) * dictsize

        G_p = toeplitz(jax.vmap(self.pos_k.rbf)(self.distances_vec))[
            :seqlen, : self.num_pos_inducing
        ]
        G_o = resh_inp @ np.hstack([k.explicit_gram for k in self.dict_k])

        if False:
            rval = (
                eo.einsum(
                    G_o,
                    self.dict_lincomb,
                    G_p,
                    self.pos_lincomb,
                    "batch seqlen stackdictsize, stackdictsize sdz2, seqlen nind, nind nind2 -> batch nind2 sdz2",
                )
                / seqlen
                / dictsize
            )
        else:
            rval = (
                jnp.einsum(
                    "bsd,dD,si,iI->bID",
                    G_o,
                    self.dict_lincomb,
                    G_p,
                    self.pos_lincomb,
                    optimize=True,
                )
                / seqlen
                / dictsize
            )

        return self.__add_bias__(
            eo.rearrange(rval, "batch nind stackdictsize -> batch (nind stackdictsize)")
        )


@dataclass
class PosEmbInducingOld(AbstractInducingPoints):
    num_pos: int = static_field(550)
    dict_k: AbstractKernel = param_field(
        CatKernel(
            stddev=np.ones(1),
            cholesky_lower=(np.tril(np.ones(2 * [1])) + np.eye(1)) / 2,
        )
    )
    pos_k: AbstractKernel = param_field(gpx.PoweredExponential())

    def __len__(self):
        return len(
            constr_feat_vec(
                np.arange(self.num_pos).reshape(-1, 1),
                self.dict_k,
                self.pos_k,
            )
        ) + int(self.include_bias)

    @property
    def features(self):
        yield "aa_1hot"
        yield "aa_len"

    def cross_covariance(self, test_inputs, train_data: Dataset):
        feat_vec = constr_feat_vec(
            np.arange(self.num_pos).reshape(-1, 1),
            self.dict_k,
            self.pos_k,
        )
        inp_vec = constr_invec(
            test_inputs,
            self.dict_k.sdev.size,
            "aa",
            self.dict_k,
            self.pos_k,
            seq_len_scale=1.0,
        )
        return self.__add_bias__(inp_vec.outer_inner(feat_vec))


@dataclass
class RidgeRegression(gpx.Module):
    ip: AbstractInducingPoints = param_field(None)
    beta: Float[Array, "N"] = param_field(None)

    @property
    def features(self):
        yield from self.ip.features

    def predict(self, test_inputs, train_data: Dataset):
        K = self.ip.cross_covariance(test_inputs, train_data)
        return K @ self.beta


class MSEObjectiveIgnoreNan(gpxobj.AbstractObjective):
    def step(self, model: Module, data: Dataset):
        pred = model.predict(data.X, data)
        if not pred.shape == data.y.shape:
            raise ValueError(
                f"Shape of prediction {pred.shape} does not match shape of target {data.y.shape}"
            )
        return jnp.sum(
            ((pred * data.observed - data.y * data.observed)) ** 2
        ) / jnp.sum(data.observed)


class RankObjectiveIgnoreNan(gpxobj.AbstractObjective):
    def step(self, model: Module, data: Dataset):
        pred = model.predict(data.X, data)
        if not pred.shape == data.y.shape:
            raise ValueError(
                f"Shape of prediction {pred.shape} does not match shape of target {data.y.shape}"
            )
        corr = 0.0
        for i in range(pred.shape[1]):
            corr += (
                jnp.corrcoef(
                    pred[data.observed[:, i], i],
                    jnp.sort(jnp.sort(data.y[data.observed[:, i], i])),
                )[0, 1]
            ) / pred.shape[1]
        return -corr


def test_simple_inducing():
    import optax as ox
    import jax.random as random
    import jax.numpy as jnp
    import gpjax as gpx
    import matplotlib.pyplot as plt

    rng = random.PRNGKey(0)
    D = gpx.Dataset(
        jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        jnp.array([[1.0, jnp.nan], [3.0, 4.0]]),
    )
    model = gpx.RidgeRegression(
        ip=gpx.InducingPoints(
            include_bias=True,
            inducing_points=jnp.array(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [7.0, 8.0, 9.0]]
            )
            + 1,
        ),
        beta=jnp.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [3.0, 4.0],
                [3.0, 4.0],
                [3.0, 4.0],
            ]
        ),
    )
    objective = gpx.MSEObjectiveIgnoreNan()

    optimized_model, loss_history = gpx.fit(
        model=model,
        objective=objective,
        train_data=D,
        optim=ox.adam(learning_rate=1e-2),
        num_iters=1000,
        safe=True,
        key=rng,
    )
    print(optimized_model.predict(D.X, D), "MSE:", loss_history[-1])
    print(
        "inducing points shape:",
        optimized_model.ip.inducing_points.shape,
        "Cross gram shape:",
        optimized_model.ip.cross_covariance(D.X, None).shape,
        "beta shape:",
        optimized_model.beta.shape,
    )
    plt.plot(loss_history)
    # plt.show()


def test_1hot_inducing():
    import pedata as ped
    import optax as ox
    import jax.random as random
    import jax.numpy as jnp
    import gpjax as gpx
    from gpjax.kernels import CatKernel
    import matplotlib.pyplot as plt
    import jaxopt as jo

    a, b = ped.load_similarity("aa", "BLOSUM62")
    # init_p = CatKernel.gram_to_stddev_cholesky_lower(b @ b.T)
    # init_p = CatKernel.gram_to_stddev_cholesky_lower(b + jnp.eye(len(b)) * 40)
    init_p = CatKernel.gram_to_sdev_cholesky_lower(jnp.eye(len(b)) * 2)
    aa_dict_k = CatKernel(
        stddev=init_p.sdev,
        cholesky_lower=init_p.cholesky_lower,
    )

    rng = random.PRNGKey(0)
    y = jnp.array([[1.0, jnp.nan], [3.0, 4.0]])
    D = gpx.Dataset(
        {"aa_1hot": jnp.vstack([jnp.eye(21).flatten()] * 2)},
        y,
        ~jnp.isnan(y),
    )
    inducing_point = gpx.MultiPositionwiseInducing(
        include_bias=True,
        num_pos=21,
        dict_k=[aa_dict_k],
    )
    model = gpx.RidgeRegression(
        ip=inducing_point,
        beta=jnp.ones((len(inducing_point), 2)),
    )
    objective = gpx.MSEObjectiveIgnoreNan()  # RankObjectiveIgnoreNan()

    if False:
        optimized_model, loss_history = gpx.fit(
            model=model,
            objective=objective,
            train_data=D,
            optim=ox.adam(learning_rate=1e-5),
            num_iters=1000,
            safe=True,
            key=rng,
        )
        print(optimized_model.predict(D.X, D), "MSE:", loss_history[-1])
        plt.plot(loss_history)
    else:
        optimized_model, opt_step = gpx.fit_jaxopt(
            model=model,
            train_data=D,
            solver=jo.GradientDescent(gpx.jaxopt_objective(objective), stepsize=1e-5),
        )
        print(optimized_model.predict(D.X, D), "MSE:", objective(optimized_model, D))


def test_1hot_real():
    import pedata as ped
    import optax as ox
    import jax.random as random
    import jax.numpy as jnp
    import gpjax as gpx
    from gpjax.kernels import CatKernel
    import gpjax.kernels as gk
    import matplotlib.pyplot as plt
    import jaxopt as jo
    import datasets as ds
    import jax
    import orbax.checkpoint as oc

    dataset_name = (
        "Exazyme/beta_glucosidase_avg_target_Regression_train_validation_test"
    )
    # dataset_name = "Exazyme/CrHydA1"
    df = (
        ds.load_dataset(dataset_name)["train"]
        .with_format("jax")
        .train_test_split(test_size=0.2, seed=0)
    )

    rng = random.PRNGKey(0)
    train_target_name, y = ped.util.get_target(
        df["train"],
    )
    D = gpx.Dataset(
        {
            "aa_1hot": df["train"].with_format("jax")["aa_1hot"],
            "aa_len": df["train"].with_format("jax")["aa_len"][:, None],
        },
        y,
        ~jnp.isnan(y),
    )
    key_dict, key_beta = jax.random.split(rng)
    num_dict_k = 2
    dict_size = 21
    num_outputs = y.shape[1]
    dict_k = []
    for key in jax.random.split(key_dict, num_dict_k):
        k1, k2 = jax.random.split(key)
        sdev = jax.random.beta(k1, 1, 1, shape=(dict_size,))
        L = jax.random.beta(k2, 1, 1, shape=(dict_size, dict_size))
        dict_k.append(gk.CatKernel(stddev=sdev, cholesky_lower=L))
    inducing_point = gpx.MultiPosEmbInducing(
        include_bias=True,
        num_pos_inducing=D.X["aa_1hot"].shape[1] // dict_size,
        num_pos_max=550,
        dict_k=dict_k[:1],
        pos_k=gk.PoweredExponential(
            lengthscale=1.0,
            power=0.5,
        ),
    )
    model = gpx.RidgeRegression(
        ip=inducing_point,
        beta=jax.random.normal(key_beta, shape=(len(inducing_point), num_outputs)),
    )
    objective = jax.jit(gpx.MSEObjectiveIgnoreNan())  # RankObjectiveIgnoreNan()

    if True:
        optimized_model, loss_history = gpx.fit(
            model=model,
            objective=objective,
            train_data=D,
            optim=ox.sgd(
                ox.linear_schedule(1e-4, 1e-5, 5000, 25000),
                momentum=0.2,
                nesterov=False,
                accumulator_dtype=None,
            ),
            # ox.adam(learning_rate=ox.linear_schedule(1e-1, 1e-4, 5000, 25000)),
            num_iters=30000,
            safe=True,
            key=rng,
        )
        cptr = oc.PyTreeCheckpointer()
        print(optimized_model.predict(D.X, D), "MSE:", loss_history[-1])
        plt.plot(loss_history)
    else:
        optimized_model2, opt_step = gpx.fit_jaxopt(
            model=model,
            train_data=D,
            solver=jo.NonlinearCG(
                gpx.jaxopt_objective(objective),
            ),
        )
        print(optimized_model2.predict(D.X, D), "MSE:", objective(optimized_model2, D))
