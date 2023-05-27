from dataclasses import dataclass
from .base import AbstractKernel
from ..rkhs.base import AbstractEncoder
from ..rkhs.encoder import StandardEncoder
from .stationary import RBF
from ..typing import ScalarFloat
from gpjax.base import param_field
from beartype.typing import Any


@dataclass
class EncoderKernel(AbstractKernel):
    enc: AbstractEncoder = param_field(StandardEncoder(RBF()))

    def __call__(
        self,
        x: Any,
        y: Any,
    ) -> ScalarFloat:
        r"""Evaluate the kernel on a pair of inputs.

        Args:
            x (Any): The left hand input of the kernel function.
            y (Any): The right hand input of the kernel function.

        Returns
        -------
            ScalarFloat: The evaluated kernel function at the supplied inputs.
        """
        # print(x.shape, y.shape, x[..., None].shape)
        return self.enc(x[..., None]).outer_inner(self.enc(y[..., None])).squeeze()
