from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any
from jaxtyping import Float, Array, Int
from beartype.typing import Union, Callable, List, TypeVar, Tuple
import jax.numpy as np
from gpjax.base.param import param_field
from gpjax.typing import ScalarInt, ScalarFloat, ScalarBool


from ..base import (
    ReduceableOrArray,
    NumberOrArray,
    AbstractReduce,
    NoReduce,
    ChainedReduce,
    LinearizableReduce,
    LinearReduce,
)
