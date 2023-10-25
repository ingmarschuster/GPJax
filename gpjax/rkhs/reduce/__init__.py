# import all reduce classes from .base
from .base import (
    ChainedReduce,
    LinearReduce,
    LinearizableReduce,
)

from .simple import NoReduce, Prefactors, Scale, Sum, Mean, BalancedRed, Center


from .repetitions import SparseReduce, Repeat, TileView, MovingAverage, Kmer
