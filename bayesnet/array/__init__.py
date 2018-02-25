from bayesnet.array.broadcast import broadcast_to
from bayesnet.array.flatten import flatten
from bayesnet.array.reshape import reshape, reshape_method
from bayesnet.array.split import split
from bayesnet.array.swapaxes import swapaxes
from bayesnet.array.transpose import transpose, transpose_method
from bayesnet.tensor.tensor import Tensor


Tensor.flatten = flatten
Tensor.reshape = reshape_method
Tensor.transpose = transpose_method

__all__ = [
    "broadcast_to",
    "flatten",
    "reshape",
    "split",
    "swapaxes",
    "transpose"
]
