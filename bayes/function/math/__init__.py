from bayes.function.math.add import add
from bayes.function.math.divide import divide, rdivide
from bayes.function.math.exp import exp
from bayes.function.math.log import log
from bayes.function.math.matmul import matmul, rmatmul
from bayes.function.math.mean_squared_error import mean_squared_error
from bayes.function.math.mean import mean
from bayes.function.math.multiply import multiply
from bayes.function.math.negative import negative
from bayes.function.math.power import power, rpower
from bayes.function.math.sqrt import sqrt
from bayes.function.math.square import square
from bayes.function.math.subtract import subtract, rsubtract
from bayes.function.math.sum_squared_error import sum_squared_error
from bayes.function.math.sum import sum


from bayes.tensor.tensor import Tensor
Tensor.__add__ = add
Tensor.__radd__ = add
Tensor.__truediv__ = divide
Tensor.__rtruediv__ = rdivide
Tensor.mean = mean
Tensor.__matmul__ = matmul
Tensor.__rmatmul__ = rmatmul
Tensor.__mul__ = multiply
Tensor.__rmul__ = multiply
Tensor.__neg__ = negative
Tensor.__pow__ = power
Tensor.__rpow__ = rpower
Tensor.__sub__ = subtract
Tensor.__rsub__ = rsubtract
Tensor.sum = sum


__all__ = [
    "add",
    "divide",
    "exp",
    "log",
    "matmul",
    "mean_squared_error",
    "mean",
    "multiply",
    "power",
    "sqrt",
    "square",
    "subtract",
    "sum_squared_error",
    "sum"
]
