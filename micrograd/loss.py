import numpy as np

from .tensor import Tensor


def binary_cross_entropy_with_logits(x: Tensor, y: np.ndarray) -> Tensor:
    N = x.data.shape[0]
    p = 1 / (1 + np.exp(-x.data))  # sigmoid

    out = Tensor(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)), (x,))

    def _backward():
        x.grad += (p - y) / N

    out.set_backward(_backward)

    return out
