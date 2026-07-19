import numpy as np

from .tensor import Tensor


def binary_cross_entropy_with_logits(x: Tensor, y: np.ndarray) -> Tensor:
    N = x.data.shape[0]
    z = x.data
    # stable form: max(z,0) - z*y + log(1 + e^{-|z|})
    loss = np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z))))
    out = Tensor(loss, children={x})

    def _backward():
        p = 1 / (1 + np.exp(-z))  # sigmoid, uniquement pour le gradient
        x.grad += (p - y) / N

    out.set_backward(_backward)
    return out


def softmax_cross_entropy_with_logits(x: Tensor, y: np.ndarray) -> Tensor:
    N = x.data.shape[0]

    e = np.exp(x.data - np.max(x.data, axis=-1, keepdims=True))
    p = e / np.sum(e, axis=-1, keepdims=True)

    out = Tensor(-np.mean(np.log(p[np.arange(N), y.ravel()])), children={x})

    def _backward():
        grad = p.copy()
        grad[np.arange(N), y.ravel()] -= 1
        x.grad += grad / N

    out.set_backward(_backward)
    return out


def mse(y: Tensor, y_true: np.ndarray) -> Tensor:
    N = y.data.size
    diff = y.data - y_true

    out = Tensor(np.mean(np.pow(diff, 2)), children={y})

    def _backward():
        y.grad += (2 / N) * (diff) * out.grad

    out.set_backward(_backward)
    return out
