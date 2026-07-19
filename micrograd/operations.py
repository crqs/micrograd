import numpy as np

from .tensor import Tensor


def log(x: Tensor) -> Tensor:
    out = Tensor(np.log(x.data), children={x})

    def _backward():
        x.grad += out.grad * x.data ** (-1)

    out.set_backward(_backward)

    return out


def sigmoid(x: Tensor) -> Tensor:
    out = Tensor(1 / (1 + np.exp(-x.data)), children={x})

    def _backward():
        x.grad += out.grad * out.data * (1 - out.data)

    out.set_backward(_backward)

    return out


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    e = np.exp(x.data - np.max(x.data, axis=axis, keepdims=True))
    out = Tensor(e / np.sum(e, axis=axis, keepdims=True), children={x})

    def _backward():
        dot = np.sum(out.grad * out.data, axis=-1, keepdims=True)
        x.grad += out.data * (out.grad - dot)

    out.set_backward(_backward)

    return out
