from __future__ import annotations

from collections.abc import Callable

import numpy as np


class Tensor:
    """
    Wraps a tensor (numpy array) and supports automatic differentiation by gradient accumulation.
    Each Tensor object keeps track of its data, gradient, and the operation that produced it.
    The class supports basic arithmetic operations, matrix multiplication, power, and ReLU activation, along with a
    backward method to compute gradients through the computational graph.
    """

    def __init__(
        self,
        data: float | list[float] | np.ndarray,
        children: tuple[Tensor, ...] = (),
        op: str = "",
    ) -> None:
        self.data = np.array(data)
        self.grad = np.zeros_like(data, dtype=float)

        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self.children = set(children)
        self.op = op  # the op that produced this node, for graphviz / debugging / etc

    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def set_backward(self, backward: Callable[[], None]) -> None:
        self._backward = backward

    def _unbroadcast(self, grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)

        for i, s in enumerate(shape):
            if s == 1:
                grad = grad.sum(axis=i, keepdims=True)

        return grad

    def __add__(self, other: Tensor | int | float) -> Tensor:

        if isinstance(other, (int, float)):
            other = Tensor(np.array(other))

        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += self._unbroadcast(out.grad, self.grad.shape)
            other.grad += self._unbroadcast(out.grad, other.grad.shape)

        out.set_backward(_backward)

        return out

    def __mul__(self, other: Tensor | int | float) -> Tensor:

        if isinstance(other, (int, float)):
            other = Tensor(np.array(other))

        out = Tensor(np.multiply(self.data, other.data), (self, other), "*")

        def _backward():
            self.grad += other.data * self._unbroadcast(out.grad, self.grad.shape)
            other.grad += self.data * self._unbroadcast(out.grad, other.grad.shape)

        out.set_backward(_backward)

        return out

    def __matmul__(self, other: Tensor) -> Tensor:
        out = Tensor(np.matmul(self.data, other.data), (self, other), "@")

        def _backward():
            self.grad += np.matmul(out.grad, other.data.T)
            other.grad += np.matmul(self.data.T, out.grad)

        out.set_backward(_backward)

        return out

    def __pow__(self, n: int | float) -> Tensor:

        out = Tensor(np.pow(self.data, n), (self,), f"**{n}")

        def _backward():
            self.grad += n * out.grad * np.power(self.data, n - 1)

        out.set_backward(_backward)

        return out

    def relu(self) -> Tensor:
        out = Tensor(np.maximum(self.data, 0), (self,), "ReLU")

        def _backward():
            self.grad += np.where(out.data > 0, out.grad, 0)

        out.set_backward(_backward)

        return out

    def backward(self) -> None:

        # topological order all of the children in the graph
        topo = []
        visited = set()

        def build_topo(tensor: Tensor):
            if tensor not in visited:
                visited.add(tensor)
                for child in tensor.children:
                    build_topo(child)
                topo.append(tensor)

        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = np.array(1.0)
        for v in reversed(topo):
            v._backward()

    def __neg__(self):
        return self * Tensor(-1)

    def __radd__(self, other: Tensor):
        return self + other

    def __sub__(self, other: Tensor):
        return self + (-other)

    def __rsub__(self, other: Tensor):
        return other + (-self)

    def __rmul__(self, other: Tensor):
        return self * other

    def __truediv__(self, other: Tensor):
        return self * other**-1

    def __rtruediv__(self, other: Tensor):
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
