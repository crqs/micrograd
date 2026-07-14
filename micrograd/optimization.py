import numpy as np

from .tensor import Tensor


class SGD:
    def __init__(self, parameters: list[Tensor], lr: float):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self) -> None:
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)

    def step(self) -> None:
        for parameters in self.parameters:
            parameters.data -= self.lr * parameters.grad
