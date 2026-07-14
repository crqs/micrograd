from abc import ABC, abstractmethod
from itertools import pairwise
from typing import Literal

import numpy as np

from .tensor import Tensor


class Module(ABC):
    @property
    @abstractmethod
    def parameters(self) -> list[Tensor]: ...

    @property
    def nb_parameters(self) -> int:
        return sum(p.data.size for p in self.parameters)

    def zero_grad(self) -> None:
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)


class Linear(Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        activation: Literal["relu"] | None = None,
    ) -> None:
        # Kaiming init for ReLU activation
        self.w = Tensor(np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in))
        self.b = Tensor(np.zeros(n_out))
        self.activation = activation

    def __call__(self, x: Tensor) -> Tensor:
        y = x @ self.w + self.b
        match self.activation:
            case "relu":
                return y.relu()
            case None:
                return y
            case _:
                raise ValueError(f"Unknown activation: {self.activation}")

    @property
    def parameters(self) -> list[Tensor]:
        return [self.w, self.b]

    def __repr__(self) -> str:
        return f"Linear({self.w.shape}, {self.b.shape})"


class MLP(Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        activation: Literal["relu"] | None = "relu",
    ) -> None:

        if len(hidden_dims) < 1:
            raise ValueError("hidden_dims must be a non-empty list of integers")

        self.layers: list[Linear] = [
            Linear(n_in, n_out, activation=activation) for n_in, n_out in pairwise([input_dim, *hidden_dims])
        ]
        self.layers.append(Linear(hidden_dims[-1], out_dim, activation=None))

    def __call__(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def parameters(self) -> list[Tensor]:
        return [p for layer in self.layers for p in layer.parameters]

    def __repr__(self) -> str:
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
