from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from itertools import pairwise
from typing import Literal

import numpy as np

from .operations import softmax
from .tensor import Tensor


class Module(ABC):
    training: bool = True

    @property
    @abstractmethod
    def parameters(self) -> list[Tensor]: ...

    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor: ...

    @property
    def nb_parameters(self) -> int:
        return sum(p.data.size for p in self.parameters)

    def zero_grad(self) -> None:
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)


class Linear(Module):
    def __init__(self, n_in: int, n_out: int, activation: Literal["relu"] | None) -> None:
        super().__init__()
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
        return f"Linear({self.w.shape}, {self.b.shape}, activation={self.activation})"


class Attention(Module):
    def __init__(self, n_in: int, d_k: int, d_v: int) -> None:
        super().__init__()
        self.d_k = d_k
        self.w_k = Tensor(np.random.randn(n_in, d_k))
        self.w_q = Tensor(np.random.randn(n_in, d_k))
        self.w_v = Tensor(np.random.randn(n_in, d_v))

    def __call__(self, x: Tensor) -> Tensor:
        # x (seq_len, n_in)
        Q = x @ self.w_q  # (seq_len, d_k)
        K = x @ self.w_k  # (seq_len, d_k)
        V = x @ self.w_v  # (seq_len, d_v)

        return softmax(Q @ K.T / np.sqrt(self.d_k)) @ V  # (seq_len, d_v)

    @property
    def parameters(self) -> list[Tensor]:
        return [self.w_k, self.w_q, self.w_v]


class Dropout(Module):
    """
    Dropout module implemented from https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf
    """

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = dropout

    def __call__(self, x: Tensor) -> Tensor:
        if self.training:
            p = np.random.rand(*x.shape) > self.dropout

            # mask data + apply inverted dropout scaling
            out = Tensor(x.data * p / (1 - self.dropout), children={x})

            def _backward():
                x.grad += out.grad * p / (1 - self.dropout)

            out.set_backward(_backward)
            return out

        return x

    @property
    def parameters(self) -> list[Tensor]:
        return []

    def __repr__(self) -> str:
        return f"Dropout({self.dropout})"


class LayerNorm(Module):
    """
    LayerNorm implemented from https://arxiv.org/pdf/1607.06450
    """

    # TODO:


class MLP(Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        dropout: float,
        activation: Literal["relu"] | None = "relu",
    ) -> None:
        super().__init__()

        if len(hidden_dims) < 1:
            raise ValueError("hidden_dims must be a non-empty list of integers")

        self.layers: list[Module] = []
        self.layers.append(Linear(input_dim, hidden_dims[0], activation=activation))

        for n_in, n_out in pairwise(hidden_dims):
            if dropout > 0.0:
                self.layers.append(Dropout(dropout=dropout))
            self.layers.append(Linear(n_in, n_out, activation=activation))

        if dropout > 0.0:
            self.layers.append(Dropout(dropout=dropout))
        self.layers.append(Linear(hidden_dims[-1], out_dim, activation=None))

    @contextmanager
    def eval(self) -> Generator[None]:
        self.training = False
        for layer in self.layers:
            layer.training = False
        yield
        self.training = True
        for layer in self.layers:
            layer.training = True

    def __call__(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def parameters(self) -> list[Tensor]:
        return [p for layer in self.layers for p in layer.parameters]

    def __repr__(self) -> str:
        return f"MLP ({self.nb_parameters} parameters)\n    {'\n    '.join(str(layer) for layer in self.layers)}]"
