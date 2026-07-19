from __future__ import annotations

import random
from collections.abc import Generator
from contextlib import contextmanager
from itertools import pairwise
from typing import cast

import numpy as np
from sklearn.model_selection import train_test_split

from examples.attention.utils import fit, make_dataset, plot_results
from micrograd import Tensor
from micrograd.loss import mse
from micrograd.nn import Attention, Linear, MeanPool, Module
from micrograd.optim import Adam, CosineDecayScheduler

SEED = 42

np.random.seed(SEED)
random.seed(SEED)


class MaxRegressionModel(Module):
    """Predicts the value of the max in a sequence"""

    def __init__(
        self,
        n_in: int,
        d_k: int,
        d_v: int,
        hidden_dims: list[int],
        out_dim: int,
    ) -> None:
        super().__init__()

        self.attention = Attention(n_in, d_k, d_v)
        self.mean_pool = MeanPool()

        self.linears: list[Linear] = []
        self.linears.append(Linear(d_v, hidden_dims[0], activation=None))
        for n_in, n_out in pairwise(hidden_dims):
            self.linears.append(Linear(n_in, n_out, activation="relu"))
        self.linears.append(Linear(hidden_dims[-1], out_dim, activation=None))

    @property
    def children(self) -> list[Module]:
        return [self.attention, self.mean_pool, *self.linears]

    @contextmanager
    def eval(self) -> Generator[None]:
        self.training = False
        for child in self.children:
            child.training = False
        yield
        self.training = True
        for child in self.children:
            child.training = True

    def __call__(self, x: Tensor, mask: np.ndarray) -> Tensor:  # type: ignore
        x = self.attention(x, mask)
        x = self.mean_pool(x, mask)
        for linear in self.linears:
            x = linear(x)
        return x

    @property
    def parameters(self) -> list[Tensor]:
        return [p for layer in self.children for p in layer.parameters]

    def __repr__(self) -> str:
        return (
            f"MaxModel ({self.nb_parameters} parameters)\n    {'\n    '.join(str(child) for child in self.children)}]"
        )


if __name__ == "__main__":
    MAX_SEQ_LEN = 20
    MIN_SEQ_LEN = 5
    HIGH = 100
    D_POS = 2

    X, y, mask = make_dataset(
        n_samples=1_000,
        high=HIGH,
        min_seq_len=MIN_SEQ_LEN,
        max_seq_len=MAX_SEQ_LEN,
        d_pos=D_POS,
        mode="max",
    )

    X_train, X_test_val, y_train, y_test_val, mask_train, mask_test_val = train_test_split(X, y, mask, test_size=0.3)
    X_test, X_val, y_test, y_val, mask_test, mask_val = train_test_split(
        X_test_val, y_test_val, mask_test_val, test_size=0.5
    )

    X_train = cast(np.ndarray, X_train)
    X_val = cast(np.ndarray, X_val)
    X_test = cast(np.ndarray, X_test)
    y_train = cast(np.ndarray, y_train)
    y_val = cast(np.ndarray, y_val)
    y_test = cast(np.ndarray, y_test)
    mask_train = cast(np.ndarray, mask_train)
    mask_val = cast(np.ndarray, mask_val)
    mask_test = cast(np.ndarray, mask_test)

    model = MaxRegressionModel(
        n_in=1 + D_POS,
        d_k=16,
        d_v=16,
        hidden_dims=[16, 16],
        out_dim=1,
    )
    print(model)

    nb_epochs = 200

    results = fit(
        model=model,
        optimizer=(optimizer := Adam(parameters=model.parameters, lr=1e-3)),
        lr_scheduler=None,
        criterion=mse,
        nb_epochs=nb_epochs,
        batch_size=64,
        X_train=X_train,
        y_train=y_train,
        mask_train=mask_train,
        X_val=X_val,
        y_val=y_val,
        mask_val=mask_val,
    )
    plot_results(
        model=model,
        X_test=X_test,
        y_test=y_test,
        mask_test=mask_test,
        losses=results.loss,
        val_losses=results.val_loss,
        task="regression",
        save_path="results_max_regression.png",
    )
