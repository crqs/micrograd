from __future__ import annotations

import random
from collections.abc import Generator
from contextlib import contextmanager
from itertools import pairwise
from typing import cast

import numpy as np
from sklearn.model_selection import train_test_split

from micrograd import Tensor
from micrograd.loss import softmax_cross_entropy_with_logits
from micrograd.nn import Attention, Linear, LogitsBinaryMask, Module
from micrograd.operations import softmax
from micrograd.optim import Adam, CosineDecayScheduler

from .utils import fit, plot_results

SEED = 42

np.random.seed(SEED)
random.seed(SEED)


def positional_encoding(max_seq_len: int, d_pos: int, base: float | None = None) -> np.ndarray:
    """Multi-frequency sinusoidal positional encoding"""
    if d_pos % 2 != 0:
        raise ValueError("d_pos must be even (pairs of sin/cos)")
    if base is None:
        base = max(max_seq_len, 2)

    positions = np.arange(max_seq_len).reshape(-1, 1)  # (max_seq_len, 1)
    i = np.arange(d_pos // 2).reshape(1, -1)  # (1, d_pos/2)
    freqs = 1.0 / (base ** (2 * i / d_pos))  # (1, d_pos/2)
    angles = positions * freqs  # (max_seq_len, d_pos/2)

    pe = np.zeros((max_seq_len, d_pos))
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)
    return pe


def make_dataset(
    n_samples: int,
    high: int,
    min_seq_len: int,
    max_seq_len: int,
    d_pos: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate sequences of variable length using masking, with a multi-frequency
    sinusoidal positional encoding concatenated to each token's value.

    Each token has shape (1 + d_pos,): its raw value, followed by d_pos positional
    encoding dimensions (pairs of sin/cos at different frequencies), so the attention
    mechanism can discriminate between positions, not just between values.

    Returns:
        X: (n_samples, max_seq_len, 1 + d_pos) — token features, padded with zeros.
        y: (n_samples, 1) — index of the max value within each sequence.
        mask: (n_samples, max_seq_len) — 1.0 for valid positions, 0.0 for padding.
    """
    X = np.random.randint(low=0, high=high, size=(n_samples, max_seq_len, 1 + d_pos)).astype(float)
    seq_len = np.random.randint(low=min_seq_len, high=max_seq_len, size=n_samples)  # (n_samples,)
    positions = np.arange(max_seq_len)  # (max_seq_len,)

    pe = positional_encoding(max_seq_len, d_pos)  # (max_seq_len, d_pos)
    X[:, :, 1:] = pe[None, :, :]  # broadcast the same positional encoding to every sample

    mask = positions[None, :] < seq_len[:, None]  # (n_samples, max_seq_len)

    X[~mask] = 0
    y = np.argmax(X[:, :, 0], axis=1).reshape(-1, 1)

    return X, y, mask.astype(np.float32)


class MaxClassificationModel(Module):
    """Predicts the position of the max in a sequence"""

    def __init__(
        self,
        n_in: int,
        d_k: int,
        d_v: int,
        hidden_dims: list[int],
        out_dim: int,
        high: int,
        max_seq_len: int,
    ) -> None:
        super().__init__()

        self.high = float(high)
        self.max_seq_len = float(max_seq_len)

        self.attention = Attention(n_in, d_k, d_v)

        self.linears: list[Linear] = []
        self.linears.append(Linear(d_v, hidden_dims[0], activation=None))
        for n_in, n_out in pairwise(hidden_dims):
            self.linears.append(Linear(n_in, n_out, activation="relu"))
        self.linears.append(Linear(hidden_dims[-1], out_dim, activation=None))

        self.mask = LogitsBinaryMask()

    @property
    def children(self) -> list[Module]:
        return [self.attention, *self.linears]

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
        # normalization
        x.data[:, :, 0] /= self.high

        x = self.attention(x, mask)
        for linear in self.linears:
            x = linear(x)
        return self.mask(x, mask).squeeze(-1)

    def predict(self, x: Tensor, mask: np.ndarray) -> np.ndarray:
        """
        The model outputs raw logits. We need to transform them in probas through
        softmax and take the most probable class.
        """
        return np.argmax(softmax(self.__call__(x, mask)).data, axis=-1)

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

    model = MaxClassificationModel(
        n_in=D_POS + 1,
        d_k=16,
        d_v=16,
        hidden_dims=[16],
        out_dim=1,
        high=HIGH,
        max_seq_len=MAX_SEQ_LEN,
    )
    print(model)

    nb_epochs = 100

    results = fit(
        model=model,
        optimizer=(optimizer := Adam(parameters=model.parameters, lr=1e-3)),
        lr_scheduler=CosineDecayScheduler(optimizer, nb_epochs),
        criterion=softmax_cross_entropy_with_logits,
        nb_epochs=nb_epochs,
        batch_size=32,
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
        task="classification",
        save_path="results_max_classification.png",
    )
