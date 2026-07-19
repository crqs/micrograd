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
from micrograd.loss import softmax_cross_entropy_with_logits
from micrograd.nn import Attention, AttentionScoreSummed, Linear, LogitsBinaryMask, Module
from micrograd.operations import softmax
from micrograd.optim import Adam, CosineDecayScheduler

SEED = 42

np.random.seed(SEED)
random.seed(SEED)


class MaxClassificationModel(Module):
    """Predicts the position of the max in a sequence"""

    def __init__(
        self,
        n_in: int,
        d_k: int,
        high: int,
    ) -> None:
        super().__init__()

        self.high = float(high)
        self.attention_sum = AttentionScoreSummed(n_in, d_k)

    @property
    def children(self) -> list[Module]:
        return [self.attention_sum]

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

        return self.attention_sum(x, mask)

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
        d_k=32,
        high=HIGH,
    )
    print(model)

    nb_epochs = 100

    results = fit(
        model=model,
        optimizer=(optimizer := Adam(parameters=model.parameters, lr=1e-2)),
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
