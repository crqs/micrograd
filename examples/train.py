from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

import micrograd.nn as nn
from micrograd import Tensor
from micrograd.optim import Optimizer


@dataclass
class Results:
    loss: np.ndarray
    val_loss: np.ndarray


def fit(
    model: nn.MLP,
    optimizer: Optimizer,
    criterion: Callable[[Tensor, np.ndarray], Tensor],
    nb_epochs: int,
    batch_size: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Results:

    nb_samples = X_train.shape[0]
    nb_batches = nb_samples // batch_size + (nb_samples % batch_size > 0)

    results = Results(
        loss=np.zeros(nb_epochs),
        val_loss=np.zeros(nb_epochs),
    )

    for epoch in (pbar := tqdm(range(nb_epochs), initial=1)):
        indices = np.random.permutation(nb_samples)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        batch = 0
        for _ in range(nb_batches):
            X_batch = X_train_shuffled[batch : batch + batch_size, :]
            y_batch = y_train_shuffled[batch : batch + batch_size, :]

            optimizer.zero_grad()

            logits = model(Tensor(X_batch))

            loss = criterion(logits, y_batch)
            loss.backward()

            optimizer.step()

            batch += batch_size

        with model.eval():
            results.loss[epoch] = float(loss := criterion(model(Tensor(X_train)), y_train).data)
            results.val_loss[epoch] = float(val_loss := criterion(model(Tensor(X_val)), y_val).data)

        pbar.set_postfix(loss=f"{loss:.4f}", val_loss=f"{val_loss:.4f}")

    return results
