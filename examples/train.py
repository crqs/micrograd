from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

import micrograd.nn as nn
from micrograd import Tensor
from micrograd.loss import binary_cross_entropy_with_logits
from micrograd.optimization import SGD


@dataclass
class Results:
    loss: np.ndarray
    val_loss: np.ndarray


def fit_with_sgd(
    model: nn.MLP,
    nb_epochs: int,
    batch_size: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lr: float = 1e-4,
) -> Results:
    sgd = SGD(model.parameters, lr=lr)

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

            sgd.zero_grad()

            logits = model(Tensor(X_batch))

            loss = binary_cross_entropy_with_logits(logits, y_batch)
            loss.backward()

            pbar.set_postfix(loss=f"{loss.data:.4f}")

            sgd.step()

            batch += batch_size

        results.loss[epoch] = float(binary_cross_entropy_with_logits(model(Tensor(X_train)), y_train).data)
        results.val_loss[epoch] = float(binary_cross_entropy_with_logits(model(Tensor(X_val)), y_val).data)

    return results
