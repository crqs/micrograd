import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from examples.train import fit
from micrograd import Tensor
from micrograd.loss import softmax_cross_entropy_with_logits
from micrograd.nn import MLP

np.random.seed(42)
random.seed(42)


def plot_results(
    losses: np.ndarray,
    val_losses: np.ndarray,
    sample_images: np.ndarray,
    sample_preds: np.ndarray,
    sample_labels: np.ndarray,
) -> None:
    _, (ax1, _) = plt.subplots(1, 2, figsize=(12, 4), dpi=150)

    # loss curves
    ax1.plot(losses, label="train loss")
    ax1.plot(val_losses, label="val loss", linestyle="--")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # sample predictions — grille 2x5
    fig2, axes = plt.subplots(2, 5, figsize=(10, 4), dpi=150)
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(sample_images[i].reshape(28, 28), cmap="gray")
        color = "green" if sample_preds[i] == sample_labels[i] else "red"
        ax.set_title(f"pred={sample_preds[i]} true={sample_labels[i]}", color=color, fontsize=8)
        ax.axis("off")
    fig2.suptitle("Sample predictions")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # load mnist
    print("Loading MNIST...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="liac-arff")
    X = X / 255.0  # normalize to [0, 1]
    y = LabelEncoder().fit_transform(y).astype(int)  # type: ignore

    # split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

    X_train = np.array(X_train)
    y_train = np.array(y_train).reshape(X_train.shape[0], 1)

    X_test = np.array(X_test)
    y_test = np.array(y_test).reshape(X_test.shape[0], 1)

    X_val = np.array(X_val)
    y_val = np.array(y_val).reshape(X_val.shape[0], 1)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    model = MLP(784, [256, 128], 10)
    print(f"Number of parameters: {model.nb_parameters}")

    results = fit(
        model=model,
        criterion=softmax_cross_entropy_with_logits,
        nb_epochs=20,
        batch_size=64,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        lr=1e-2,
    )

    # sample predictions on 10 exemples from test set
    logits = model(Tensor(X_test[:10]))
    preds = np.argmax(logits.data, axis=-1)

    plot_results(
        losses=results.loss,
        val_losses=results.val_loss,
        sample_images=X_test[:10],
        sample_preds=preds,
        sample_labels=y_test[:10].ravel(),
    )
