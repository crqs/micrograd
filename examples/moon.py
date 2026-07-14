import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from examples.train import fit_with_sgd
from micrograd import Tensor
from micrograd.nn import MLP

np.random.seed(42)
random.seed(42)


def plot_results(
    model: MLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    losses: np.ndarray,
    val_losses: np.ndarray,
) -> None:
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

    # decision boundary
    X = np.vstack([X_train, X_val, X_test])
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    logits = model(Tensor(grid))
    probs = 1 / (1 + np.exp(-logits.data))
    probs = probs.reshape(xx.shape)

    ax1.contourf(xx, yy, probs, levels=50, cmap="RdBu", alpha=0.8)
    # train — circles
    ax1.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="RdBu", edgecolors="k", s=20, marker="o", label="train")
    # val — squares
    ax1.scatter(X_val[:, 0], X_val[:, 1], c=y_val, cmap="RdBu", edgecolors="yellow", s=30, marker="s", label="val")
    # test — triangles
    ax1.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="RdBu", edgecolors="green", s=30, marker="^", label="test")
    ax1.legend()
    ax1.set_title("Decision boundary")

    # losses
    ax2.plot(losses, label="train loss")
    ax2.plot(val_losses, label="val loss", linestyle="--")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("results.png")
    plt.show()


if __name__ == "__main__":
    n_samples = 500

    X, y = make_moons(n_samples=n_samples, noise=0.1)

    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.3)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5)

    X_train = np.array(X_train)
    y_train = np.array(y_train).reshape(X_train.shape[0], 1)

    X_test = np.array(X_test)
    y_test = np.array(y_test).reshape(X_test.shape[0], 1)

    X_val = np.array(X_val)
    y_val = np.array(y_val).reshape(X_val.shape[0], 1)

    model = MLP(2, [32, 32], 1)
    # print(model)
    print(f"Number of parameters: {model.nb_parameters}")

    results = fit_with_sgd(
        model=model,
        nb_epochs=2_000,
        batch_size=64,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        lr=1e-2,
    )

    plot_results(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        losses=results.loss,
        val_losses=results.val_loss,
    )
