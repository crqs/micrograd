from __future__ import annotations

import random
from collections.abc import Callable, Generator
from contextlib import contextmanager
from itertools import pairwise
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from examples.train import Results
from micrograd import Tensor
from micrograd.loss import softmax_cross_entropy_with_logits
from micrograd.nn import Attention, Dropout, Linear, LogitsBinaryMask, MeanPool, Module
from micrograd.operations import softmax
from micrograd.optim import Adam, Optimizer

np.random.seed(42)
random.seed(42)


def plot_results(
    model: MaxClassificationModel | MaxRegressionModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
    mask_test: np.ndarray,
    losses: np.ndarray,
    val_losses: np.ndarray,
    n_examples: int = 8,
) -> None:
    fig = plt.figure(figsize=(15, 10), dpi=150)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.4], height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, 0])  # loss
    ax3 = fig.add_subplot(gs[1, 0])  # confusion matrix
    ax2 = fig.add_subplot(gs[:, 1])  # example sequences (spans both rows)

    # ---- losses ----
    epochs = np.arange(len(losses))
    ax1.plot(epochs, losses, label="train loss", linewidth=1.8)
    ax1.plot(epochs, val_losses, label="val loss", linestyle="--", linewidth=1.8)
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.3)
    ax1.legend()

    # ---- full-test-set predictions, for reliable accuracy stats ----
    seq_len = X_test.shape[1]
    with model.eval():
        all_preds = model.predict(Tensor(X_test), mask_test)
    all_preds = np.asarray(all_preds).reshape(-1).astype(int)
    all_trues = y_test.reshape(-1).astype(int)

    accuracy = float((all_preds == all_trues).mean())
    most_frequent_idx = int(np.bincount(all_trues, minlength=seq_len).argmax())
    baseline_accuracy = float((all_trues == most_frequent_idx).mean())

    confusion = np.zeros((seq_len, seq_len), dtype=int)
    for t, p in zip(all_trues, all_preds):
        confusion[t, p] += 1

    per_index_accuracy = np.full(seq_len, np.nan)
    for idx in range(seq_len):
        rows = all_trues == idx
        if rows.sum() > 0:
            per_index_accuracy[idx] = (all_preds[rows] == all_trues[rows]).mean()

    # ---- confusion matrix panel ----
    im = ax3.imshow(confusion, cmap="Blues")
    ax3.set_title(
        f"Confusion matrix — acc {accuracy:.3f}  (baseline 'always idx {most_frequent_idx}': {baseline_accuracy:.3f})",
        fontsize=10,
    )
    ax3.set_xlabel("Predicted idx")
    ax3.set_ylabel("True idx")
    ax3.set_xticks(range(seq_len))
    ax3.set_yticks(range(seq_len))
    for t in range(seq_len):
        for p in range(seq_len):
            if confusion[t, p] > 0:
                ax3.text(
                    p,
                    t,
                    str(confusion[t, p]),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if confusion[t, p] > confusion.max() / 2 else "black",
                )
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    # small per-index accuracy annotation under the confusion matrix
    per_idx_str = "  ".join(f"{i}:{a:.2f}" for i, a in enumerate(per_index_accuracy) if not np.isnan(a))
    ax3.text(
        0,
        seq_len + 1.2,
        f"per true-idx accuracy:  {per_idx_str}",
        fontsize=7,
        ha="left",
        transform=ax3.transData,
    )

    # ---- a few example sequences, true vs predicted position ----
    preds = all_preds[:n_examples]
    trues = all_trues[:n_examples]
    order = np.argsort(-trues)

    ax2.set_title("Predicted vs true max position on example sequences")
    ax2.set_xlim(-0.5, seq_len + 3.6)
    ax2.set_ylim(0.3, n_examples + 0.7)
    ax2.set_yticks([])
    ax2.set_xticks(range(seq_len))
    ax2.set_xlabel("Position in sequence")
    for spine in ("top", "right", "left"):
        ax2.spines[spine].set_visible(False)

    for row, i in enumerate(order):
        seq = X_test[i][:, 0]
        y = n_examples - row
        true_pos = trues[i]
        pred_pos = preds[i]
        correct = pred_pos == true_pos

        for j, v in enumerate(seq):
            if not mask_test[i, j]:
                continue
            is_true = j == true_pos
            is_pred = j == pred_pos

            if is_true and is_pred:
                edge_color, lw = "#2e7d32", 3.0
            elif is_true:
                edge_color, lw = "crimson", 2.4
            elif is_pred:
                edge_color, lw = "#1565c0", 2.4
            else:
                edge_color, lw = "black", 0.8

            ax2.scatter(
                j,
                y,
                s=380,
                c="white",
                edgecolors=edge_color,
                linewidths=lw,
                zorder=2,
            )
            ax2.text(
                j,
                y,
                str(int(v)),
                ha="center",
                va="center",
                fontsize=8,
                zorder=3,
                color="black",
                fontweight="bold",
            )

        status_color = "#2e7d32" if correct else "#c62828"
        status = "correct" if correct else "wrong"
        ax2.text(seq_len + 0.3, y, f"true idx {true_pos}", ha="left", va="center", fontsize=9)
        ax2.text(
            seq_len + 2.1,
            y,
            f"pred idx {pred_pos}  ({status})",
            ha="left",
            va="center",
            fontsize=9,
            color=status_color,
        )

    ax2.text(0, n_examples + 1.15, "crimson = true max position", fontsize=8, color="crimson", ha="left")
    ax2.text(0, n_examples + 0.9, "blue = predicted position (if wrong)", fontsize=8, color="#1565c0", ha="left")
    ax2.text(0, n_examples + 0.65, "green = correct prediction", fontsize=8, color="#2e7d32", ha="left")

    plt.tight_layout()
    plt.savefig("results_max.png")
    plt.show()


def positional_encoding(max_seq_len: int, d_pos: int, base: float | None = None) -> np.ndarray:
    """
    Multi-frequency sinusoidal positional encoding (Vaswani et al., 2017), rescaled
    for short sequences: base defaults to max_seq_len instead of the NLP-scale 10000,
    so frequencies are spread out over a range that actually matches the sequence.

    PE(p, 2i)   = sin(p / base^(2i/d_pos))
    PE(p, 2i+1) = cos(p / base^(2i/d_pos))

    Args:
        max_seq_len: number of positions to encode (0 .. max_seq_len-1).
        d_pos: number of encoding dimensions, must be even (pairs of sin/cos).
        base: frequency base. Defaults to max_seq_len (clamped to >= 2).

    Returns:
        Array of shape (max_seq_len, d_pos).
    """
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


def fit(
    model: MaxClassificationModel | MaxRegressionModel,
    optimizer: Optimizer,
    criterion: Callable[[Tensor, np.ndarray], Tensor],
    nb_epochs: int,
    batch_size: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    mask_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    mask_val: np.ndarray,
) -> Results:

    nb_samples = X_train.shape[0]
    nb_batches = nb_samples // batch_size + (nb_samples % batch_size > 0)

    results = Results(loss=np.zeros(nb_epochs), val_loss=np.zeros(nb_epochs))

    for epoch in (pbar := tqdm(range(nb_epochs), initial=1)):
        indices = np.random.permutation(nb_samples)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        mask_train_shuffled = mask_train[indices]

        batch = 0
        for _ in range(nb_batches):
            X_batch = X_train_shuffled[batch : batch + batch_size, :]
            y_batch = y_train_shuffled[batch : batch + batch_size, :]
            mask_batch = mask_train_shuffled[batch : batch + batch_size, :]

            optimizer.zero_grad()

            pred = model(Tensor(X_batch), mask_batch)

            loss = criterion(pred, y_batch)
            loss.backward()

            optimizer.step()

            batch += batch_size

        with model.eval():
            results.loss[epoch] = float(loss := criterion(model(Tensor(X_train), mask_train), y_train).data)
            results.val_loss[epoch] = float(val_loss := criterion(model(Tensor(X_val), mask_val), y_val).data)

        pbar.set_postfix(loss=f"{loss:.4f}", val_loss=f"{val_loss:.4f}")

    return results


if __name__ == "__main__":
    MAX_SEQ_LEN = 10
    MIN_SEQ_LEN = 5
    HIGH = 100
    D_POS = 4

    X, y, mask = make_dataset(
        n_samples=50_000,
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
        d_v=16,
        hidden_dims=[16, 32],
        out_dim=1,
        high=HIGH,
        max_seq_len=MAX_SEQ_LEN,
    )
    print(model)

    results = fit(
        model=model,
        optimizer=Adam(parameters=model.parameters, lr=1e-3),
        criterion=softmax_cross_entropy_with_logits,
        nb_epochs=1_000,
        batch_size=512,
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
    )
