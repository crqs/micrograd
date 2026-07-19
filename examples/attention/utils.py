from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tqdm import tqdm

from examples.train import Results
from micrograd import Tensor
from micrograd.optim import LearningRateScheduler, Optimizer

if TYPE_CHECKING:
    from .maximum_classification import MaxClassificationModel
    from .maximum_regression import MaxRegressionModel

Task = Literal["classification", "regression"]


def make_dataset(
    n_samples: int,
    high: int,
    min_seq_len: int,
    max_seq_len: int,
    d_pos: int,
    mode: Literal["max", "argmax"],
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

    pe = _positional_encoding(max_seq_len, d_pos)  # (max_seq_len, d_pos)
    X[:, :, 1:] = pe[None, :, :]  # broadcast the same positional encoding to every sample

    mask = positions[None, :] < seq_len[:, None]  # (n_samples, max_seq_len)

    X[~mask] = 0
    match mode:
        case "max":
            y = np.max(X[:, :, 0], axis=1).reshape(-1, 1)
        case "argmax":
            y = np.argmax(X[:, :, 0], axis=1).reshape(-1, 1)

    return X, y, mask.astype(np.float32)


def _positional_encoding(max_seq_len: int, d_pos: int, base: float | None = None) -> np.ndarray:
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


def fit(
    model: MaxClassificationModel | MaxRegressionModel,
    optimizer: Optimizer,
    lr_scheduler: LearningRateScheduler | None,
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

        if lr_scheduler is not None:
            lr_scheduler.step(epoch)

        with model.eval():
            results.loss[epoch] = float(loss := criterion(model(Tensor(X_train), mask_train), y_train).data)
            results.val_loss[epoch] = float(val_loss := criterion(model(Tensor(X_val), mask_val), y_val).data)

        pbar.set_postfix(loss=f"{loss:.4f}", val_loss=f"{val_loss:.4f}")

    return results


def plot_results(
    model: MaxClassificationModel | MaxRegressionModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
    mask_test: np.ndarray,
    losses: np.ndarray,
    val_losses: np.ndarray,
    task: Task,
    n_examples: int = 8,
    save_path: str = "results_max.png",
) -> None:
    """
    Plot training results for the max-of-sequence attention examples.

    The left column always shows the loss curve on top; the panel below and the
    right-hand example panel depend on the task:

    - "classification": predicts the *position* of the max. The bottom-left panel
      is a confusion matrix over positions, and examples highlight the true and
      predicted positions.
    - "regression": predicts the *value* of the max. The bottom-left panel is a
      parity plot (predicted vs true value), and examples annotate the predicted
      value against the true max.
    """

    fig = plt.figure(figsize=(15, 10), dpi=150)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.4], height_ratios=[1, 1])
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_metric = fig.add_subplot(gs[1, 0])
    ax_examples = fig.add_subplot(gs[:, 1])

    _plot_loss(ax_loss, losses, val_losses)

    seq_len = X_test.shape[1]
    with model.eval():
        if task == "classification":
            model = cast("MaxClassificationModel", model)
            preds = np.asarray(model.predict(Tensor(X_test), mask_test)).reshape(-1).astype(int)
        else:
            preds = np.asarray(model(Tensor(X_test), mask_test).data).reshape(-1)

    trues = y_test.reshape(-1)

    if task == "classification":
        trues = trues.astype(int)
        _plot_confusion(fig, ax_metric, trues, preds, seq_len)
        _plot_classification_examples(ax_examples, X_test, mask_test, trues, preds, n_examples, seq_len)
    else:
        _plot_parity(ax_metric, trues, preds)
        _plot_regression_examples(ax_examples, X_test, mask_test, trues, preds, n_examples, seq_len)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def _plot_loss(ax: Axes, losses: np.ndarray, val_losses: np.ndarray) -> None:
    epochs = np.arange(len(losses))
    ax.plot(epochs, losses, label="train loss", linewidth=1.8)
    ax.plot(epochs, val_losses, label="val loss", linestyle="--", linewidth=1.8)
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend()


def _plot_confusion(fig: Figure, ax: Axes, trues: np.ndarray, preds: np.ndarray, seq_len: int) -> None:
    accuracy = float((preds == trues).mean())
    most_frequent_idx = int(np.bincount(trues, minlength=seq_len).argmax())
    baseline_accuracy = float((trues == most_frequent_idx).mean())

    confusion = np.zeros((seq_len, seq_len), dtype=int)
    for t, p in zip(trues, preds, strict=True):
        confusion[t, p] += 1

    per_index_accuracy = np.full(seq_len, np.nan)
    for idx in range(seq_len):
        rows = trues == idx
        if rows.sum() > 0:
            per_index_accuracy[idx] = (preds[rows] == trues[rows]).mean()

    im = ax.imshow(confusion, cmap="Blues")
    ax.set_title(
        f"Confusion matrix — acc {accuracy:.3f}  (baseline 'always idx {most_frequent_idx}': {baseline_accuracy:.3f})",
        fontsize=10,
    )
    ax.set_xlabel("Predicted idx")
    ax.set_ylabel("True idx")
    ax.set_xticks(range(seq_len))
    ax.set_yticks(range(seq_len))
    for t in range(seq_len):
        for p in range(seq_len):
            if confusion[t, p] > 0:
                ax.text(
                    p,
                    t,
                    str(confusion[t, p]),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if confusion[t, p] > confusion.max() / 2 else "black",
                )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # small per-index accuracy annotation under the confusion matrix
    # per_idx_str = "  ".join(f"{i}:{a:.2f}" for i, a in enumerate(per_index_accuracy) if not np.isnan(a))
    # ax.text(
    #     0,
    #     seq_len + 1.2,
    #     f"per true-idx accuracy:  {per_idx_str}",
    #     fontsize=7,
    #     ha="left",
    #     transform=ax.transData,
    # )


def _plot_parity(ax: Axes, trues: np.ndarray, preds: np.ndarray) -> None:
    mae = float(np.mean(np.abs(preds - trues)))
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    ss_res = float(np.sum((trues - preds) ** 2))
    ss_tot = float(np.sum((trues - trues.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    baseline_mae = float(np.mean(np.abs(trues - trues.mean())))

    lo = float(min(trues.min(), preds.min()))
    hi = float(max(trues.max(), preds.max()))

    ax.scatter(trues, preds, s=18, alpha=0.5, edgecolors="none", color="#1565c0")
    ax.plot([lo, hi], [lo, hi], color="crimson", linestyle="--", linewidth=1.5, label="ideal")
    ax.set_title(
        f"Predicted vs true max value — MAE {mae:.2f}  RMSE {rmse:.2f}  R² {r2:.3f}  (baseline MAE {baseline_mae:.2f})",
        fontsize=10,
    )
    ax.set_xlabel("True max value")
    ax.set_ylabel("Predicted max value")
    ax.grid(alpha=0.3)
    ax.legend()


def _plot_classification_examples(
    ax: Axes,
    X_test: np.ndarray,
    mask_test: np.ndarray,
    trues: np.ndarray,
    preds: np.ndarray,
    n_examples: int,
    seq_len: int,
) -> None:
    order = np.argsort(-trues[:n_examples])

    ax.set_title("Predicted vs true max position on example sequences")
    ax.set_xlim(-0.5, seq_len + 3.6)
    ax.set_ylim(0.3, n_examples + 0.7)
    ax.set_yticks([])
    ax.set_xticks(range(seq_len))
    ax.set_xlabel("Position in sequence")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

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

            ax.scatter(j, y, s=380, c="white", edgecolors=edge_color, linewidths=lw, zorder=2)
            ax.text(j, y, str(int(v)), ha="center", va="center", fontsize=8, zorder=3, color="black", fontweight="bold")

        status_color = "#2e7d32" if correct else "#c62828"
        status = "correct" if correct else "wrong"
        ax.text(seq_len + 0.3, y, f"true idx {true_pos}", ha="left", va="center", fontsize=9)
        ax.text(
            seq_len + 2.1,
            y,
            f"pred idx {pred_pos}  ({status})",
            ha="left",
            va="center",
            fontsize=9,
            color=status_color,
        )

    ax.text(0, n_examples + 1.15, "crimson = true max position", fontsize=8, color="crimson", ha="left")
    ax.text(0, n_examples + 0.9, "blue = predicted position (if wrong)", fontsize=8, color="#1565c0", ha="left")
    ax.text(0, n_examples + 0.65, "green = correct prediction", fontsize=8, color="#2e7d32", ha="left")


def _plot_regression_examples(
    ax: Axes,
    X_test: np.ndarray,
    mask_test: np.ndarray,
    trues: np.ndarray,
    preds: np.ndarray,
    n_examples: int,
    seq_len: int,
) -> None:
    order = np.argsort(-trues[:n_examples])

    ax.set_title("Predicted vs true max value on example sequences")
    ax.set_xlim(-0.5, seq_len + 4.8)
    ax.set_ylim(0.3, n_examples + 0.7)
    ax.set_yticks([])
    ax.set_xticks(range(seq_len))
    ax.set_xlabel("Position in sequence")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    for row, i in enumerate(order):
        seq = X_test[i][:, 0]
        y = n_examples - row
        true_val = trues[i]
        pred_val = preds[i]
        # position of the max among valid (unmasked) tokens
        valid = mask_test[i].astype(bool)
        true_pos = int(np.argmax(np.where(valid, seq, -np.inf)))

        for j, v in enumerate(seq):
            if not mask_test[i, j]:
                continue
            edge_color, lw = ("crimson", 2.4) if j == true_pos else ("black", 0.8)
            ax.scatter(j, y, s=380, c="white", edgecolors=edge_color, linewidths=lw, zorder=2)
            ax.text(j, y, str(int(v)), ha="center", va="center", fontsize=8, zorder=3, color="black", fontweight="bold")

        err = abs(pred_val - true_val)
        ax.text(seq_len + 0.3, y, f"true {true_val:.0f}", ha="left", va="center", fontsize=9)
        ax.text(
            seq_len + 2.1,
            y,
            f"pred {pred_val:.1f}  (err {err:.1f})",
            ha="left",
            va="center",
            fontsize=9,
            color="#1565c0",
        )

    ax.text(0, n_examples + 0.9, "crimson = true max token", fontsize=8, color="crimson", ha="left")
