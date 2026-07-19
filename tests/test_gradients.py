from collections.abc import Callable

import numpy as np

from micrograd.loss import mse
from micrograd.nn import Attention, LogitsBinaryMask, MeanPool
from micrograd.tensor import Tensor


def gradcheck(f: Callable[[Tensor], Tensor], x: Tensor, epsilon: float = 1e-5):

    # compute a numerical gradient using the finite differences method
    data = x.data.copy()
    numerical_gradient = np.zeros_like(data)
    for idx in np.ndindex(data.shape):
        orig = data[idx]

        data[idx] = orig + epsilon
        loss_plus = f(Tensor(data.copy())).data.sum()

        data[idx] = orig - epsilon
        loss_minus = f(Tensor(data.copy())).data.sum()

        numerical_gradient[idx] = (loss_plus - loss_minus) / (2 * epsilon)
        data[idx] = orig

    # compute gradient with autograd
    out = f(x)
    loss = out.sum()  # get a scalar before computing backprop
    loss.backward()
    auto_gradient = x.grad

    assert np.max(np.abs(auto_gradient - numerical_gradient)) < 1e-4


def test_attention():
    gradcheck(Attention(2, 3, 4), Tensor(np.array([[1.2, 3.4]])))


def test_mse():
    y_true = np.array([1.0, 2.3, 4.7, 8.9])
    gradcheck(
        lambda y: mse(y, y_true),
        Tensor([2.0, 2.0, 5.3, 10.2]),
    )


def test_add():
    x = Tensor(
        np.array(
            [
                [-4.0, 2.0],
                [1.0, 1.0],
            ]
        )
    )
    y = Tensor(
        np.array(
            [
                [1.0, -3.0],
                [2.0, 0.0],
            ]
        )
    )
    gradcheck(lambda x: x + y, x)


def test_mean_pool():
    x = Tensor(
        np.array(
            [
                [
                    [1.0, 2.3],
                    [2.4, 5.6],
                    [0.1, 2.0],
                    [0.1, -1.0],
                ],
                [
                    [2.0, 3.3],
                    [3.4, 6.6],
                    [3.4, 6.6],
                    [0.4, 0.0],
                ],
                [
                    [1.0, 3.3],
                    [2.4, 8.4],
                    [3.4, 5.6],
                    [0.9, 0.0],
                ],
            ]
        )
    )
    mask = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
        ]
    )
    gradcheck(lambda x: MeanPool()(x, mask), x)


def test_logist_binary_mask():
    x = Tensor(
        np.array(
            [
                [
                    [1.0, 2.3],
                    [2.4, 5.6],
                    [0.1, 2.0],
                    [0.1, -1.0],
                ],
                [
                    [2.0, 3.3],
                    [3.4, 6.6],
                    [3.4, 6.6],
                    [0.4, 0.0],
                ],
                [
                    [1.0, 3.3],
                    [2.4, 8.4],
                    [3.4, 5.6],
                    [0.9, 0.0],
                ],
            ]
        )
    )
    mask = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
        ]
    )
    gradcheck(lambda x: LogitsBinaryMask()(x, mask), x)
