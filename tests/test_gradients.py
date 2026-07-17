from collections.abc import Callable

import numpy as np

from micrograd.nn import Attention
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

    assert np.sum(np.abs(auto_gradient - numerical_gradient)) < 1e-4


def test_attention():
    gradcheck(Attention(2, 3, 4), Tensor(np.array([[1.2, 3.4]])))
