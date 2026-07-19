from collections.abc import Callable

import numpy as np

from micrograd.tensor import Tensor


def _reaches(out: Tensor, x: Tensor) -> bool:
    """True if x is reachable from out through the autograd graph (any depth)."""
    seen: set[int] = set()
    stack: list[Tensor] = [out]
    while stack:
        node = stack.pop()
        if node is x:
            return True
        if id(node) in seen:
            continue
        seen.add(id(node))
        stack.extend(node.children)
    return False


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
    assert _reaches(out, x), f"{f} does not reference its input in the graph (children missing)"

    loss = out.sum()  # get a scalar before computing backprop
    loss.backward()
    auto_gradient = x.grad

    assert np.max(np.abs(auto_gradient - numerical_gradient)) < 1e-4
