import numpy as np
import torch
from numpy.testing import assert_allclose, assert_array_equal

from micrograd import Tensor
from micrograd.operations import sigmoid, softmax


def test_backward_sigmoid():
    x = Tensor(
        np.array(
            [
                [-4.0, 2.0],
                [1.0, 1.0],
            ]
        )
    )
    z = sigmoid(x)

    expected_data = 1 / (1 + np.exp(-x.data))
    assert_array_equal(z.data, expected_data)

    # inject some gradient to test backward
    z.grad = np.array(
        [
            [4.0, 3.0],
            [0.0, 8.0],
        ]
    )
    z._backward()

    assert_array_equal(
        x.grad,  # dL/dx = dL/dz * sigmoid(x) * (1 - sigmoid(x))
        z.grad * expected_data * (1 - expected_data),
    )


def test_backward_softmax():
    x = Tensor(
        np.array(
            [
                [1.0, 2.0, 0.5],
                [0.1, -1.0, 3.0],
            ]
        )
    )
    z = softmax(x, axis=-1)

    e = np.exp(x.data - np.max(x.data))
    expected_data = e / np.sum(e, axis=-1, keepdims=True)
    assert_array_equal(z.data, expected_data)

    # inject some gradient to test backward
    z.grad = np.array(
        [
            [4.0, 3.0, 1.0],
            [0.0, 8.0, -2.0],
        ]
    )
    z._backward()

    dot = np.sum(z.grad * expected_data, axis=-1, keepdims=True)
    assert_array_equal(
        x.grad,  # dL/dx = softmax(x) * (dL/dz - sum(dL/dz * softmax(x)))
        expected_data * (z.grad - dot),
    )


def test_sigmoid_matches_pytorch():
    """
    Test that micrograd's sigmoid produces the same forward/backward results as PyTorch's.
    """

    x_data = [
        [1.0, -2.0, 0.5],
        [3.0, -0.5, 2.0],
    ]

    x = Tensor(np.array(x_data))
    z = sigmoid(x)
    z.backward()  # equivalent to loss = sum(z)
    xmg, zmg = x, z

    x_pt = torch.tensor(x_data, dtype=torch.float64, requires_grad=True)
    z_pt = torch.sigmoid(x_pt)
    z_pt.sum().backward()

    tol = 1e-6
    # forward pass went well
    assert_allclose(zmg.data, z_pt.detach().numpy(), atol=tol)
    # backward pass went well
    assert_allclose(xmg.grad, x_pt.grad.numpy(), atol=tol)


def test_softmax_matches_pytorch():
    """
    Test that micrograd's softmax produces the same forward/backward results as PyTorch's.
    """

    x_data = [
        [1.0, 2.0, 0.5],
        [0.1, -1.0, 3.0],
    ]

    x = Tensor(np.array(x_data))
    z = softmax(x, axis=-1)
    z.backward()  # equivalent to loss = sum(z)
    xmg, zmg = x, z

    x_pt = torch.tensor(x_data, dtype=torch.float64, requires_grad=True)
    z_pt = torch.softmax(x_pt, dim=-1)
    z_pt.sum().backward()

    tol = 1e-6
    # forward pass went well
    assert_allclose(zmg.data, z_pt.detach().numpy(), atol=tol)
    # backward pass went well
    assert_allclose(xmg.grad, x_pt.grad.numpy(), atol=tol)
