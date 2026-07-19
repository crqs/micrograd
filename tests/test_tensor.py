import numpy as np
from numpy.testing import assert_array_equal

from micrograd import Tensor

from tests.utils import gradcheck


def test_gradient_add():
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


def test_backward_add():
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
    z = x + y

    assert_array_equal(
        z.data,
        np.array(
            [
                [-3.0, -1.0],
                [3.0, 1.0],
            ]
        ),
    )

    # inject some gradient to test backward
    z.grad = np.array(
        [
            [4.0, 3.0],
            [0.0, 8.0],
        ]
    )
    z._backward()

    assert_array_equal(x.grad, z.grad)  # dL/dx = dL/dz
    assert_array_equal(y.grad, z.grad)  # dL/dx = dL/dz


def test_backward_multiply():
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
    z = x * y

    assert_array_equal(
        z.data,
        np.array(
            [
                [-4.0, -6.0],
                [2.0, 0.0],
            ]
        ),
    )

    # inject some gradient to test backward
    z.grad = np.array(
        [
            [2.0, 5.0],
            [1.0, 1.0],
        ]
    )
    z._backward()

    assert_array_equal(
        x.grad,  # dL/dx = dL/dz * y
        np.array(
            [
                [2.0, -15.0],
                [2.0, 0.0],
            ]
        ),
    )
    assert_array_equal(
        y.grad,  # dL/dy = dL/dz * X
        np.array(
            [
                [-8.0, 10.0],
                [1.0, 1.0],
            ]
        ),
    )


def test_backward_matmul():
    x = Tensor(
        np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        )
    )

    # x.T = [
    #     [1.0, 4.0],
    #     [2.0, 5.0],
    #     [3.0, 6.0],
    # ]

    y = Tensor(
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
    )
    # y.T = [
    #     [1.0, 0.0, 1.0],
    #     [0.0, 1.0, 1.0],
    # ]
    z = x @ y

    assert_array_equal(
        z.data,
        np.array(
            [
                [4.0, 5.0],
                [10.0, 11.0],
            ]
        ),
    )

    # inject some gradient to test backward
    z.grad = np.array(
        [
            [3.0, 2.0],
            [1.0, 1.0],
        ]
    )
    z._backward()

    assert_array_equal(
        x.grad,  # dL/dx = y.T @ dL/dz
        np.array(
            [
                [3.0, 2.0, 5.0],
                [1.0, 1.0, 2.0],
            ]
        ),
    )
    assert_array_equal(
        y.grad,  # dL/dy = x.T @ dL/dz
        np.array(
            [
                [7.0, 6.0],
                [11.0, 9.0],
                [15.0, 12.0],
            ]
        ),
    )


def test_backward_pow():
    x = Tensor(
        np.array(
            [
                [-4.0, 2.0],
                [1.0, 1.0],
            ]
        )
    )
    z = x**2

    assert_array_equal(
        z.data,
        np.array(
            [
                [16.0, 4.0],
                [1.0, 1.0],
            ]
        ),
    )

    # inject some gradient to test backward
    z.grad = np.array(
        [
            [2.0, 5.0],
            [1.0, 1.0],
        ]
    )
    z._backward()

    assert_array_equal(
        x.grad,  # dL/dx = n * dL/dz * x^(n-1)
        np.array(
            [
                [-16.0, 20.0],
                [2.0, 2.0],
            ]
        ),
    )


def test_backward_relu():
    x = Tensor(
        np.array(
            [
                [-4.0, 2.0],
                [1.0, 3.0],
            ]
        )
    )
    z = x.relu()

    assert_array_equal(
        z.data,
        np.array(
            [
                [0.0, 2.0],
                [1.0, 3.0],
            ]
        ),
    )

    # inject some gradient to test backward
    z.grad = np.array(
        [
            [4.0, 3.0],
            [-2.0, 8.0],
        ]
    )
    z._backward()

    assert_array_equal(
        x.grad,
        np.array(
            [
                [0.0, 3.0],
                [-2.0, 8.0],
            ]
        ),
    )
