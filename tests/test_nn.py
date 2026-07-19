import numpy as np
import pytest

from micrograd.nn import Attention, LogitsBinaryMask, MeanPool
from micrograd.tensor import Tensor

from tests.utils import gradcheck


def test_attention():
    gradcheck(Attention(2, 3, 4), Tensor(np.array([[1.2, 3.4]])))


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


@pytest.mark.xfail
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
