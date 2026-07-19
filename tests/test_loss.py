import numpy as np

from micrograd.loss import mse
from micrograd.tensor import Tensor

from tests.utils import gradcheck


def test_mse():
    y_true = np.array([1.0, 2.3, 4.7, 8.9])
    gradcheck(
        lambda y: mse(y, y_true),
        Tensor([2.0, 2.0, 5.3, 10.2]),
    )
