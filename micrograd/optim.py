from abc import ABC, abstractmethod

import numpy as np

from .tensor import Tensor


class Optimizer(ABC):
    lr: float
    parameters: list[Tensor]

    def zero_grad(self) -> None:
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)

    @abstractmethod
    def step(self) -> None: ...


class SGD(Optimizer):
    def __init__(self, parameters: list[Tensor], lr: float):
        self.parameters = parameters
        self.lr = lr

    def step(self) -> None:
        for p in self.parameters:
            p.data -= self.lr * p.grad


class Adam(Optimizer):
    """
    Adam optimizer implemented from https://arxiv.org/pdf/1412.6980
    """

    def __init__(
        self,
        parameters: list[Tensor],
        lr: float,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        self.parameters = parameters
        self.lr = lr  # lr is called alpha in the original paper
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]
        self.t = 0

    def step(self) -> None:
        self.t += 1
        for i, p in enumerate(self.parameters):
            g = p.grad
            self.m[i] = self.beta_1 * self.m[i] + (1 - self.beta_1) * g
            self.v[i] = self.beta_2 * self.v[i] + (1 - self.beta_2) * g**2
            m_hat = self.m[i] / (1 - self.beta_1**self.t)
            v_hat = self.v[i] / (1 - self.beta_2**self.t)

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


class LearningRateScheduler(ABC):
    @abstractmethod
    def step(self, epoch: int) -> None: ...


class CosineDecayScheduler(LearningRateScheduler):
    def __init__(self, optimizer: Optimizer, nb_epochs: int, lr_min: float = 0.0):
        self.optimizer = optimizer
        self.lr_0 = optimizer.lr
        self.nb_epochs = nb_epochs
        self.lr_min = lr_min

    def step(self, epoch: int) -> None:
        self.optimizer.lr = self.lr_min + 0.5 * (self.lr_0 - self.lr_min) * (1 + np.cos(np.pi * epoch / self.nb_epochs))
