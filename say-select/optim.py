# optimizer for tabular learning
from collections import defaultdict
import numpy as np


class AdamOptimizer:
    """
    Adam optimizer for tabular learning.
    The q-table being optimized may grow on the fly.
    """

    def __init__(self, *, betas=(0.9, 0.999), eps=1e-08):
        # self._lr = lr
        self._beta1 = betas[0]
        self._beta2 = betas[1]
        self._eps = eps

        # {obs -> {action -> grad}}
        self._grad = defaultdict(lambda: defaultdict(float))
        self._grad2 = defaultdict(lambda: defaultdict(float))
        self._t = defaultdict(lambda: defaultdict(float))

    def process_grad(self, obs, action, grad):
        self._grad[obs][action] *= self._beta1
        self._grad[obs][action] += (1 - self._beta1) * grad

        self._grad2[obs][action] *= self._beta2
        self._grad2[obs][action] += (1 - self._beta2) * grad * grad

        self._t[obs][action] += 1

        t = self._t[obs][action]
        m1 = self._grad[obs][action] / (1 - self._beta1**t)
        m2 = self._grad2[obs][action] / (1 - self._beta2**t)
        adam_grad: float = m1 / (np.sqrt(m2) + self._eps)
        return adam_grad
