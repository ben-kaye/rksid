import math
import numpy as np


class Kernel():
    def k(self, t, s):
        raise NotImplementedError


class StableSpline1(Kernel):

    def __init__(self, beta = 1.) -> None:
        self.beta = beta

    def k(self, t, s):
        return math.exp(-self.beta*max(t, s))


class ImpulseIdentifier():
    def __init__(self, kernel=StableSpline1()) -> None:
        self.kernel = kernel

    def fit(self, u, y, reg=1.):
        N = max(y.shape)
        m = 50 # system memory

        def a_ts(t, s):
            conv = 0.
            for h in range(m):
                if s - h > 0:
                    conv += self.kernel.k(t, h)*u[s-h]
            return conv

        def A_ts(t, s):
            conv = 0.
            for h in range(m):
                if t - h > 0:
                    conv += a_ts(h, s)*u[t-h]
            return conv

        self.A_m = np.ndarray([[A_ts(t, s)
                             for s in range(N)] for t in range(N)])

        self.c = np.linalg.solve((self.A_m + reg*np.eye(N)), y)
        self.impulse = np.dot(self.c.T, self.A_m)

        return self.impulse

    def __call__(self, u, y, **kwds):
        return self.fit(u, y, **kwds)