import numpy as np
from scipy.optimize import newton, anderson
import warnings

from particle.generate_particle import GenerateParticle
from utils.utils import check_butcher_sum, check_explicit_array


class RungeKutta(GenerateParticle):
    def __init__(self, A, b, c, *args, **kwargs):
        self.A = A
        self.b = b
        self.c = c
        self.order = len(c)

        check_butcher_sum(A=self.A, b=self.b, c=self.c, s=self.order)

        super().__init__(*args, **kwargs)


class RKExplicit(RungeKutta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not check_explicit_array(self.A, semi=False):
            warnings.warn('Explicit was called, but Implicit Butcher array was given')

    def generateODE(self):
        for n in range(self.num_it):
            k = np.zeros((self.order, len(self.u)))
            for i in range(self.order):
                k[i, :] = self.f(
                    self.u[:, n] + self.dt * np.dot(self.A[i, :], k),
                    self.t[n] + self.dt * self.c[i],
                )
                self.u[:, n + 1] = self.u[:, n] + self.dt * np.dot(self.b, k)

    def generatePDE(self):
        # TODO
        pass


class RKSemiImplicit(RungeKutta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        check_explicit_array(self.A, semi=True)

    def generateODE(self):
        for n in range(self.num_it):
            k = np.zeros((self.order, len(self.u[:, 0])))
            size = np.mean(np.abs(self.u[:, n]))

            for i in range(self.order):
                if self.A[i, i] == 0:
                    k[i, :] = self.f(
                        self.u[:, n] + self.dt * np.dot(self.A[i, :], k),
                        self.t[n] + self.c[i] * self.dt,
                    )
                else:

                    def g(v):
                        res = np.zeros(len(self.u))
                        res = v - self.f(
                            self.u[:, n] + self.dt * np.dot(self.A[i, :], k),
                            self.t[n] + self.dt * self.c[i],
                        )
                        return res

                    k_tmp = anderson(g, xin=np.zeros(len(self.u)), f_rtol=1)
                    k[i, :] = newton(g, x0=k_tmp, tol=size * 1e-2)

            self.u[:, n + 1] = self.u[:, n] + self.dt * np.dot(self.b, k)

    def generatePDE(self):
        # TODO
        pass


class RKImplicit(RungeKutta):
    def generateODE(self):
        tol = 1e-2
        for n in range(self.num_it):
            k = np.zeros((self.order, len(self.u)))
            size = np.mean(np.abs(self.u[:, n]))

            def g(v):
                res = np.zeros_like(k)
                for i in range(self.order):
                    res[i, :] = v[i, :] - self.f(
                        self.u[:, n] + self.dt * np.dot(self.A[i, :], v),
                        self.t[n] + self.dt * self.c[i],
                    )
                return res

            k_tmp = anderson(g, xin=np.zeros_like(k), f_rtol=1)
            k = newton(g, x0=k_tmp, tol=size * tol)

            self.u[:, n + 1] = self.u[:, n] + self.dt * np.dot(self.b, k)

    def generatePDE(self):
        # TODO
        pass


class RKHeun(RKExplicit):
    def __init__(self, *args, **kwargs):
        A = np.array([[0, 0], [1, 0]], dtype=np.float32)
        b = np.array([0.5, 0.5], dtype=np.float32)
        c = np.array([0, 1], dtype=np.float32)

        super().__init__(A=A, b=b, c=c, *args, **kwargs)


class RKRalston(RKExplicit):
    def __init__(self, *args, **kwargs):
        A = np.array([[0, 0], [2 / 3, 0]], dtype=np.float32)
        b = np.array([0.25, 0.75], dtype=np.float32)
        c = np.array([0, 2 / 3], dtype=np.float32)

        super().__init__(A=A, b=b, c=c, *args, **kwargs)


class RK4(RKExplicit):
    def __init__(self, *args, **kwargs):
        A = np.array(
            [[0, 0, 0, 0], [1 / 3, 0, 0, 0], [-1 / 3, 1, 0, 0], [1, -1, 1, 0]],
            dtype=np.float32,
        )
        b = np.array([1 / 8, 3 / 8, 3 / 8, 1 / 8], dtype=np.float32)
        c = np.array([0, 1 / 3, 2 / 3, 1], dtype=np.float32)

        super().__init__(A=A, b=b, c=c, *args, **kwargs)
