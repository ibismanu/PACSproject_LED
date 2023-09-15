from GenerateParticle import GenerateParticles
from utilities.utils import integral
from math import factorial
from abc import abstractmethod

import numpy as np


class Multistep(GenerateParticles):

    mul_A: np.array
    mul_b: np.array

    def __init__(self, T, dt, u0, eqtype, mul_A, mul_b, f=None, M=None, A=None, F=None):
        self.mul_A = mul_A
        self.mul_b = mul_b
        super().__init__(T=T, dt=dt, u0=u0, eqtype=eqtype, f=f, M=M, A=A, F=F)

    @abstractmethod
    def generate(self):
        pass


class AdamsBashforth(Multistep):

    order: int

    def __init__(self, T, dt, u0, eqtype, order, f=None, M=None, A=None, F=None):

        self.order = order
        self.mul_b = np.zeros(order)

        def g(v, j):
            return np.prod(np.array([v+i for i in range(order)])) / (v+j)

        for j in range(order):
            self.mul_b[order-j-1] = (1-2*(j % 2)) / (factorial(j)
                                                     * factorial(order-j-1)) * integral(g, j, order)

        super().__init__(T=T, dt=dt, u0=u0, eqtype=eqtype, f=f, M=M, A=A, F=F)

    def generate(self):
        if self.eqtype == "ODE":
            return self.generateODE()
        elif self.eqtype == "PDE":
            return self.generatePDE()

    def generateODE(self):
        for n in range(self.num_it-1):
            if n < self.order-1:
                self.u[:, n+1] = self.u[:, n] + self.dt * \
                    self.f(self.u[:, n], self.t[n])
            else:
                self.u[:, n+1] = self.u[:, n]

                for i in range(self.order):
                    self.u[:, n+1] += self.dt * self.mul_b[i] * \
                        self.f(self.u[:, n-self.order+1+i],
                               self.t[n-self.order+1+i])

    def generatePDE(self):
        pass


class AdamsMoulton(Multistep):
    pass


class BDF(Multistep):
    pass
