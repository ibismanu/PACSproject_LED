import numpy as np
from math import factorial

import sys
sys.path.append('..')
from particle.generate_particle import GenerateParticle
from utilities.params import Params
from utilities.utils import integral


class Multistep(GenerateParticle):

    # multistep arrays
    A: np.array
    b: np.array

    def __init__(self, eqtype: str, params: Params):
        super().__init__(eqtype, params)

        self.A = params.multi_A
        self.b = params.multi_b


class AdamsBashforth(Multistep):

    order: int

    def __init__(self, eqtype: str, params: Params):
        assert params.multi_order is not None

        super().__init__(eqtype, params)

        self.order = params.multi_order
        self.b = np.zeros(self.order)

        def g(v, j):
            return np.prod(np.array([v+i for i in range(self.order)])) / (v+j)

        for j in range(self.order):
            self.b[self.order-j-1] = (1-2*(j % 2)) / (factorial(j) *
                                                      factorial(self.order-j-1)) * integral(g, j, self.order)

    def generateODE(self):
        print(self.b)
        for n in range(self.order):
            self.u[:, n+1] = self.u[:, n] + self.dt * \
                self.f(self.u[:, n], self.t[n])
        for k in range(self.num_it-self.order):
            n = self.order+k
            self.u[:, n+1] = self.u[:, n]

            for i in range(self.order):
                self.u[:, n+1] += self.dt * self.b[i] * \
                    self.f(self.u[:, k+1+i], self.t[k+1+i])

    def generatePDE(self):
        pass


class AdamsMoulton(Multistep):
    pass


class BDF(Multistep):
    pass
