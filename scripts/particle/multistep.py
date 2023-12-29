import numpy as np
from math import factorial

from scripts.particle.generate_particle import GenerateParticle
from scripts.utils.utils import integral


class Multistep(GenerateParticle):
    def __init__(self, params,f):
        self.b = params.b
        super().__init__(params,f)


class AdamsBashforth(GenerateParticle):
    def __init__(self, params,f):
        self.order = params.multi_order
        super().__init__(params,f)
        self.b = np.zeros(self.order)

        def g(v, j):
            return np.prod(np.array([v + i for i in range(self.order)])) / (v + j)

        for j in range(self.order):
            self.b[self.order - j - 1] = (
                (1 - 2 * (j % 2))
                / (factorial(j) * factorial(self.order - j - 1))
                * integral(g, j, self.order)
            )

    def generateODE(self):
        for n in range(self.order):
            self.u[:, n + 1] = self.u[:, n] + self.dt * self.f(self.u[:, n], self.t[n])
        for k in range(self.num_it - self.order):
            n = self.order + k
            self.u[:, n + 1] = self.u[:, n]

            for i in range(self.order):
                self.u[:, n + 1] += (
                    self.dt
                    * self.b[i]
                    * self.f(self.u[:, k + 1 + i], self.t[k + 1 + i])
                )

class AdamsMoulton(GenerateParticle):
    # TODO
    pass


class BDF(GenerateParticle):
    # TODO
    pass
