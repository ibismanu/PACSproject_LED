import numpy as np
from math import factorial

from scripts.particle.generate_particle import GenerateParticle
from scripts.utils.utils import integral
from functools import singledispatchmethod
from scripts.utils.params import SolverParams


# Use a Multistep method to solve the equation

# "Multistep" objects cannot be instantiated the abstract method "generate" is not implemented
class Multistep(GenerateParticle):
    def __init__(self, params,f=None):
        self.b = params.b
        super().__init__(params,f)


class AdamsBashforth(Multistep):
    
    @singledispatchmethod
    def __init__(self, params,f=None):
        self.order = params.multi_order
        super().__init__(params,f)
        self.b = np.zeros(self.order)

        def g(v, j):
            return np.prod(np.array([v + i for i in range(self.order)])) / (v + j)

        # Initialize the array according to the theoretical background
        for j in range(self.order):
            self.b[self.order - j - 1] = (
                (1 - 2 * (j % 2))
                / (factorial(j) * factorial(self.order - j - 1))
                * integral(g, j, self.order)
            )
    
    # Constructor overloading
    @__init__.register(str)
    def _from_file(self, params, f=None):
        P = SolverParams.get_from_file(filedir=params)
        self. __init__(P, f)
        
    def generate(self):
        
        # Compute the first steps via Backward Euler
        for n in range(self.order):
            self.u[:, n + 1] = self.u[:, n] + self.dt * self.f(self.u[:, n], self.t[n])
        
        # Loop over the remaining time steps
        for k in range(self.num_it - self.order):
            n = self.order + k
            self.u[:, n + 1] = self.u[:, n]

            for i in range(self.order):
                self.u[:, n + 1] += (
                    self.dt
                    * self.b[i]
                    * self.f(self.u[:, k + 1 + i], self.t[k + 1 + i])
                )