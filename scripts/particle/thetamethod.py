import numpy as np
from scipy.optimize import newton

from scripts.particle.generate_particle import GenerateParticle
from functools import singledispatchmethod
from scripts.utils.params import SolverParams


# Use the Theta method to solve the equation
# The formulation reads theta*f(u^(k+1))+(1-theta)*f(u^k)=0
class ThetaMethod(GenerateParticle):
    
    @singledispatchmethod
    def __init__(self, params,f=None):
        self.theta = params.theta
        self.tol = params.tol
        super().__init__(params,f)

    # Constructor overloading
    @__init__.register(str)
    def _from_file(self, params, f=None):
        P = SolverParams.get_from_file(filedir=params)
        self.__init__(P, f)
        
    def generateODE(self):

        self.u[:, 0] = self.u0

        # Loop over times
        for n in range(self.num_it):
            
            # Compute right hand side
            rhs = self.u[:, n] + self.dt * (1 - self.theta) * self.f(
                self.u[:, n], self.t[n]
            )

            # Solve each step with the Newton method
            def g(v):
                return v - self.theta * self.f(v, self.t[n + 1]) * self.dt - rhs

            size = np.mean(np.abs(self.u[:, n]))

            self.u[:, n + 1] = newton(g, x0=self.u[:, n], tol=(size + 1) * self.tol)

