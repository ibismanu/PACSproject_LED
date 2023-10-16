import numpy as np
from scipy.optimize import newton

import sys
sys.path.append('..')
from particle.generate_particle_parallel import GenerateParticle, Params


class ThetaMethod(GenerateParticle):

    theta: float
    tol: float

    def __init__(self, eqtype: str, params: Params):
        assert params.theta is not None and params.tol is not None

        self.theta = params.theta
        self.tol = params.tol

        super().__init__(eqtype, params)

    def generateODE(self):

        for n in range(self.num_it):
            rhs = self.u[:, n] + self.dt * \
                (1-self.theta)*self.f(self.u[:, n], self.t[n])

            def g(v): return v - self.theta * \
                self.f(v, self.t[n+1])*self.dt - rhs

            size = np.mean(np.abs(self.u[:, n]))

            self.u[:, n+1] = newton(g, x0=self.u[:, n], tol=(size+1)*self.tol)

    def generatePDE(self):
        pass
