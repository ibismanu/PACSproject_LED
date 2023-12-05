import numpy as np
from scipy.optimize import newton

from scripts.particle.generate_particle import GenerateParticle


class ThetaMethod(GenerateParticle):
    def __init__(self, theta, tol, *args, **kwargs):
        self.theta = theta
        self.tol = tol
        super().__init__(*args, **kwargs)

    def generateODE(self):
        for n in range(self.num_it):
            rhs = self.u[:, n] + self.dt * (1 - self.theta) * self.f(
                self.u[:, n], self.t[n]
            )

            def g(v):
                return v - self.theta * self.f(v, self.t[n + 1]) * self.dt - rhs

            size = np.mean(np.abs(self.u[:, n]))

            self.u[:, n + 1] = newton(g, x0=self.u[:, n], tol=(size + 1) * self.tol)

    def generatePDE(self):
        # TODO
        pass
