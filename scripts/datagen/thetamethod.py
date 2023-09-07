from datagen import DataGen

import numpy as np
from scipy.optimize import newton

class ThetaMethod(DataGen):

    theta: float
    tol: float

    def __init__(self, T, dt, u0, eqtype, theta=0.5, tol=1e-2, f=None, M=None, A=None, F=None):
        super().__init__(T=T, dt=dt, u0=u0, f=f, M=M, A=A, F=F, eqtype=eqtype)
        self.theta = theta
        self.tol = tol

    def generate(self):
        if self.eqtype == "ODE":
            return self.generateODE()
        elif self.eqtype == "PDE":
            return self.generatePDE()

    def generateODE(self):

        for n in range(self.num_it-1):
            rhs = self.u[:, n] + self.dt * \
                (1-self.theta)*self.f(self.u[:, n], self.t[n])

            def g(v): return v - self.theta*self.f(v, self.t[n+1])*self.dt - rhs
            
            size = np.mean(np.abs(self.u[:, n]))

            self.u[:,n+1] = newton(g,x0=self.u[:,n],tol=size*self.tol)

    def generatePDE(self):
        pass
