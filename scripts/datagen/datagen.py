from abc import ABC, abstractmethod
import numpy as np
import sys

sys.path.append('..')
from utilities import utils


class DataGen(ABC):

    T: int
    dt: float
    num_it: int

    u: np.array
    t: np.array

    f: np.array

    M: np.array
    A: np.array
    F: np.array

    eqtype: str

    def __init__(self, T, dt, u0, eqtype, f=None, M=None, A=None, F=None):

        self.eqtype = eqtype
        self.dt = dt
        self.T = T
        self.num_it = int(T // dt)

        if T != self.num_it*dt:
            T = self.num_it*dt
            print('Final time reached: ', T)

        if eqtype == 'ODE':
            if np.isscalar(u0):
                self.u = np.zeros((1, self.num_it))
            else:
                self.u = np.zeros((len(u0), self.num_it))

            self.f = utils.to_numpy(f)

        elif eqtype == 'PDE':
            self.M = M
            self.F = F
            self.A = A
            self.u = np.zeros(M.size()[0], self.numIT)

        else:
            raise ValueError("Equation type not supported")

        self.u[:, 0] = u0

        self.t = np.linspace(0, T, self.num_it)

    @classmethod
    def fromODE(cls, T, dt, u0, f):
        return cls(T=T, dt=dt, u0=u0, f=f, eqtype='ODE')

    @classmethod
    def fromPDE(cls, T, dt, u0, FEMobj):
        M, A, F = FEMobj.generate()  # @TODO
        return cls(T=T, dt=dt, u0=u0, M=M, A=A, F=F, eqtype="PDE")

    @classmethod
    def fromFile(cls, filename):
        pass

    @abstractmethod
    def generate(self):
        pass

    def save(self, format):
        pass
