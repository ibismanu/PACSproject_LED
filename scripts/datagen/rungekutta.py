from datagen import DataGen
from utilities.utils import valid_butcher, valid_RK

from abc import abstractmethod
import numpy as np
from scipy.optimize import newton, anderson


class RungeKutta(DataGen):

    but_A: np.array
    but_b: np.array
    but_c: np.array
    s: int

    def __init__(self, T, dt, u0, eqtype, but_A, but_b, but_c, f=None, M=None, A=None, F=None):

        self.s = len(but_c)

        valid_butcher(but_A, but_b, but_c, self.s)

        self.but_A = but_A
        self.but_b = but_b
        self.but_c = but_c

        super().__init__(T=T, dt=dt, u0=u0, eqtype=eqtype, f=f, M=M, A=A, F=F)

    @abstractmethod
    def generate(self):
        pass


class RK_explicit(RungeKutta):

    def __init__(self, T, dt, u0, eqtype, but_A, but_b, but_c, f=None, M=None, A=None, F=None):
        valid_RK(but_A, 'explicit')
        super().__init__(T=T, dt=dt, u0=u0, eqtype=eqtype, but_A=but_A,
                         but_b=but_b, but_c=but_c, f=f, M=M, A=A, F=F)

    def generate(self):
        if self.eqtype == 'ODE':
            return self.generateODE()
        elif self.eqtype == 'PDE':
            return self.generatePDE()

    def generateODE(self):
        for n in range(self.num_it-1):
            k = np.zeros((self.s, len(self.u[:, 0])))
            for i in range(self.s):
                k[i, :] = self.f(
                    self.u[:, n]+self.dt*np.dot(self.but_A[i, :], k), self.t[n]+self.but_c[i]*self.dt)
                self.u[:, n+1] = self.u[:, n] + self.dt*np.dot(self.but_b, k)

    def generatePDE(self):
        pass


class RK_semiimplicit(RungeKutta):

    def __init__(self, T, dt, u0, eqtype, but_A, but_b, but_c, f=None, M=None, A=None, F=None):
            valid_RK(but_A,'semi')
            super().__init__(T=T, dt=dt, u0=u0, eqtype=eqtype, but_A=but_A,
                         but_b=but_b, but_c=but_c, f=f, M=M, A=A, F=F)
            
    def generate(self):
        if self.eqtype == 'ODE':
            return self.generateODE()
        elif self.eqtype == 'PDE':
            return self.generatePDE()
        
    def generateODE(self):

        tol = 1e-2

        for n in range(self.num_it-1):
            k = np.zeros((self.s, len(self.u[:, 0])))
            size = np.mean(np.abs(self.u[:, n]))

            for i in range(self.s):

                if self.but_A[i,i] != 0:
                    k[i, :] = self.f(
                    self.u[:, n]+self.dt*np.dot(self.but_A[i, :], k), self.t[n]+self.but_c[i]*self.dt)

                else:
                    def g(v):
                        res = np.zeros((1,len(self.u[:,0])))
                        res = v - self.f(
                        self.u[:,n]+self.dt*np.dot(self.but_A[i,:],k),self.t[n]+self.dt*self.but_c[i])

                        return res
                    
                    k_tmp = anderson(g, xin=np.zeros(len(self.u[:,0])), f_rtol=1)
                    k[i,:] = newton(g, x0=k_tmp, tol=size*tol)

            self.u[:, n+1] = self.u[:, n] + self.dt*np.dot(self.but_b, k)

    def generatePDE(self):
        pass


class RK_implicit(RungeKutta):

    def __init__(self, T, dt, u0, eqtype, but_A, but_b, but_c, f=None, M=None, A=None, F=None):
        valid_RK(but_A, 'implicit')
        super().__init__(T=T, dt=dt, u0=u0, eqtype=eqtype, but_A=but_A,
                         but_b=but_b, but_c=but_c, f=f, M=M, A=A, F=F)

    def generate(self):
        if self.eqtype == 'ODE':
            return self.generateODE()
        elif self.eqtype == 'PDE':
            return self.generatePDE()

    def generateODE(self):
        tol = 1e-2

        for n in range(self.num_it-1):
            size = np.mean(np.abs(self.u[:, n]))
            k = np.zeros((self.s, len(self.u[:, 0])))

            def g(v):
                res = np.zeros_like(k)

                for i in range(self.s):
                    res[i, :] = v[i, :] - self.f(self.u[:, n] + self.dt*np.dot(
                        self.but_A[i, :], v), self.t[n] + self.but_c[i]*self.dt)

                return res

            k_tmp = anderson(g, xin=np.zeros_like(k), f_rtol=1)
            k = newton(g, x0=k_tmp, tol=size*tol)

            self.u[:, n+1] = self.u[:, n] + self.dt*np.dot(self.but_b, k)

    def generatePDE(self):
        pass


class RK_Heun(RK_explicit):

    def __init__(self, T, dt, u0, eqtype, f=None, M=None, A=None, F=None):
        but_A = np.array([[0, 0], [1, 0]], dtype=np.float32)
        but_b = np.array([0.5, 0.5], dtype=np.float32)
        but_c = np.array([0, 1], dtype=np.float32)

        super().__init__(T=T, dt=dt, u0=u0, eqtype=eqtype, but_A=but_A,
                         but_b=but_b, but_c=but_c, f=f, M=M, A=A, F=F)


class RK_Ralston(RK_explicit):

    def __init__(self, T, dt, u0, eqtype, f=None, M=None, A=None, F=None):
        but_A = np.array([[0, 0], [2/3, 0]], dtype=np.float32)
        but_b = np.array([0.25, 0.75], dtype=np.float32)
        but_c = np.array([0, 2/3], dtype=np.float32)

        super().__init__(T=T, dt=dt, u0=u0, eqtype=eqtype, but_A=but_A,
                         but_b=but_b, but_c=but_c, f=f, M=M, A=A, F=F)


class RK_4(RK_explicit):

    def __init__(self, T, dt, u0, eqtype, f=None, M=None, A=None, F=None):
        but_A = np.array([[0, 0, 0, 0], [1/3, 0, 0, 0],
                         [-1/3, 1, 0, 0], [1, -1, 1, 0]], dtype=np.float32)
        but_b = np.array([0.125, 0.375, 0.375, 0.125], dtype=np.float32)
        but_c = np.array([0, 1/3, 2/3, 1], dtype=np.float32)

        super().__init__(T=T, dt=dt, u0=u0, eqtype=eqtype, but_A=but_A,
                         but_b=but_b, but_c=but_c, f=f, M=M, A=A, F=F)


class RK_45(RK_explicit):  # @TODO
    pass
