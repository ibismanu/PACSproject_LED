import numpy as np
from scipy.optimize import newton, anderson

from scripts.particle.generate_particle import GenerateParticle
from scripts.utils.utils import check_butcher_sum, check_explicit_array
from functools import singledispatchmethod
from scripts.utils.params import SolverParams

class RungeKutta(GenerateParticle):
    @singledispatchmethod
    def __init__(self, params, f=None):


        self.A = params.RK_A
        self.b = params.RK_b
        self.c = params.RK_c
        self.order = len(self.c)

        check_butcher_sum(A=self.A, b=self.b, c=self.c, s=self.order)
        
        super().__init__(params=params,f=f)

    @__init__.register(str)
    def _from_file(self, params, f=None):
        P = SolverParams.get_from_file(filedir=params)
        self.__init__(P, f)

class RKExplicit(RungeKutta):
    @singledispatchmethod
    def __init__(self, params, f=None):
        super().__init__(params,f)
        is_explicit = check_explicit_array(self.A, semi=False)
        assert is_explicit, "explicit was called, but implicit Butcher array was given"
        
    @__init__.register(str)
    def _from_file(self, params, f=None):
        P = SolverParams.get_from_file(filedir=params)
        self.__init__(P, f)
        
    def generateODE(self):
        for n in range(self.num_it):
            k = np.zeros((self.order, len(self.u)))
            for i in range(self.order):
                k[i, :] = self.f(
                    self.u[:, n] + self.dt * np.dot(self.A[i, :], k),
                    self.t[n] + self.dt * self.c[i],
                )
                self.u[:, n + 1] = self.u[:, n] + self.dt * np.dot(self.b, k)

class RKSemiImplicit(RungeKutta):
    @singledispatchmethod
    def __init__(self, params,f=None):
        super().__init__(params,f)
        check_explicit_array(self.A, semi=True)

    @__init__.register(str)
    def _from_file(self, params, f=None):
        P = SolverParams.get_from_file(filedir=params)
        self.__init__(P, f)
        
    def generateODE(self):
        for n in range(self.num_it):
            k = np.zeros((self.order, len(self.u[:, 0])))
            size = np.mean(np.abs(self.u[:, n]))

            for i in range(self.order):
                if self.A[i, i] == 0:
                    k[i, :] = self.f(
                        self.u[:, n] + self.dt * np.dot(self.A[i, :], k),
                        self.t[n] + self.c[i] * self.dt,
                    )
                else:

                    def g(v):
                        res = np.zeros(len(self.u))
                        res = v - self.f(
                            self.u[:, n] + self.dt * np.dot(self.A[i, :], k),
                            self.t[n] + self.dt * self.c[i],
                        )
                        return res

                    k_tmp = anderson(g, xin=np.zeros(len(self.u)), f_rtol=1)
                    k[i, :] = newton(g, x0=k_tmp, tol=size * 1e-2)

            self.u[:, n + 1] = self.u[:, n] + self.dt * np.dot(self.b, k)

class RKImplicit(RungeKutta):
    @singledispatchmethod
    def __init__(self, params, f=None):
        super().__init__(params,f)
    
    @__init__.register(str)
    def _from_file(self, params, f=None):
        P = SolverParams.get_from_file(filedir=params)
        self. __init__(P, f)
        
    def generateODE(self):
        tol = 1e-2
        for n in range(self.num_it):
            k = np.zeros((self.order, len(self.u)))
            size = np.mean(np.abs(self.u[:, n]))

            def g(v):
                res = np.zeros_like(k)
                for i in range(self.order):
                    res[i, :] = v[i, :] - self.f(
                        self.u[:, n] + self.dt * np.dot(self.A[i, :], v),
                        self.t[n] + self.dt * self.c[i],
                    )
                return res

            k_tmp = anderson(g, xin=np.zeros_like(k), f_rtol=1)
            k = newton(g, x0=k_tmp, tol=size * tol)

            self.u[:, n + 1] = self.u[:, n] + self.dt * np.dot(self.b, k)

class RKHeun(RKExplicit):
    @singledispatchmethod
    def __init__(self,params,f=None):
        params.RK_A = np.array([[0, 0], [1, 0]], dtype=np.float32)
        params.RK_b = np.array([0.5, 0.5], dtype=np.float32)
        params.RK_c = np.array([0, 1], dtype=np.float32)

        super().__init__(params,f)

    @__init__.register(str)
    def _from_file(self, params, f=None):
        P = SolverParams.get_from_file(filedir=params)
        self. __init__(P, f)

class RKRalston(RKExplicit):
    @singledispatchmethod
    def __init__(self, params,f=None):
        params.RK_A = np.array([[0, 0], [2 / 3, 0]], dtype=np.float32)
        params.RK_b = np.array([0.25, 0.75], dtype=np.float32)
        params.RK_c = np.array([0, 2 / 3], dtype=np.float32)

        super().__init__(params,f)
    
    @__init__.register(str)
    def _from_file(self, params, f=None):
        P = SolverParams.get_from_file(filedir=params)
        self. __init__(P, f)

class RK4(RKExplicit):
    @singledispatchmethod
    def __init__(self, params,f=None):
        params.RK_A = np.array(
            [[0, 0, 0, 0], [1 / 3, 0, 0, 0], [-1 / 3, 1, 0, 0], [1, -1, 1, 0]],
            dtype=np.float32,
        )
        params.RK_b = np.array([1 / 8, 3 / 8, 3 / 8, 1 / 8], dtype=np.float32)
        params.RK_c = np.array([0, 1 / 3, 2 / 3, 1], dtype=np.float32)

        super().__init__(params,f)
    
    @__init__.register(str)
    def _from_file(self, params, f=None):
        P = SolverParams.get_from_file(filedir=params)
        self. __init__(P, f)