import numpy as np
from scipy.optimize import newton, anderson

from scripts.particle.generate_particle import GenerateParticle
from scripts.utils.utils import check_butcher_sum, check_explicit_array
from functools import singledispatchmethod
from scripts.utils.params import SolverParams

# Use the Runge-Kutta method to solve the equation

# "RungeKutta" objects cannot be instantiated the abstract method "generateODE" is not implemented
class RungeKutta(GenerateParticle):
    
    @singledispatchmethod
    def __init__(self, params, f=None):


        self.A = params.RK_A
        self.b = params.RK_b
        self.c = params.RK_c
        self.order = len(self.c)
        
        # Check the theoretical conditions for the stability of the Runge Kutta method
        check_butcher_sum(A=self.A, b=self.b, c=self.c, s=self.order)
        
        super().__init__(params=params,f=f)

    # Constructor overloading
    @__init__.register(str)
    def _from_file(self, params, f=None):
        P = SolverParams.get_from_file(filedir=params)
        self.__init__(P, f)


# Specializion for explicit Runge-Kutta methods
class RKExplicit(RungeKutta):
    
    @singledispatchmethod
    def __init__(self, params, f=None):
        super().__init__(params,f)
        
        # Check that the given array is explicit
        is_explicit = check_explicit_array(self.A, semi=False)
        assert is_explicit, "explicit was called, but implicit Butcher array was given"
    
    # Constructor overloading
    @__init__.register(str)
    def _from_file(self, params, f=None):
        P = SolverParams.get_from_file(filedir=params)
        self.__init__(P, f)
        
    def generateODE(self):
        
        # Loop over times
        for n in range(self.num_it):
            
            k = np.zeros((self.order, len(self.u)))
            for i in range(self.order):
                k[i, :] = self.f(
                    self.u[:, n] + self.dt * np.dot(self.A[i, :], k),
                    self.t[n] + self.dt * self.c[i],
                )
                self.u[:, n + 1] = self.u[:, n] + self.dt * np.dot(self.b, k)


# Specializion for semi-implicit Runge-Kutta methods
class RKSemiImplicit(RungeKutta):
    
    @singledispatchmethod
    def __init__(self, params,f=None):
        super().__init__(params,f)
        
        # If the given array is explicit, print a warning
        check_explicit_array(self.A, semi=True)

    # Constructor overloading
    @__init__.register(str)
    def _from_file(self, params, f=None):
        P = SolverParams.get_from_file(filedir=params)
        self.__init__(P, f)
        
    def generateODE(self):
        
        # Loop over times
        for n in range(self.num_it):
            
            k = np.zeros((self.order, len(self.u[:, 0])))
            size = np.mean(np.abs(self.u[:, n]))

            for i in range(self.order):
                
                # Use direct computation if A(i,i) is null, otherwise use the Newton method
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

                    k_tmp = anderson(g, xin=np.zeros(len(self.u)), f_rtol=1)    #The Anderson method is used to compute an initial guess for Newton
                    k[i, :] = newton(g, x0=k_tmp, tol=size * 1e-2)

            self.u[:, n + 1] = self.u[:, n] + self.dt * np.dot(self.b, k)


# Specializion for implicit Runge-Kutta methods
class RKImplicit(RungeKutta):
    
    @singledispatchmethod
    def __init__(self, params, f=None):
        super().__init__(params,f)
    
    # Constructor overloading    
    @__init__.register(str)
    def _from_file(self, params, f=None):
        P = SolverParams.get_from_file(filedir=params)
        self. __init__(P, f)
        
    def generateODE(self):
        tol = 1e-2
        
        # Loop over times
        for n in range(self.num_it):
            
            # Solve each iteration via the Newton method
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

            k_tmp = anderson(g, xin=np.zeros_like(k), f_rtol=1)     #The Anderson method is used to compute an initial guess for Newton
            k = newton(g, x0=k_tmp, tol=size * tol)

            self.u[:, n + 1] = self.u[:, n] + self.dt * np.dot(self.b, k)


# Specializion for the Heun method
class RKHeun(RKExplicit):
    
    @singledispatchmethod
    def __init__(self,params,f=None):
        params.RK_A = np.array([[0, 0], [1, 0]], dtype=np.float32)
        params.RK_b = np.array([0.5, 0.5], dtype=np.float32)
        params.RK_c = np.array([0, 1], dtype=np.float32)

        super().__init__(params,f)
    
    # Constructor overloading
    @__init__.register(str)
    def _from_file(self, params, f=None):
        P = SolverParams.get_from_file(filedir=params)
        self. __init__(P, f)


# Specializion for the Ralston method
class RKRalston(RKExplicit):

    @singledispatchmethod
    def __init__(self, params,f=None):
        params.RK_A = np.array([[0, 0], [2 / 3, 0]], dtype=np.float32)
        params.RK_b = np.array([0.25, 0.75], dtype=np.float32)
        params.RK_c = np.array([0, 2 / 3], dtype=np.float32)

        super().__init__(params,f)
    
    # Constructor overloading
    @__init__.register(str)
    def _from_file(self, params, f=None):
        P = SolverParams.get_from_file(filedir=params)
        self. __init__(P, f)

# Specializion for the Runge-Kutta 4 method
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
        
    # Constructor overloading
    @__init__.register(str)
    def _from_file(self, params, f=None):
        P = SolverParams.get_from_file(filedir=params)
        self. __init__(P, f)