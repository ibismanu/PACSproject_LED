from abc import ABC, abstractmethod
import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append('..')
# import utils
from utilities import utils


class GenerateParticle(ABC):

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
    '''
    ODE: u'=f(u)
    
    PDE: MU'+AU=F 
    '''
    
    def __init__(self, T, dt, u0, eqtype, f=None, M=None, A=None, F=None):

        self.eqtype = eqtype
        self.dt = dt
        self.T = T
        self.num_it = int(T / dt)
        
        if T != self.num_it*dt:
            T = self.num_it*dt
            print('Final time reached: ', T)
        
        self.num_it = self.num_it+1
        
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
        
        # T = 10
        # dt = 5
        # eqtype = 'ODE'
        
        # read_line
        # find "="
        # symbol = left_of_=
        # value = right_of_=
        
        # if symbol == "T":
        #     T = value
        
        
        pass

    @abstractmethod
    def generate(self):
        pass

    def reset(self):
        u0 = self.u[:,0]
        
        if self.eqtype == 'ODE':
            if np.isscalar(u0):
                self.u = np.zeros((1, self.num_it))
            else:
                self.u = np.zeros((len(u0), self.num_it))

        elif self.eqtype == 'PDE':
            self.u = np.zeros(self.M.size()[0], self.numIT)
        
        self.u[:, 0] = u0
        
        
        
    def save(self, format="NPY"):
        #set directory
        np.save(self.u)
        pass
    
    def plot_solution(self, u_ex=None):
        
        if u_ex != None:
            exact = u_ex(self.t)
        
        n_plots = len(self.u[:,0])
        fig, axs = plt.subplots(n_plots)
            
        for i in range(n_plots):
            axs[i].plot(self.t,self.u[i,:],label="numerical solution")
            if u_ex != None:
                axs[i].plot(self.t,exact[i],linestyle='dashed', label="exact solution")
            axs[i].set_title('component %f of the solution' %i)
            axs[i].legend()
            
        fig.tight_layout()

