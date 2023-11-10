from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from utilities.params import Params,ODEParams,PDEParams



class GenerateParticle(ABC):

    eqtype: str

    T: int
    dt: float
    num_it: int

    u: np.array
    t: np.array

    eqtype: str

    def __init__(self, eqtype: str, params: Params):

        self.eqtype = eqtype

        self.T = params.T
        self.dt = params.dt
        self.num_it = int(self.T/self.dt)

        if self.T != self.num_it*self.dt:
            self.T = self.num_it*self.dt
            print('Final time reached: ', self.T)

        if eqtype == 'ODE':
            # assert isinstance(params, ODEParams)
            self.f = params.f
            self.u = np.zeros((len(params.u0), self.num_it+1))
        elif eqtype == 'PDE':
            assert isinstance(params, PDEParams)
            self.mass_matrix = params.mass_matrix
            self.forcing_term = params.forcing_term
            self.system_matrix = params.system_matrix
            self.u = np.zeros((self.M.size()[0], self.num_it+1))

        self.u[:, 0] = params.u0

        self.t = np.linspace(0, self.T, self.num_it+1)

    @classmethod
    def from_file(cls):
        pass

    def generate(self):
        if self.eqtype == 'ODE':
            return self.generateODE()
        elif self.eqtype == 'PDE':
            return self.generatePDE()

    @abstractmethod
    def generateODE(self):
        pass

    @abstractmethod
    def generatePDE(self):
        pass

    def reset(self):
        self.u[:, 1:] = np.zeros_like(self.u[:, 1:])

    def save(self, filename):
        np.save(file='../data/saved_particles'+filename, arr=self.u)

    def plot_solution(self, exact_sol=None):

        if exact_sol is not None:
            u_ex = exact_sol(self.t)

        n_plots = len(self.u)
        fig, axs = plt.subplots(n_plots)

        for i in range(n_plots):
            axs[i].plot(self.t, self.u[i, :], label='numerical solution')
            if exact_sol is not None:
                axs[i].plot(self.t, u_ex[i], linestyle='dashed',
                            label="exact solution")
            axs[i].set_title('component %f of the solution' % i)
            axs[i].legend()

        fig.tight_layout()
