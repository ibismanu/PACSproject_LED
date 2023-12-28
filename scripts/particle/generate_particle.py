import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from scripts.utils.utils import to_numpy


class GenerateParticle(ABC):
    def __init__(
        self,
        final_time,
        time_step,
        u0,
        f=None
    ):

        # Time parameters
        self.dt = time_step
        self.T = final_time
        self.num_it = int(self.T / self.dt)
        self.f = f
        self.u = np.zeros((len(u0), self.num_it + 1))

        if self.T != self.num_it * self.dt:
            self.T = self.num_it * self.dt
            print("Final time reached: ", self.T)

        self.t = np.linspace(0, self.T, self.num_it + 1)

        if np.isscalar(u0):
            u0 = np.array([u0])

        self.u[:, 0] = u0

    @abstractmethod
    def generateODE(self):
        pass

    def reset(self):
        self.u[:, 1:] = np.zeros_like(self.u[:, 1:])

    def save(self, name):
        np.save(file="../data/saved_particles" + name, arr=self.u)

    def set_f(self, f):
        self.f = to_numpy(f)

    def plot_solution(self, exact_sol=None):
        n_plots = len(self.u)
        fig, axs = plt.subplots(n_plots)

        for i in range(n_plots):
            axs[i].plot(self.t, self.u[i, :], label="numerical solution")
            if exact_sol is not None:
                u_ex = exact_sol(self.t)
                axs[i].plot(self.t, u_ex[i], linestyle="dashed", label="exact solution")
            axs[i].set_title("component %f of the solution" % i)
            axs[i].legend()

        fig.tight_layout()
