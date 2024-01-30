import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from scripts.utils.utils import to_numpy
from functools import singledispatchmethod
from scripts.utils.params import SolverParams

# Base Abstract Class
# Used to solve an ODE in the form f(u)=0

class GenerateParticle(ABC):
    
    def __init__(
        self,
        params,
        f=None
    ):

        # Simulation parameters
        self.dt = params.time_step                          # Time step
        self.T = params.final_time                          # Final time
        self.u0 = params.u0                                 # Initial data
        self.num_it = int(self.T / self.dt)                 # Number of iterations in time
        if f is not None:                                   # Forcing term
            self.f = to_numpy(f)                                
        else:
            self.f = None
        self.u = np.zeros((len(self.u0), self.num_it + 1))  # Inizializing solution

        # adjust final time so that it is a multiple of the time step
        if self.T != self.num_it * self.dt:
            self.T = self.num_it * self.dt
            print("Final time reached: ", self.T)

        self.t = np.linspace(0, self.T, self.num_it + 1)    # Initializing times

        # Assigning initial solution
        if np.isscalar(self.u0):
            self.u0 = np.array([self.u0])

        self.u[:, 0] = self.u0
        
        
    @abstractmethod
    def generateODE(self):
        # Abstract method, solve the equation
        pass
    
    def reset(self):
        # Set the solution array to zero (apart from the initiall value)
        
        self.u[:, 1:] = np.zeros_like(self.u[:, 1:])

    def save(self, name):
        # Save the solution
        
        np.save(file="../data/saved_particles" + name, arr=self.u)

    def set_f(self, f):
        # Assign the problem
        
        self.f = to_numpy(f)

    def plot_solution(self, exact_sol=None):
        # Plot the numerical solution 
        
        n_plots = len(self.u)
        fig, axs = plt.subplots(n_plots)

        for i in range(n_plots):
            axs[i].plot(self.t, self.u[i, :], label="numerical solution")
            
            # Optional: also plot the exact solution
            if exact_sol is not None:
                u_ex = exact_sol(self.t)
                axs[i].plot(self.t, u_ex[i], linestyle="dashed", label="exact solution")
            axs[i].set_title("component %i of the solution" % i)
            axs[i].legend()

        fig.tight_layout()
