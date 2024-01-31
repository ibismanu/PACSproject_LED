import numpy as np
import time
from multiprocessing import Pool

from scripts.datagen.datagen import DataGen
from scripts.utils.params import SolverParams
from scripts.particle.thetamethod import ThetaMethod
from scripts.particle.rungekutta import RKHeun
from scripts.particle.multistep import AdamsBashforth
from functools import singledispatchmethod

def f_temp(x):
    return np.array([0])

# Generate a dataset of data following the Fitzhug-Nagumo system of equation
# The firing signal is assumed to be propagating radially
class FitzhugNagumo(DataGen):
    params: SolverParams

    @singledispatchmethod
    def __init__(
        self, params, k, alpha, epsilon, I, gamma, grid_size
    ):
        # Simulation parameters
        self.params = params
        
        # Problem parameters
        self.k = k
        self.alpha = alpha
        self.epsilon = epsilon
        self.I = I
        self.gamma = gamma
        
        self.grid_size = grid_size
        self.num_it = int(params.final_time / params.time_step)

        # Solution inizialization 
        self.sample = np.zeros(
            (self.num_it + 1, grid_size[0], grid_size[1], len(params.u0))
        )

        if self.params.solver_name=="thetamethod":
            self.solver = ThetaMethod(params)
        elif self.params.solver_name=="rungekutta":
            self.solver = RKHeun(params)
        elif self.params.solver_name=="multistep":
            self.solver = AdamsBashforth(params)

    @__init__.register(str)
    def _from_file(self, params, k, alpha, epsilon, I, gamma, grid_size):
        P = SolverParams.get_from_file(filedir=params)
        self.__init__(P, k, alpha, epsilon, I, gamma, grid_size)
        
    # Generate a single sample
    def generate_sample(self, name, x0=None, plot=False):
        
        # If the initial firing point is not gived, it is randomly generated in [0,1]^2
        if x0 is None:
            x0 = (np.random.uniform(0, 1), np.random.uniform(0, 1))

        t_begin = 0.0
        v = 5e-3
        length_t = 2.0

        hx = 1.0 / self.grid_size[0]
        hy = 1.0 / self.grid_size[1]

        for i in range(self.grid_size[0]):
            x = i * hx
            for j in range(self.grid_size[1]):
                y = j * hy
                
                # Compute distance of current cell from firing point and time instant on which it will be reached by the stumili
                distance = np.sqrt((x - x0[0]) ** 2 + (y - x0[1]) ** 2)
                t_begin = distance / v
                t_end = t_begin + length_t

                def f(u, t):
                    I_app = self.I * (t >= t_begin and t < t_end)
                    return [
                        self.k * u[0] * (u[0] - self.alpha) * (1 - u[0]) - u[1] + I_app,
                        self.epsilon * (u[0] - self.gamma * u[1]),
                    ]

                self.solver.set_f(f)
                self.solver.reset()
                self.solver.generate()
                if plot:
                    self.solver.plot_solution()

                self.sample[:,i,j] = self.solver.u.transpose()


        self.save_sample(name)

    # Create a dataset by generating multiple samples with different firing locations
    def generate_dataset(self, num_samples, num_processes, x0=None, plot=False):
        args = [("sample_" + str(i) + ".npy", x0, plot) for i in range(num_samples)]

        start = time.perf_counter()
        
        # The generation of samples is parallelized
        with Pool(processes=num_processes) as pool:
            pool.starmap(self.generate_sample, args)

        finish = time.perf_counter()

        print("Program finished in " + str(finish - start) + " seconds")
