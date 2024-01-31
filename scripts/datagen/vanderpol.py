# x' = mu*(x - 1/3 x^3 -y)
# y' = 1/mu *x

import numpy as np
import time
from multiprocessing import Pool

from scripts.datagen.datagen import DataGen
from scripts.utils.params import SolverParams
from scripts.particle.thetamethod import ThetaMethod
from scripts.particle.rungekutta import RKHeun
from scripts.particle.multistep import AdamsBashforth
from functools import singledispatchmethod


# Generate a dataset of data following the Van Del Pol system of equation
class VanDerPol(DataGen):

    params: SolverParams

    @singledispatchmethod
    def __init__(self,params,mu):
        
        # Simulation parameters
        self.params = params
        
        # Problem parameter
        self.mu = mu

        if self.params.solver_name=="thetamethod":
            self.solver = ThetaMethod(params)
        elif self.params.solver_name=="rungekutta":
            self.solver = RKHeun(params)
        elif self.params.solver_name=="multistep":
            self.solver = AdamsBashforth(params)

    @__init__.register(str)
    def _from_file(self, params, mu):
        P = SolverParams.get_from_file(filedir=params)
        self.__init__(P, mu)
        
    # Generate a single sample
    def generate_sample(self,name,x0=None,plot=False):
        
        # If the initial data is not gived, it is randomly generated in [-5,5]^2
        if x0 is None:
            self.solver.u0 = [np.random.uniform(-5, 5), np.random.uniform(-5, 5)]
            self.params.u0 = self.solver.u0

        def f(u,t):
            return [self.mu*(u[0] - 1./3*(u[0]**3)-u[1]),
                    1./self.mu*u[0],
            ]

        self.solver.set_f(f)
        self.solver.generate()
        if plot:
            self.solver.plot_solution()


        self.sample = self.solver.u.transpose()
        self.save_sample(name)

    
    # Create a dataset by generating multiple samples with different initial data
    def generate_dataset(self, num_samples, num_processes, x0=None, plot=False):
        args = [("sample_" + str(i) + ".npy", x0, plot) for i in range(num_samples)]

        start = time.perf_counter()
        
        # The generation of samples is parallelizzed
        with Pool(processes=num_processes) as pool:
            pool.starmap(self.generate_sample, args)

        finish = time.perf_counter()

        print("Program finished in " + str(finish - start) + " seconds")