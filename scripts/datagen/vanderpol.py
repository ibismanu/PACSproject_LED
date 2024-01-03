# x' = mu*(x - 1/3 x^3 -y)
# y' = 1/mu *x

# NOTE: the paper AdaLed says that they do not use the Autoencoder

import numpy as np
import time
from multiprocessing import Pool

from scripts.datagen.datagen import DataGen
from scripts.utils.params import SolverParams
from scripts.particle.thetamethod import ThetaMethod
from scripts.particle.rungekutta import RKHeun
from scripts.particle.multistep import AdamsBashforth

class VanDerPol(DataGen):

    params: SolverParams

    def __init__(self,params,mu):
        self.params = params
        self.mu = mu

        match self.params.solver_name:
            case "thetamethod":
                self.solver = ThetaMethod(params)
            case "rungekutta":
                self.solver = RKHeun(params)
            case "multistep":
                self.solver = AdamsBashforth(params)

    def generate_sample(self,name,x0=None,plot=False):

        if x0 is None:
            self.solver.u0 = [np.random.uniform(-5, 5), np.random.uniform(-5, 5)]
            self.params.u0 = self.solver.u0

        def f(u,t):
            return [self.mu*(u[0] - 1./3*(u[0]**3)-u[1]),
                    1./self.mu*u[0],
            ]

        self.solver.set_f(f)
        self.solver.generateODE()
        if plot:
            self.solver.plot_solution()


        self.sample = self.solver.u.transpose()
        self.save_sample(name)


    def generate_dataset(self, num_samples, num_processes, x0=None, plot=False):
        args = [("sample_" + str(i) + ".npy", x0, plot) for i in range(num_samples)]

        start = time.perf_counter()

        with Pool(processes=num_processes) as pool:
            pool.starmap(self.generate_sample, args)

        finish = time.perf_counter()

        print("Program finished in " + str(finish - start) + " seconds")