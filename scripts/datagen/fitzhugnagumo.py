import numpy as np
import time
from multiprocessing import Pool

from scripts.datagen.datagen import DataGen
from scripts.utils.params import SolverParams
from scripts.particle.thetamethod import ThetaMethod


def f_temp(x):
    return np.array([0])


class FitzhugNagumo(DataGen):
    params: SolverParams

    def __init__(
        self, params, k, alpha, epsilon, I, gamma, grid_size, solver_name
    ):
        self.params = params
        self.k = k
        self.alpha = alpha
        self.epsilon = epsilon
        self.I = I
        self.gamma = gamma
        self.grid_size = grid_size
        self.num_it = int(params.final_time / params.time_step)

        self.sample = np.zeros(
            (self.num_it + 1, grid_size[0], grid_size[1], len(params.u0))
        )

        match solver_name:
            case "thetamethod":
                self.solver = ThetaMethod(
                    final_time=params.final_time,
                    time_step=params.time_step,
                    u0=params.u0,
                    f=f_temp,
                    theta=params.theta,
                    tol=params.tol,
                )
            case _:
                pass

    def generate_sample(self, name, x0=None, plot=False):
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
                self.solver.generateODE()
                if plot:
                    self.solver.plot_solution()

                self.sample[:,i,j] = self.solver.u.transpose()


        self.save_sample(name)

    def generate_dataset(self, num_samples, num_processes, x0=None, plot=False):
        args = [("sample_" + str(i) + ".npy", x0, plot) for i in range(num_samples)]

        start = time.perf_counter()

        with Pool(processes=num_processes) as pool:
            pool.starmap(self.generate_sample, args)

        finish = time.perf_counter()

        print("Program finished in " + str(finish - start) + " seconds")
