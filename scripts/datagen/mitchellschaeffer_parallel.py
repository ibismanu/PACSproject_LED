import numpy as np
from tqdm.auto import tqdm
import sys

sys.path.append("..")
from particle.generate_particle_parallel import GenerateParticle, Params
from datagen_parallel import DataGen
from multiprocessing import Pool
from particle.thetamethod import ThetaMethod
import time
from functools import partial


class MitchellSchaeffer(DataGen):
    # TODO better names
    eqtype: str
    params: Params
    k: float
    alpha: float
    epsilon: float
    I: float
    gamma: float
    grid_size: tuple
    solver: str
    num_it: int  # number of iterations

    def __init__(
        self, eqtype, params, k, alpha, epsilon, I, gamma, grid_size: tuple, solver=None
    ):
        n_particles = grid_size[0] * grid_size[1]

        super().__init__(n_particles)

        self.eqtype = eqtype
        self.params = params
        self.k = k
        self.alpha = alpha
        self.epsilon = epsilon
        self.I = I
        self.gamma = gamma
        self.grid_size = grid_size
        self.solver = solver
        self.num_it = int(params.T / params.dt)

        self.sample = np.zeros(
            (self.num_it + 1, grid_size[0], grid_size[1], len(params.u0))
        )

    def generate_sample(self, args):
        sample_id = args[0]
        x0 = args[1]
        plot = args[2]

        if sample_id == 1:
            print("Inside proc 1")

        t_begin = 0.0
        v = 0.005
        length_t = 2.0

        hx = 1.0 / self.grid_size[0]
        hy = 1.0 / self.grid_size[1]

        if x0 is None:
            x0 = (np.random.uniform(0, 1), np.random.uniform(0, 1))

        for i in range(self.grid_size[0]):
            x = i * hx
            for j in range(self.grid_size[1]):
                y = j * hy
                distance = np.sqrt((x - x0[0]) ** 2 + (y - x0[1]) ** 2)
                t_begin = distance / v
                t_end = t_begin + length_t

                def f(u, t):
                    I_app = self.I * (t >= t_begin and t < t_end)
                    res = [
                        self.k * u[0] * (u[0] - self.alpha) * (1 - u[0]) - u[1] + I_app,
                        self.epsilon * (u[0] - self.gamma * u[1]),
                    ]
                    return np.array(res)

                solver = ThetaMethod(eqtype=self.eqtype, params=self.params)

                solver.f = f
                solver.reset()
                solver.generate()
                if plot:
                    solver.plot_solution()

                self.sample[:, i, j] = solver.u.transpose()

        filename = f"sample_{sample_id}.npz"
        np.savez_compressed(filename, self.sample)

    def generate_dataset_parallel(
        self, num_samples, num_processes, x0=None, plot=False
    ):
        arg_list = [(sample_id, x0, plot) for sample_id in range(num_samples)]

        # Create a pool of processes
        start_time = time.perf_counter()
        with Pool(processes=num_processes) as pool:
            # Generate samples in parallel
            pool.map(self.generate_sample, arg_list)
        finish_time = time.perf_counter()
        print("Program finished in {} seconds".format(finish_time - start_time))
