import numpy as np
from tqdm.auto import tqdm
import sys
sys.path.append('..')
from particle.generate_particle_parallel import GenerateParticle, Params
from datagen_parallel import DataGen
from multiprocessing import Pool
from particle.thetamethod import ThetaMethod
import time

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
    num_it: int         # number of iterations
    
    

    def __init__(self, eqtype, params,
                 k, alpha, epsilon,
                 I, gamma, grid_size:tuple, solver=None):

        n_particles = grid_size[0]*grid_size[1]

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
        self.num_it = int(params.T/params.dt)
        
        self.sample = np.zeros((self.num_it+1,grid_size[0],grid_size[1],len(params.u0)))

    def generate_sample(self, sample_id, x0=None, plot=False):

        #TODO metodo alternativo per calcolare I_app, griglia reale
        # dato iniziale: velocità di propagazione della corrente
        # t_begin = velocità*distanza
        # t_end = velocità*distanza + length_t 
        # loop sulla griglia per calcolare I_app
        if x0 is None:
            x0 = (np.random.randint(0,self.grid_size[0]),np.random.randint(0,self.grid_size[1]))

        t_begin = 0.
        delta_t = 50.
        length_t = 2

    
        current_explore = []
        explored = []
        future_explore = [(x0[0],x0[1],t_begin)]

        while future_explore:  # same as while future_explore != []
            current_explore = future_explore
            future_explore = []

            for point in current_explore:
                above = (point[0]-1, point[1], point[2]+delta_t)
                below = (point[0]+1, point[1], point[2]+delta_t)
                left = (point[0], point[1]-1, point[2]+delta_t)
                right = (point[0], point[1]+1, point[2]+delta_t)

                if above[0] >= 0 and (above[0], above[1]) not in explored and above not in future_explore:
                    future_explore.append(above)

                if below[0] < self.grid_size[0] and (below[0], below[1]) not in explored and below not in future_explore:
                    future_explore.append(below)

                if left[1] >= 0 and (left[0], left[1]) not in explored and left not in future_explore:
                    future_explore.append(left)

                if right[1] < self.grid_size[1] and (right[0], right[1]) not in explored and right not in future_explore:
                    future_explore.append(right)

                def f(u, t):
                    I_app = self.I * \
                        (t >= point[2] and t < point[2]+length_t)
                    res = [self.k*u[0]*(u[0] - self.alpha) * (1 - u[0]) - u[1] + I_app,
                            self.epsilon*(u[0] - self.gamma*u[1])]
                    return np.array(res)
                
                solver = ThetaMethod(eqtype=self.eqtype, params=self.params)

                solver.f = f
                solver.reset()
                solver.generate()
                if plot:
                    solver.plot_solution()

                self.sample[:, point[0], point[1]] = solver.u.transpose()

                explored.append((point[0], point[1]))

        filename = f'sample_{sample_id}.npy'
        np.save(filename, self.sample)

    def generate_dataset_parallel(self, num_samples, num_processes):

        # Create a pool of processes
        start_time = time.perf_counter()
        with Pool(processes=num_processes) as pool:
            # Generate samples in parallel
            pool.map(self.generate_sample, range(num_samples))
        finish_time = time.perf_counter()
        print('Program finished in {} seconds'.format(finish_time-start_time))

