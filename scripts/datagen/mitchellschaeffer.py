import numpy as np

import sys
sys.path.append('..')
from particle.generate_particle import GenerateParticle
from datagen.datagen import DataGen

class MitchellSchaeffer(DataGen):

    # TODO better names
    k: float
    alpha: float
    epsilon: float
    I: float
    gamma: float
    grid_size: tuple
    x0: list

    def __init__(self, solver: GenerateParticle,
                 k: float, alpha: float, epsilon: float,
                 I: float, gamma: float, grid_size: tuple):

        n_particles = grid_size[0]*grid_size[1]

        super().__init__(n_particles, solver)

        self.k = k
        self.alpha = alpha
        self.epsilon = epsilon
        self.I = I
        self.gamma = gamma
        self.grid_size = grid_size
        
        self.sample = np.zeros((solver.num_it+1,grid_size[0],grid_size[1],len(solver.u)))

    def generate_sample(self, x0:list, plot=False):

        t_begin = 0.
        delta_t = 20.
        length_t = 2.

    
        current_explore = []
        explored = []
        future_explore = [(x0[0],x0[1],t_begin)]

        while future_explore:  # same as while future_explore != []
            current_explore = future_explore
            future_explore = []

            for point in current_explore:
                above = (point[0], point[1]-1, point[2]+delta_t)
                below = (point[0], point[1]+1, point[2]+delta_t)
                left = (point[0]-1, point[1], point[2]+delta_t)
                right = (point[0]+1, point[1], point[2]+delta_t)

                if above[1] >= 0 and (above[0], above[1]) not in explored:
                    future_explore.append(above)
                if below[1] < self.grid_size[0] and (below[0], below[1]) not in explored:
                    future_explore.append(below)
                if left[0] >= 0 and (left[0], left[1]) not in explored:
                    future_explore.append(left)
                if right[0] < self.grid_size[1] and (right[0], right[1]) not in explored:
                    future_explore.append(right)

                def f(u, t):
                    I_app = self.I * \
                        (t >= point[2] and t < point[2]+length_t)
                    res = [self.k*u[0]*(u[0] - self.alpha) * (1 - u[0]) - u[1] + I_app,
                            self.epsilon*(u[0] - self.gamma*u[1])]
                    return np.array(res)

                self.solver.f = f
                self.solver.reset()
                self.solver.generate()
                if plot:
                    self.solver.plot_solution()
                
                # print(np.shape(self.sample))
                # print(np.shape(self.solver.u))

                self.sample[:, point[0], point[1]] = self.solver.u.transpose()
                # forse sbagliato

                explored.append((point[0], point[1]))

        self.dataset.append(self.sample)

    def generate_dataset(self, n_samples, filename, format='npy'):
        for i in range(n_samples):
            x0 = (0,0) # to be randomized
            self.generate_sample(x0)
        self.save_dataset(filename, format)
