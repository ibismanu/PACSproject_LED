from particle.generate_particle import GenerateParticle
from utilities.params import Params
import numpy as np
from abc import ABC, abstractmethod
from tqdm.auto import tqdm

import sys
sys.path.append('..')


class DataGen(ABC):

    solver: GenerateParticle
    sample: np.array
    dataset: list
    n_samples: int
    n_particles: int

    def __init__(self, n_particles: int, solver: GenerateParticle):
        self.n_particles = n_particles
        self.solver = solver

        shape = (n_particles, np.shape(solver.u)[0], np.shape(solver.u)[1])
        self.sample = np.zeros(shape=shape)
        self.dataset = []

    @abstractmethod
    def generate_sample(self):
        pass

    def save_sample(self, name):
        np.save('data/saved_particles/'+name, self.sample)

    def save_dataset(self, name, format='npy'):
        if format == 'npy':
            np.save('../../dataset/'+name, self.dataset)
        # TODO other formats
