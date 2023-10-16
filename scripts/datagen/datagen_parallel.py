from particle.generate_particle_parallel import GenerateParticle, Params
import numpy as np
from abc import ABC, abstractmethod
from tqdm.auto import tqdm

import sys
sys.path.append('..')


class DataGen(ABC):

    sample: np.array
    dataset: list
    sample: np.array
    n_particles: int

    def __init__(self, n_particles: int):

        self.n_particles = n_particles
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
