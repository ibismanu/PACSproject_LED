import numpy as np
from abc import ABC, abstractmethod


class DataGen(ABC):
    def __init__(self):
        self.sample = []

    @abstractmethod
    def generate_sample(self):
        pass

    def save_sample(self, name):
        match name[-4:]:
            case ".npy":
                np.save("data/samples/" + name, self.sample)
            case ".npz":
                np.savez_compressed("data/samples/" + name, my_data=self.sample)
            case _:
                pass
