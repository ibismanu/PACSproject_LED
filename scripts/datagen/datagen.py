import numpy as np
from abc import ABC, abstractmethod


# Abstract class, used to generate the datasets on which the LED model will be trained and tested
class DataGen(ABC):
    def __init__(self):
        self.sample = []

    @abstractmethod
    def generate_sample(self):
        pass

    def save_sample(self, name):
        if name[-4:] == ".npy":
            np.save("../../dataset/samples/" + name, self.sample)
        elif name[-4:] == ".npz":
            np.savez_compressed("../../dataset/samples/" + name, my_data=self.sample)
        else:
            pass
