import numpy as np
import time
from multiprocessing import Pool

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


    # Create a dataset by generating multiple samples with different initial data
    def generate_dataset(self, num_samples, num_processes, x0=None, plot=False):
        args = [("sample_" + str(i) + ".npy", x0, plot) for i in range(num_samples)]

        start = time.perf_counter()

        # The generation of samples is parallelizzed
        with Pool(processes=num_processes) as pool:
            pool.starmap(self.generate_sample, args)

        finish = time.perf_counter()

        print("Program finished in " + str(finish - start) + " seconds")