from datagen import DataGen
from utilities.utils import valid_butcher

from abc import abstractmethod
import numpy as np


class RungeKutta(DataGen):

    but_A: np.array
    but_b: np.array
    but_c: np.array
    s: int

    def __init__(self, T, dt, u0, eqtype, but_A, but_b, but_c, f=None, M=None, A=None, F=None):

        self.s = len(but_c)

        valid_butcher(but_A, but_b, but_c, self.s)

        self.but_A = but_A
        self.but_b = but_b
        self.but_c = but_c

        super().__init__(T=T, dt=dt, u0=u0, eqtype=eqtype, f=f, M=M, A=A, F=F)
        
    @abstractmethod
    def generate(self):
        pass
    
