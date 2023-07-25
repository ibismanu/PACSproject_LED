from abc import ABC, abstractmethod
import numpy

#elenco idee
    #i parametri della simulazione in un file

class DataGen(ABC):
    # u: numpy vector, contains solution
    
    def __init__(self):
        u=np.zeros()
    
    @classmethod
    def FromFile(cls, url):
        #read from file
        params = read_from_file
        
        return cls(params)
        
    @abstractmethod
    def generate(self):
        pass
    
    def save(self, csv=True,npy=True):
        if csv:
            #save in csv
        if npy:
            #save in npy


#V = DataGen.FromFile("...")

class RungeKutta(DataGen):
    def __init__(self):
        super().__init__()
    
    def generate(self):
        #implement Runge Kutta
        
        
        