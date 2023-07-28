from abc import ABC, abstractmethod
import numpy

#elenco idee
    #i parametri della simulazione in un file
"""

MU'+AU=F

MU'=f(U)

"""



class DataGen(ABC): #risolve ODE/sistema di ODE
    # u: numpy vector, contains solution
    
    T: int          #final time
    dt: float       #time step    
    eqtype: str    #equation to solve, can be "ODE" or "PDE"
    
    def __init__(self, T, dt, M=None,A=None,F=None,f=None, eqtype=""):
        if eqtype == "ODE":
            #initialize ODE
        elif eqtype == "PDE":
            #initialize PDE
        else:
            raise ValueError("equation type not supported")
        
    
    @classmethod
    def FromODE(cls, f):
        #u'=f(x,u)
        return cls(T,dt,f="...",eqtype="ODE")
        
    @classmethod
    def FromPDE(cls, FEMobj):
        #MU'+AU=F
        M,A,F = FEMobj.generate()
        return cls(T,dt, M=M, A=A, F=F, eqtype="PDE")
        

    @classmethod
    def FromFile(cls, url):
        #read from file
        params = read_from_file
        
        return cls(params)
        
    #@abstractmethod
    def generate(self):
        #if linear:
        #    linear_solve();
        #else 
        #    solve();
            
   """ @abstractmethod
    def solve(self):
        pass
    
    @abstractmethod
    def linear_solve(self):
        pass
        """
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
        
        
class ThetaMethod(DataGen):
    def __init__(self, "params", theta=0):
        super().__init__("params")
        self.theta = theta
    
    def generate(self):
        #implement Theta method
        
        
class Adams_name_pending(DataGen):
    def __init__(self, T, dt):
        super().__init__(T, dt)
    
    def generate(self):
        #implement adams/...
        
        