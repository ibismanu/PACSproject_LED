from abc import ABC, abstractmethod
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import numbers
import utils

#elenco idee
    #i parametri della simulazione in un file
"""

MU'+AU=F

U'=f(U)
maybe de-generalize to U'=f(u)?

"""



class DataGen(ABC): #risolve ODE/sistema di ODE
    # u: numpy vector, contains solution, size (T/dt,N), N being problem size
    
    T: int          #final time
    dt: float       #time step    
    eqtype: str    #equation to solve, can be "ODE" or "PDE"
    
    #initial assumpion to improve: F/f depend on time
    def __init__(self, T, dt, u0, M=None,A=None,F=None,f=None, eqtype=""):
        self.eqtype=eqtype
        self.dt = dt
        numIT = np.int32(T/dt)
        if T != numIT*dt:
            T = numIT*dt
            print("final time reached: ", T)
        self.numIT = numIT+1
        self.T = T
        if eqtype == "ODE":
            self.f = utils.to_numpy(f)
            
            if isinstance(u0, numbers.Number):
                L = 1
                self.u = np.zeros((L,self.numIT))
                self.u = np.reshape(self.u,[1,self.numIT])
            else:
                L = len(u0)
                self.u = np.zeros((L,self.numIT))
                
        elif eqtype == "PDE":
            self.M = M
            self.F = F
            self.A = A
            self.u = np.zeros(M.size()[0],self.numIT)
        else:
            raise ValueError("equation type not supported")
        self.u[:,0] = u0
        self.times = np.arange(0,T+dt,dt)
        
    
    @classmethod
    def FromODE(cls, T, dt, u0, f):
        #u'=f(x,u)
        return cls(T=T,dt=dt,u0=u0,f=f,eqtype="ODE")
        
    @classmethod
    def FromPDE(cls, T, dt, u0, FEMobj):
        #MU'+AU=F
        M,A,F = FEMobj.generate()
        return cls(T=T, dt=dt, u0=u0, M=M, A=A, F=F, eqtype="PDE")
        

    @classmethod
    def FromFile(cls, url):
        #read from file
        #params = read_from_file
        
        #return cls(params)
        return 1
        
    @abstractmethod
    def generate(self):
        pass

            
    """ @abstractmethod
    def solve(self):
        pass
    
    @abstractmethod
    def linear_solve(self):
        pass
        """
    def save(self, csv=True,npy=True):
        if csv:
            return 1
        if npy:
            return 1


#V = DataGen.FromFile("...")


class RungeKutta(DataGen):
    
    but_A : np.array
    but_b : np.array
    but_c : np.array
    s : int
    def __init__(self, T, dt, u0, but_A, but_b, but_c, M=None,A=None,F=None,f=None, eqtype=""):
        
        
        is_valid_b = np.sum(but_b)==1
        is_valid_c = np.array_equal(np.array([np.sum(row) for row in but_A]),but_c)
        len_b = len(but_b)
        size_A = but_A.shape
        
        self.s = len(but_c)
        
        assert is_valid_b and is_valid_c and len_b==self.s and\
            size_A[0] == self.s and size_A[1] == self.s, "invalid butcher array"
        
        self.but_b = but_b
        self.but_c = but_c

        
        super().__init__(T=T,dt=dt,u0=u0,M=M,A=A,F=F,f=f,eqtype=eqtype)
    
    @abstractmethod
    def generate(self):
        pass
       
class RK_explicit(RungeKutta):
    def __init__(self, T, dt, u0, but_A, but_b, but_c, M=None,A=None,F=None,f=None, eqtype=""):
        super().__init__(T=T,dt=dt,u0=u0,but_A=but_A,but_b=but_b,but_c=but_c,M=M,A=A,F=F,f=f,eqtype=eqtype)
        
        is_valid_A = np.array_equal(but_A, np.tril(but_A,-1))
        assert is_valid_A==True, "explicit method called but implicit butcher array given"
        
        self.but_A = but_A
    
    def generate(self):
        if self.eqtype=="ODE":
            return self.generateODE()
        elif self.eqtype=="PDE":
            return self.generatePDE()
    
    def generatePDE(self):
        return 1
    
    def generateODE(self):
        for n in range(self.numIT-1):
            k = np.zeros((self.s,len(self.u[:,0])))
            for i in range(self.s):
                k[i,:] = self.f(self.u[:,n]+self.dt*np.dot(self.but_A[i,:],k),\
                              self.times[n]+self.but_c[i]*self.dt)
            self.u[:,n+1] = self.u[:,n]+self.dt*np.dot(self.but_b,k)
            
        
        
     
#class RK_semi_implicit(RungeKutta):
        
#class RK_implicit(RungeKutta):
    
#class RK23
#     def __init__(self):
#         but = ...
#         super().__init__(but)
         
#class RK4(  ):
   

#solver = RK23
     
class ThetaMethod(DataGen):
    def __init__(self, T, dt, u0, M=None,A=None,F=None,f=None, eqtype="", theta=0):
        super().__init__(T=T,dt=dt,u0=u0,M=M,A=A,F=F,f=f,eqtype=eqtype)
        self.theta = theta
    
    def generate(self):
        if self.eqtype=="ODE":
            return self.generateODE()
        elif self.eqtype=="PDE":
            return self.generatePDE()
        
    """
    MU'=f(U)
    M(U*-U)/dt=af(U*,t*)+(1-a)f(U,t)
    MU*-af(U*)*dt-MU+dt(1-a)f(U)=0
    """
        
    def generateODE(self):
        
        for k in range(self.numIT-1):
            rhs=self.u[:,k]+self.dt*(1-self.theta)*self.f(self.u[:,k],self.times[k])
            def g(v):
                return v-self.theta*self.f(v,self.times[k+1])*self.dt-rhs
            
            #FixedPoint(g,toll=1e-2)
            #u(:,k+1) = Solver(g,toll=1e-4)
            size = np.mean(np.abs(self.u[:,k]))
            self.u[:,k+1] = sp.optimize.newton(g,x0=self.u[:,k],tol=size*1e-2)
        #return u,times
    
    """
    MU'+AU=F
    M(U*-U)/dt+aAU*+(1-a)AU=aF*+(1-a)F
    MU*+adtAU*=adtF*+(1-a)dtF+MU-(1-a)dtAU
    """
    
    def generatePDE(self):
        
        for k in range(self.numIT-1):
            matrix = self.M+self.theta*self.dt*self.A
            rhs = self.theta*self.dt*self.F(self.times(k+1))+\
            (1-self.theta)*self.dt*self.F(self.times(k+1))+\
                self.M*self.U-(1-self.theta)*self.dt*self.A*self.u[:,k]
            
            self.u[:,k+1]=np.linalg.solve(matrix,rhs)
        #return self.u, self.times
        
    
class Adams_name_pending(DataGen):
    def __init__(self, T, dt):
        super().__init__(T, dt)
    
    def generate(self):
        #implement adams/...
        return 1
        
        
        


