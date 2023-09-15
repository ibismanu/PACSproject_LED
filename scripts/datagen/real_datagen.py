from thetamethod import ThetaMethod
from GenerateParticle import GenerateParticle
from abc import ABC, abstractmethod
import numpy as np

class DataGen(ABC):
    
    solver: GenerateParticle
    sample: np.array
    dataset: list
    N_sample: int
    N_particles: int
    create_dataset: bool
    
    def __init__(self,N_sample,N_particles,solver,create_dataset=True):
        
        self.N_sample=N_sample
        self.N_particles=N_particles
        self.create_dataset=create_dataset
        
        self.sample = np.zeros((N_particles,np.shape(solver.u)[0],np.shape(solver.u)[1]))
        if create_dataset:
            #self.dataset = np.zeros((N_sample,N_particles,np.shape(solver.u)[0],np.shape(solver.u)[1]))
            self.dataset = []

    def GenerateSample(self):
        pass

    #TODO aggiungere altri formati
    def save_sample(self):
        np.save(self.sample)
        
    def save_datast(self):
        np.save(self.dataset)
    

class Mitchel_Schaffer(DataGen):

    k: float
    alpha: float
    eps: float
    I: float
    gamma: float    
    grid_size: int
    
    def __init__(self, N_sample, grid_size, solver, k, alpha, eps, I, gamma):
        self.k=k
        self.alpha=alpha
        self.eps=eps
        self.I=I
        self.gamma=gamma
        
        self.solver=solver
        
        self.grid_size=grid_size
        self.N_particles = grid_size**2
        self.N_sample = N_sample
        self.sample = np.zeros((grid_size,grid_size,np.shape(solver.u)[0],np.shape(solver.u)[1]))
        if self.create_dataset:
            #self.dataset = np.zeros((N_sample,grid_size,grid_size,np.shape(solver.u)[0],np.shape(solver.u)[1]))
            self.dataset = []
            
    def GenerateSample(self):
        
        2 1 2
        1 0 1
        2 1 2
        
        current_explore=[]
        future_explore=[(x0,y0)]
        t_begin=0
        t_end=2
        while future_explore != []:
            current_explore = future_explore
            future_explore = []
            for point in current_explore:
                if point_above isvalid and not expored:
                    future_explore.add(point_above)
                ...
                def f, solver, ..
            t_begin+=2
            t_end+=2
                
                
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                
                t_begin = i
                t_end = i+2
                
                def f(u,t):
                    I_app = self.I*(t>=t_begin and t<t_end)
                    return [self.k*u[0]*(u[0]-self.alpha)*(1-u[0])-u[1]+I_app,self.eps*(u[0]-self.gamma*u[1])]
                
                self.solver.f=f
                
                self.solver.reset()
                self.solver.generate()
                #solver.plot_solution()
                self.sample[i,j,:,:]=self.solver.u
                
        if self.create_dataset:
            self.dataset.append(self.sample)

solver = ThetaMethod(T=500, dt=0.1, u0=np.array([0,0]), eqtype='ODE', theta=0.5)

generator = DataGen(10,100,solver)
print(np.shape(generator.dataset))

# Mitchel_Schaffer(ThetaMethod(dt,T,eqtype,f=None))



# for i in range(N):
#     for j in range(i):
#         t_begin = i
#         t_end = i+2
        
#         solver = ThetaMethod(T=T, dt=dt, u0=u0, eqtype=eqtype, theta=0.5, f=f)
#         solver.generate()
#         solver.plot_solution()
        
#         grid[i,j] = solver.u
        
# dataset.append(grid)