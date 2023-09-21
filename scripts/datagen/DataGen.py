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
        
        #scritto male
        self.sample = np.zeros((N_particles,np.shape(solver.u)[0],np.shape(solver.u)[1]))
        if create_dataset:
            #self.dataset = np.zeros((N_sample,N_particles,np.shape(solver.u)[0],np.shape(solver.u)[1]))
            self.dataset = []

    def GenerateSample(self):
        pass

    #TODO aggiungere altri formati
    def save_sample(self):
        np.save(self.sample)
        
    def save_dataset(self, filename, format='npy'):
        if format == 'npy':
            np.save(filename, self.dataset)
        #TODO other formats
    

class Mitchell_Schaeffer(DataGen):

    k: float
    alpha: float
    eps: float
    I: float
    gamma: float    
    grid_size: int
    
    def __init__(self, N_sample, grid_size, solver, x0, k, alpha, eps, I, gamma,create_dataset=True):
        self.k=k
        self.alpha=alpha
        self.eps=eps
        self.I=I
        self.gamma=gamma
        self.x0=x0
        self.create_dataset=create_dataset
        
        self.solver=solver
        
        self.grid_size=grid_size
        self.N_particles = grid_size**2
        self.N_sample = N_sample
        self.sample = np.zeros((self.grid_size,self.grid_size,np.shape(solver.u)[0],np.shape(solver.u)[1]))
        if self.create_dataset:
            #self.dataset = np.zeros((N_sample,grid_size,grid_size,np.shape(solver.u)[0],np.shape(solver.u)[1]))
            self.dataset = []
            
    def GenerateSample(self, filename, format='npy', plot=False):

        t_begin = 0
        delta_t = 20
        length_t = 2

        current_explore=[]
        explored = []
        future_explore=[(self.x0[0],self.x0[1],t_begin)]

        while future_explore != []:
            current_explore = future_explore
            future_explore = []
            for point in current_explore:
                point_above = (point[0],point[1]-1,point[2]+delta_t)
                point_below = (point[0],point[1]+1,point[2]+delta_t)
                point_right = (point[0]+1,point[1],point[2]+delta_t)
                point_left = (point[0]-1,point[1],point[2]+delta_t)

                if point_above[1]>=0 and (point_above[0],point_above[1]) not in explored:
                    future_explore.append(point_above)
                if point_below[1]<self.grid_size and (point_below[0],point_below[1]) not in explored:
                    future_explore.append(point_below)
                if point_left[0]>=0 and (point_left[0],point_left[1]) not in explored:
                    future_explore.append(point_left)
                if point_right[0]<self.grid_size and (point_right[0],point_right[1]) not in explored:
                    future_explore.append(point_right)

                def f(u,t):
                    I_app = self.I*(t>=point[2] and t<point[2]+length_t)
                    return np.array([self.k*u[0]*(u[0]-self.alpha)*(1-u[0])-u[1]+I_app,self.eps*(u[0]-self.gamma*u[1])])
                
                self.solver.f=f
                
                self.solver.reset()
                self.solver.generate()
                if plot:
                    self.solver.plot_solution()
                self.sample[point[0],point[1],:,:]=self.solver.u
                explored.append((point[0],point[1]))
                
        if self.create_dataset:
            self.dataset.append(self.sample)
        
        self.save_dataset(filename, format) 
