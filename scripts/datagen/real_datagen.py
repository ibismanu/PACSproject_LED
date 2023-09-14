
class DataGen():
    
    solver: GenerateParticle
    
    def GenerateCollection:
        for i in range(N):
            for j in range(i):
                t_begin = i
                t_end = i+2
                
                solver = ThetaMethod(T=T, dt=dt, u0=u0, eqtype=eqtype, theta=0.5, f=f)
                solver.generate()
                solver.plot_solution()
                
                grid[i,j] = solver.u
                
        dataset.append(grid)
        
        
    def save:
        np.save(dateset)