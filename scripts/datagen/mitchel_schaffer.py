import numpy as np
from thetamethod import ThetaMethod
from rungekutta import RK_4


T = 500.0
dt = 0.1
u0 = np.array([0,0])

eqtype = 'ODE'

'''
u' = k*u*(u-a)(1-u)-v+I_app
v' = eps*(u-gammma*v)
'''
def f(u,t):
    I_app = I*(t>=t_begin and t<t_end)
    return [k*u[0]*(u[0]-alpha)*(1-u[0])-u[1]+I_app,eps*(u[0]-gamma*u[1])]

k = 8.0
alpha = 0.15
gamma = 0.1
eps = 0.01
I = 0.125

t_begin = 0
t_end = 2

def u_ex(t):
    return [0.5+0*t,0.5+0*t]

solver = ThetaMethod(T=T, dt=dt, u0=u0, eqtype=eqtype, theta=0.5, f=f)
solver.generate()
solver.plot_solution(u_ex)


# for i in range(N):
#     for j in range(i):
#         t_begin = i
#         t_end = i+2
        
#         solver = ThetaMethod(T=T, dt=dt, u0=u0, eqtype=eqtype, theta=0.5, f=f)
#         solver.generate()
#         solver.plot_solution()
        
#         casella[i,j] = solver.u




