from DataGen import ThetaMethod
from DataGen import RK_explicit
import numpy as np
import matplotlib.pyplot as plt


########################################################
#               SIMPLE TEST - THETA METHOD             #
########################################################   
      
"""
x'= -y
y'= x
y(0) = 1
x(0) = 1

"""
    
T = 1
dt = 0.01
u0 = [1,1]
f = lambda v,t : np.array([-v[1],v[0]])
u_ex = lambda t: np.cos(t)-np.sin(t)
v_ex = lambda t: np.cos(t)+np.sin(t)



TM = ThetaMethod(T=T,dt=dt,u0=u0,f=f,eqtype='ODE',theta=0)
TM.generate()
#print(TM.u)
#print(TM.times)

#plt.plot(TM.times,TM.u[0,:])
#plt.plot(TM.times,u_ex(TM.times))
#plt.title("u")
#plt.show()


#plt.plot(TM.times,TM.u[1,:])
#plt.plot(TM.times,v_ex(TM.times))
#plt.title("v")
#plt.show()



########################################################
#               SIMPLE TEST - RUNGE KUTTA              #
######################################################## 

"""
x'= -x
x(0) = 1

"""

T = 1
dt = 0.1
u0 = 1
but_A = np.array([[0.0,0.0],[0.5,0.0]])
but_b = np.array([0.0,1.0])
but_c = np.array([0.0,0.5])
u0 = [1,1]
f = lambda v,t : np.array([-v[1],v[0]])
u_ex = lambda t: np.cos(t)-np.sin(t)
v_ex = lambda t: np.cos(t)+np.sin(t)

solver = RK_explicit(T=T,dt=dt,u0=u0, but_A=but_A,but_b=but_b,but_c=but_c, f=f,eqtype='ODE')   
solver.generate()

print(solver.u)

plt.plot(solver.times,solver.u[0,:])
plt.plot(solver.times,u_ex(solver.times))
plt.title("u")
plt.show()

plt.plot(solver.times,solver.u[1,:])
plt.plot(solver.times,v_ex(solver.times))
plt.title("v")
plt.show()