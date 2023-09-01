from DataGen import ThetaMethod
from DataGen import RK_explicit
import DataGen
import numpy as np
import matplotlib.pyplot as plt

"""
x'= -y
y'= x
y(0) = 1
x(0) = 1

"""

"""
x'= -x
x(0) = 1

"""

T = 1
dt = 0.01
u0_1 = 1
u0_2 = [1,1]

f_1 = lambda v,t : np.array(-v)
f_2 = lambda v,t : np.array([-v[1],v[0]])

u_ex2 = lambda t: np.cos(t)-np.sin(t)
v_ex2 = lambda t: np.cos(t)+np.sin(t)

u_ex1 = lambda t:np.exp(-t)

########################################################
#               SIMPLE TEST - THETA METHOD             #
########################################################   


TM = ThetaMethod(T=T,dt=dt,u0=u0_1,f=f_1,eqtype='ODE',theta=0)
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




but_A = np.array([[0.0,0.0],[0.5,0.5]])
but_b = np.array([0.5,0.5])
but_c = np.array([0.0,1.0])

solver = DataGen.RK_implicit(T=T,dt=dt,u0=u0_2, but_A=but_A,but_b=but_b,but_c=but_c, f=f_2,eqtype='ODE')   
#solver = DataGen.RK4(T=T,dt=dt,u0=u0_1,f=f_1,eqtype='ODE')
solver.generate()

# plt.plot(solver.times,solver.u[0,:])
# plt.plot(solver.times,u_ex1(solver.times))
# plt.title("u")
# plt.show()

plt.plot(solver.times,solver.u[0,:])
plt.plot(solver.times,u_ex2(solver.times))
plt.title("u")
plt.show()


plt.plot(solver.times,solver.u[1,:])
plt.plot(solver.times,v_ex2(solver.times))
plt.title("v")
plt.show()






