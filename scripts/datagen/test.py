from DataGen import ThetaMethod
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
print(TM.u)
print(TM.times)

plt.plot(TM.times,TM.u[0,:])
plt.plot(TM.times,u_ex(TM.times))
plt.title("u")
plt.show()


plt.plot(TM.times,TM.u[1,:])
plt.plot(TM.times,v_ex(TM.times))
plt.title("v")
plt.show()