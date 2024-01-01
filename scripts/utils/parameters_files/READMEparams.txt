How to write the parameters file.
It's important to respect all the spaces as indicated:
T=...
dt=...
u0=[...,...]
solver=ThetaMethod
theta=...
tol=...

or
solver=RungeKutta 
A=[...,...] [...,...]
b=[...,...]
c=[...,...]

or 
solver=AdamsBashforth
order=...

or
solver=RK4, RKHeun, RKRalston
