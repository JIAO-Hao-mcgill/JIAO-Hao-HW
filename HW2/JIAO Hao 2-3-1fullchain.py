import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import time

name=['U238','Th234','Pa234','U234','Th230','Ra226','Rn222',\
      'Po218','Pb214','Bi214','Po214','Pb210','Bi210','Po210','Pb206']

def fun(x,y,half_life=[1.4e17, 2.1e6, 2.4e4, 7.7e12, 2.4e12, 5.0e10, 3.3e5,\
                       186, 1608, 1194, 1.6e-4, 7.0e8, 1.6e8, 1.2e7]):
    # the Half_life is in the unit of second
    n=len(half_life)
    dydx=np.zeros(n+1)
    dydx[0]=-y[0]/half_life[0]
    for i in range(1,n):
        dydx[i]=y[i-1]/half_life[i-1]-y[i]/half_life[i]
    dydx[-1]=y[-1]/half_life[-1]
    return dydx

y0=np.zeros(15) # set the initial condition of y
y0[0]=1
x0=0
x1=1e18

#t1=time.time()
#rk4=integrate.solve_ivp(fun,[x0,x1],y0)
t2=time.time()
#print(t2-t1)
stiff=integrate.solve_ivp(fun,[x0,x1],y0,method='Radau')
t3=time.time()
print('it takes ',t3-t2,'s to solve the ODE implicitly' )
