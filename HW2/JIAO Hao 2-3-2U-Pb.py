import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import time

name=['U238','Pb206']

def fun(x,y,half_life=1.4e17):
    # the Half_life is in the unit of second
    dydx=np.zeros(2)
    dydx[0]=-y[0]/half_life
    dydx[1]=y[0]/half_life
    return dydx

y0=np.asarray([1,0]) # set the initial condition of y
x0=0
x1=1e18

t1=time.time()
rk4=integrate.solve_ivp(fun,[x0,x1],y0)
t2=time.time()
print('it takes ',t2-t1,'s to solve the ODE with RK4')
stiff=integrate.solve_ivp(fun,[x0,x1],y0,method='Radau')
t3=time.time()
print('it takes ',t3-t2,'s to solve the ODE implicitly' )

plt.subplot(211)
plt.plot(rk4.t,rk4.y[0,:],'-',rk4.t,rk4.y[1,:],'-')
plt.plot(stiff.t,stiff.y[0,:],'--',stiff.t,stiff.y[1,:],'--')
plt.legend(['U238(rk4)','Pb206(rk4)','U238(stiff)','Pb206(stiff)'],loc='best')
plt.ylabel('abundance')
plt.subplot(212)
plt.plot(rk4.t,rk4.y[1,:]/rk4.y[0,:],'-',stiff.t,stiff.y[1,:]/stiff.y[0,:],'--')
plt.legend(['rk4','stiff'],loc='best')
plt.xlabel('time / s')
plt.ylabel('ratio of Pb206 to U238')
plt.show()
