import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as intp

x=np.linspace(-1,1,5)
def lorentz(x):
    y=[]
    for i in range(len(x)):
        y.append(1/(1+x[i]**2))
    return y
y=lorentz(x)

x0=np.linspace(x[0],x[-1],1001)

ytrue=lorentz(x0)

########################################
# polynomial interpolation
yline=intp.interp1d(x,y)

dline=0
for i in range(len(x0)):
    dlinei=np.abs(ytrue[i]-yline(x0)[i])
    if(dline<dlinei):
        dline=dlinei
print('The maximal error of polynomial interpolation:',dline)

########################################
# cubic spline interpolation
spln=intp.splrep(x,y)
ycubic=intp.splev(x0,spln)
dcubic=0
for i in range(len(x0)):
    dcubici=np.abs(ytrue[i]-ycubic[i])
    if(dcubic<dcubici):
        dcubic=dcubici
print('The maximal error of cubic spline interpolation:',dcubic)

########################################
# rational function interpolation
def rat_eval(p,q,x):
    top=0
    for i in range(len(p)):
        top=top+p[i]*x**i
    bot=1
    for i in range(len(q)):
        bot=bot+q[i]*x**(i+1)
    return top/bot
def rat_fit(x,y,n,m):
    assert(len(x)==n+m-1)
    assert(len(y)==len(x))
    mat=np.zeros([n+m-1,n+m-1])
    for i in range(n):
        mat[:,i]=x**i
    for i in range(1,m):
        mat[:,i-1+n]=y*x**i
        mat[:,i-1+n]=-mat[:,i-1+n]
    pars=np.dot(np.linalg.inv(mat),y)
    p=pars[:n]
    q=pars[n:]
    return p,q
n=3
m=3
p,q=rat_fit(x,y,n,m)
pred=rat_eval(p,q,x)
yrat=rat_eval(p,q,x0)
drat=0
for i in range(len(x0)):
    drati=np.abs(ytrue[i]-yrat[i])
    if(drat<drati):
        drat=drati
print('The maximal error of (4,5) rational function interpolation:',drat)



#plt.plot(x,y,'o',x0,ytrue,'-',x0,yline(x0),'--',x0,ycubic,'-.',x0,yrat,':')
#plt.legend(['data','ture','linear','cubic','ration'],loc='best')
#plt.show()

plt.plot(x0,yline(x0)-ytrue,'--',x0,ycubic-ytrue,'-.',x0,yrat-ytrue,':')
plt.legend(['linear','cubic','ration'],loc='best')
plt.show()
