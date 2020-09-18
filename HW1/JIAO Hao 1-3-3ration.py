import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as intp

def lorentz(x):
    y=[]
    for i in range(len(x)):
        y.append(1/(1+x[i]**2))
    return y

x0=np.linspace(-1,1,201)
ytrue=lorentz(x0)


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


########################################
n=2
m=3

x=np.linspace(-1,1,m+n-1)
y=lorentz(x)

p,q=rat_fit(x,y,n,m)
pred=rat_eval(p,q,x)
yrat=rat_eval(p,q,x0)

plt.subplot(421)
plt.plot(x0,ytrue,'-',x0,yrat,'-.')
plt.legend(['true','ration (2,3)'],loc='best')
plt.subplot(422)
plt.plot(x0,yrat-ytrue,'-')
plt.legend(['ration(2,3)'],loc='best')


########################################
n=3
m=3

x=np.linspace(-1,1,m+n-1)
y=lorentz(x)
x0=np.linspace(x[0],x[-1],201)
ytrue=lorentz(x0)

p,q=rat_fit(x,y,n,m)
pred=rat_eval(p,q,x)
yrat=rat_eval(p,q,x0)

plt.subplot(423)
plt.plot(x0,ytrue,'-',x0,yrat,'-.')
plt.legend(['true','ration (3,3)'],loc='best')
plt.subplot(424)
plt.plot(x0,yrat-ytrue,'-')
plt.legend(['ration(3,3)'],loc='best')

########################################
n=3
m=4

x=np.linspace(-1,1,m+n-1)
y=lorentz(x)
x0=np.linspace(x[0],x[-1],201)
ytrue=lorentz(x0)

p,q=rat_fit(x,y,n,m)
pred=rat_eval(p,q,x)
yrat=rat_eval(p,q,x0)

plt.subplot(425)
plt.plot(x0,ytrue,'-',x0,yrat,'-.')
plt.legend(['true','ration (3,4)'],loc='best')
plt.subplot(426)
plt.plot(x0,yrat-ytrue,'-')
plt.legend(['ration(3,4)'],loc='best')

########################################
n=4
m=5

x=np.linspace(-1,1,m+n-1)
y=lorentz(x)
x0=np.linspace(x[0],x[-1],201)
ytrue=lorentz(x0)

p,q=rat_fit(x,y,n,m)
pred=rat_eval(p,q,x)
yrat=rat_eval(p,q,x0)

plt.subplot(427)
plt.plot(x0,ytrue,'-',x0,yrat,'-.')
plt.legend(['true','ration (4,5)'],loc='best')
plt.subplot(428)
plt.plot(x0,yrat-ytrue,'-')
plt.legend(['ration(4,5)'],loc='best')

########################################


plt.show()



