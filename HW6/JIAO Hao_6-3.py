import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

tau=1
n=10000000

def expon(x,tau):
    return np.exp(-x/tau)/tau

u=np.random.rand(n)
v=np.random.rand(n)

rat=v/u
accept=u<np.sqrt(expon(rat,tau))
x=rat[accept]

a,b=np.histogram(x,100)
aa=a/a[0]
bb=0.5*(b[1:]+b[:-1])
x1=np.linspace(0,np.max(x),1001)
y1=expon(x1,tau)

print('The accept rate of ratio-of-uniforms generator is',len(x)/n)

plt.ion()
plt.bar(b[:-1],aa,width=0.5*(b[1]-b[0]))
plt.plot(x1,y1,c='red',linewidth=1)
plt.title('ratio-of-uniforms generator')
plt.show()










