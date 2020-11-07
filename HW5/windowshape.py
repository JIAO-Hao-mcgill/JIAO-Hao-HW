import numpy as np
from matplotlib import pyplot as plt


n=131072

x=np.linspace(-1,1,n)
win1=0.5+0.5*np.cos(np.pi*x)
win2=1-x*x
win3=np.ones(n)
for i in range(n):
    if np.abs(x[i])>0.5:
        win3[i]=0.5+0.5*np.cos(2*np.pi*(np.abs(x[i])-0.5))

plt.plot(x,win1,x,win2,x,win3)
plt.legend(['cos','welch','tukey'],loc='best')
plt.show()
