import numpy as np
import ctypes
import numba as nb
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mylib=ctypes.cdll.LoadLibrary("libc.so")
#mylib=ctypes.cdll.LoadLibrary("libc.dll")
rand=mylib.rand
rand.argtypes=[]
rand.restype=ctypes.c_int


@nb.njit
def get_rands_nb(vals):
    n=len(vals)
    for i in range(n):
        vals[i]=rand()
    return vals

def get_rands(n):
    vec=np.empty(n,dtype='int32')
    get_rands_nb(vec)
    return vec


n=300000000
vec=get_rands(n*3)
#vv=vec&(2**16-1)

vv=np.reshape(vec,[n,3])
vmax=np.max(vv,axis=1)

maxval=1e8
vv2=vv[vmax<maxval,:]



plt.ion()
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.plot(vv2[:,0],vv2[:,1],vv2[:,2],',')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
