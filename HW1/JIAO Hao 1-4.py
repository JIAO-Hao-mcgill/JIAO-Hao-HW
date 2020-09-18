import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate as intg

N=101 # number of integrate points

def fun(theta,z):
    q=1  # for simplification, set charge density = 1
    R=1  # for simplification
    if (z==R) and (theta==0):
        return 0
        # avoid the effect of singularity
        # otherwise, the code will have problem!
    y1=R**2*q*(z-R*np.cos(theta))*np.sin(theta)
    y2=R**2+z**2-2*R*z*np.cos(theta)
    y2=y2*np.sqrt(y2)
    y=y1/y2
    return y
# ignore the factor 1/(4*np.pi*epsilon_0)

def myint_linear(z,n): # z as a number
    theta=np.linspace(0,np.pi,n)
    intg=0.5*(fun(theta[0],z)+fun(theta[-1],z))
    for i in range(1,n-1):
        intg=intg+fun(theta[i],z)
    return intg*(theta[1]-theta[0])

def myint_simpson(z,n):
    theta=np.linspace(0,np.pi,n)
    intg=fun(theta[0],z)+fun(theta[-1],z)
    for i in range(1,n-1,2):
        intg=intg+4*fun(theta[i],z)
    for i in range(2,n-2,2):
        intg=intg+2*fun(theta[i],z)
    return intg*(theta[1]-theta[0])/3


z0=np.linspace(0,3,301)

intg_quad=[]
for i in range(len(z0)):
    intg_quad.append(intg.quad(lambda x : fun(x,z0[i]), 0, np.pi)[0])

my_linear=[]
for i in range(len(z0)):
    my_linear.append(myint_linear(z0[i],N))

my_simpson=[]
for i in range(len(z0)):
    my_simpson.append(myint_linear(z0[i],N))


plt.plot(z0,intg_quad,'-',z0,my_linear,'--',z0,my_simpson,':')
plt.legend(['integrate.quad','linear','simpson'],loc='best')
plt.show()

# There is not obvious different between Linear and Simpson integrate,
# but the number of point make difference.

# there is a singularity in the integrand.
# In my integrator, this singularity is important; but quad do not care it.
