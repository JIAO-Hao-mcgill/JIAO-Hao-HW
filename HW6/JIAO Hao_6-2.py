import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def expon(x,tau):
    return np.exp(-x/tau)/tau

def expon_trans(n,tau):
    r=np.random.rand(n)
    return -tau*np.log(1-r)
def Lorentz_trans(n):  # only return positive number
    r=np.random.rand(n)
    return np.abs(np.tan(np.pi*(r-0.5)))
def PowerLaw_trans(n,alpha):
    r=np.random.rand(n)
    return pow(1-r,1/(1+alpha))-1

def expon_reject(n,tau,upper,distr_type='Lorentz'):
    if distr_type=='Lorentz':
        x=Lorentz_trans(n)
        xx=x[x<upper]
        accept_prob=expon(xx,tau)/(1/(1+xx*xx))
    if distr_type=='PowerLaw':
        x=PowerLaw_trans(n,alpha)
        xx=x[x<upper]
        accept_prob=expon(xx,tau)/pow(xx+1,alpha)
        # From the derivative of this expression,
        # we can find the maximum is at x=-alpha-1
        amix_inv=pow(-alpha,alpha)/expon(-alpha-1,tau)
        #print(np.max(accept_prob),1/amix_inv) #check if they are equal
        accept_prob=accept_prob*amix_inv
    if distr_type=='Uniform':
        xx=upper*np.random.rand(n)
        accept_prob=expon(xx,tau)
    assert(np.max(accept_prob)<=1)
    accept=np.random.rand(len(accept_prob))<accept_prob
    y=xx[accept]
    print('The accept rate from',distr_type,'to Exponential deviates is',\
          len(y),'/',len(xx),'=',len(y)/len(xx))
    return y


tau=1
upper=10
n=10000000
alpha=-2.25 # the largest accept rate
#dis_type='Lorentz'
#dis_type='PowerLaw'
dis_type='Uniform'

t1=time.time()
x=expon_reject(n,tau,upper,distr_type=dis_type)
t2=time.time()

t3=time.time()
x0=expon_trans(len(x),tau)
t4=time.time()

print('Transformation method use', t4-t3,'s.')
print('The rejection method use',t2-t1,'s.')

a,b=np.histogram(x,100)
aa=a/a[0]
bb=0.5*(b[1:]+b[:-1])
x1=np.linspace(0,upper,1001)
y1=expon(x1,tau)
plt.ion()
plt.bar(b[:-1],aa,width=0.5*(b[1]-b[0]))
plt.plot(x1,y1,c='red',linewidth=1)
plt.title('Exponential deviates by rejection method from '+dis_type)
plt.show()


