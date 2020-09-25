import numpy as np
import matplotlib.pyplot as plt

order=8
N=1001

def function(x):
    y=[]
    for i in range(len(x)):
        y.append(np.log2(0.5+0.5*(x[i]+1)/2))
    return y
# change the domain of log2

def cheb_fit(fun,ord):
    x=np.linspace(-1,1,ord+1)
    y=fun(x)
    mat=np.zeros([ord+1,ord+1])
    mat[:,0]=1
    mat[:,1]=x
    for i in range(2,ord+1):
        mat[:,i]=2*x*mat[:,i-1]-mat[:,i-2]
    coeffs=np.linalg.inv(mat)@y
    return coeffs

def poly_fit(fun,x1,x2,deg):
    x=np.linspace(x1,x2,deg+1)
    y=fun(x)
    mat=np.zeros([deg+1,deg+1])
    for i in range(deg+1):
        mat[:,i]=x**i
    coeffs=np.linalg.inv(mat)@y
    return coeffs

x0=np.linspace(0.5,1,N)
x1=np.linspace(-1,1,N)

truth=np.log2(x0)

plt.subplot(221)
for order in range(7,9):
    coeff_cheb=cheb_fit(function,order)
    mat1=np.zeros([N,order+1])
    mat1[:,0]=1
    mat1[:,1]=x1
    for i in range(2,order+1):
        mat1[:,i]=2*x1*mat1[:,i-1]-mat1[:,i-2]
    cheb_my=mat1@coeff_cheb
    plt.plot(x0,cheb_my-truth,'-')
plt.legend(['chebyshev of order 7','chebyshev of order 8'],loc='best')

plt.subplot(222)
for deg in range(7,9):
    coeff_poly=poly_fit(np.log2,0.5,1,deg)
    mat2=np.zeros([N,deg+1])
    for i in range(deg+1):
        mat2[:,i]=x0**i
    poly_my=mat2@coeff_poly
    plt.plot(x0,poly_my-truth,'-')
plt.legend(['polynomial of order 7','polynomial of order 8'],loc='best')

plt.subplot(212)
plt.plot(x0,cheb_my-truth,'-',x0,poly_my-truth,'-')
plt.legend(['chebyshev','polynomial'],loc='best')

plt.show()

