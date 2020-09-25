import numpy as np
import matplotlib.pyplot as plt

N=1001

def function(x):
    y=[]
    for i in range(len(x)):
        y.append(np.log2(0.5+0.5*(x[i]+1)/2))
    return y
# change the domain of log2

def cheb_mat(nx,ord):
    x=np.linspace(-1,1,nx)
    mat=np.zeros([nx,ord+1])
    mat[:,0]=1
    mat[:,1]=x
    for i in range(2,ord+1):
        mat[:,i]=2*x*mat[:,i-1]-mat[:,i-2]
    return mat,x


x0=np.linspace(0.5,1,N)
x1=np.linspace(-1,1,N)

truth=np.log2(x0)

plt.subplot(211)
for order in range(6,8): #compare the chev. of order 6 and order 7
    mat,x=cheb_mat(N,order)
    y=np.log2(x0)
    lhs=mat.T@mat
    rhs=mat.T@y
    coeff_cheb=np.linalg.inv(lhs)@rhs
    cheb_my=mat@coeff_cheb
    plt.plot(x0,cheb_my-truth,'-')
plt.legend(['chebyshev residual of order 6','chebyshev residual of order 7'],loc='best')

rms_cheb=np.std(cheb_my-truth)
error_cheb=np.max(np.abs(cheb_my-truth))
print('To an accuracy in the region better than 10^{-6},\n\
we have to use Chebyshev polynomial fit with ',order,'terms to fit np.log2.')
print('For Chebyshev fit, the mean square error is',rms_cheb,\
      ' and the maximum errors is ',error_cheb)

x=np.linspace(0.5,1,order+1)
y=np.log2(x)
coeff_poly=np.polynomial.legendre.legfit(x,y,order)
poly_my=np.polynomial.legendre.legval(x0,coeff_poly)

rms_poly=np.std(poly_my-truth)
error_poly=np.max(np.abs(poly_my-truth))
print('For same order Ploynomial fit, the mean square error is',rms_poly,\
      ' and the maximum errors is ',error_poly)

plt.subplot(212)
plt.plot(x0,cheb_my-truth,'-',x0,poly_my-truth,'-')
plt.legend(['chebyshev residual','polynomial residual'],loc='best')

plt.show()

