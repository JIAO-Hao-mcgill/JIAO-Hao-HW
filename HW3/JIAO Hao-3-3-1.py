import numpy as np
import camb
import time
from matplotlib import pyplot as plt


def get_spectrum(pars,lmax=1199):
    H0=pars[0]; ombh2=pars[1]; omch2=pars[2]
    As=pars[3]; ns=pars[4]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=0.05) #fixed tau
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[2:lmax+2,0] # remove the first two entries
    return tt

def chisq(data,fun,pars):
    x=data[0]
    y=data[1]
    noise=data[2]
    model=fun(pars)
    chisq=np.sum((y-model)**2/noise**2)
    return chisq

def deriv(fun,lmax,pars,dpar):
    derivs=np.zeros([lmax,len(pars)])
    for i in range(len(pars)):
        pars2=pars.copy()
        pars2[i]=pars[i]+dpar[i]
        f_right=fun(pars2,lmax=lmax)
        pars2[i]=pars[i]-dpar[i]
        f_left=fun(pars2,lmax=lmax)
        derivs[:,i]=(f_right-f_left)/(2*dpar[i])
    return derivs

par=np.asarray([65, 0.02, 0.1, 2e-9, 0.96]) # fixed tau
# H0=65, omegab*h^2=0.02, omegac*h^2=0.1, As=2e-9, ns=0.96
dpar=par*1e-5  # delta parameters
wmap=np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')
x=wmap[:,0]
y=wmap[:,1]
noise=wmap[:,2]
TT0=get_spectrum(par)
chisq0=np.sum((y-TT0)**2/noise**2)

#run Newton's with numerical derivatives
Ninv=np.diag(1/noise)
for i in range(10):
    t1=time.time()
    TT=get_spectrum(par)
    chisq1=np.sum((y-TT)**2/noise**2)
    print('The chi-square is',chisq1)
    derivs=deriv(get_spectrum,1199,par,dpar)
    resid=y-TT
    lhs=derivs.T@Ninv@derivs
    rhs=derivs.T@Ninv@resid
    lhs_inv=np.linalg.inv(lhs)
    step=lhs_inv@rhs
    par=par+step
    t2=time.time()
    print('\nThe ',i+1,'step take ',t2-t1,'sec to get the new parameters:' )
    print('H0=',par[0],',\nombh2=',par[1],', omch2=',par[2],\
          ',\nAs=',par[3],', ns=',par[4])

TT=get_spectrum(par)
chisq1=np.sum((y-TT)**2/noise**2)
print('The final chi-square is',chisq1)

print('\nThe step of the last step is ')
print('dH0=',step[0],',\ndombh2=',step[1],', domch2=',step[2],\
      ',\ndAs=',step[3],', dns=',step[4])

error=np.sum((y-TT)**2)
print('\nError is ',error)

plt.ion()
plt.plot(x,TT0,'-',c='blue')
plt.plot(x,TT,'-',c='green')
plt.scatter(x,y,s=1,c='orange')
plt.legend(['initial model','final model','data'],loc='best')
plt.show()
