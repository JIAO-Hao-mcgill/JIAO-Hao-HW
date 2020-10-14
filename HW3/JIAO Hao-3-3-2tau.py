import numpy as np
import camb
import time
from matplotlib import pyplot as plt


def get_spectrum(tau,lmax=1199):
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=67.8753462432025,ombh2=0.0224797374072905,\
                       omch2=0.11636774179760019,mnu=0.06,omk=0,tau=tau) 
    pars.InitPower.set_params(As=2.060520942240872e-09,ns=0.9687158967380484,r=0)
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

def deriv(fun,lmax,par,dpar):
    derivs=np.zeros(lmax)
    par2=par
    par2=par2+dpar
    f_right=fun(par2,lmax=lmax)
    par2=par-dpar
    f_left=fun(par2,lmax=lmax)
    derivs=(f_right-f_left)/(2*dpar)
    return derivs


tau=0.05 # fixed tau
wmap=np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')
x=wmap[:,0]
y=wmap[:,1]
noise=wmap[:,2]
TT0=get_spectrum(tau)
chisq0=np.sum((y-TT0)**2/noise**2)

#run Newton's with numerical derivatives
Ninv=np.diag(1/noise)
dtau=tau*1e-5  # delta parameters
for i in range(10):
    t1=time.time()
    TT=get_spectrum(tau)
    chisq1=np.sum((y-TT)**2/noise**2)
    print('The chi-square is',chisq1)
    derivs=deriv(get_spectrum,1199,tau,dtau)
    resid=y-TT
    lhs=derivs.T@Ninv@derivs
    rhs=derivs.T@Ninv@resid
    step=rhs/lhs
    tau=tau+step
    t2=time.time()
    print('\nThe',i+1,'step take ',t2-t1,'sec to get the new parameters:' )
    print('tau=',tau)

TT=get_spectrum(tau)
chisq1=np.sum((y-TT)**2/noise**2)
print('The final chi-square is',chisq1)

print('\nThe step of the last step is ')
print('dtau',step)

error=np.sum((y-TT)**2)
print('\nError is ',error)

plt.ion()
plt.plot(x,TT0,'-',c='blue')
plt.plot(x,TT,'-',c='green')
plt.scatter(x,y,s=1,c='orange')
plt.legend(['initial model','final model','data'],loc='best')
plt.show()
