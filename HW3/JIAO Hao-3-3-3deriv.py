import numpy as np
import camb
from matplotlib import pyplot as plt
import time


def get_spectrum(pars,lmax=1199):
    H0=pars[0]; ombh2=pars[1]; omch2=pars[2]
    tau=pars[3]; As=pars[4]; ns=pars[5]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[2:lmax+2,0]    # remove the first two terms
    return tt

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


par=np.asarray([65, 0.02, 0.1, 0.05, 2e-9, 0.96])
# H0=65, omegab*h^2=0.02, omegac*h^2=0.1, tau=0.05, As=2e-9, ns=0.96
dp1=par*1e-3
d1=deriv(get_spectrum,1199,par,dp1)
dp2=par*1e-4
d2=deriv(get_spectrum,1199,par,dp2)
dp3=par*1e-5
print(dp3)
d3=deriv(get_spectrum,1199,par,dp3)

plt.ion()

plt.subplot(231)
plt.plot(d1[:,0],'-',d2[:,0],'--',d3[:,0],':')
plt.title('deriv wrt H0')

plt.subplot(232)
plt.plot(d1[:,1],'-',d2[:,1],'--',d3[:,1],':')
plt.title('deriv wrt ombh')

plt.subplot(233)
plt.plot(d1[:,2],'-',d2[:,2],'--',d3[:,2],':')
plt.title('deriv wrt omch')

plt.subplot(234)
plt.plot(d1[:,3],'-',d2[:,3],'--',d3[:,3],':')
plt.title('deriv wrt tau')

plt.subplot(235)
plt.plot(d1[:,4],'-',d2[:,4],'--',d3[:,4],':')
plt.title('deriv wrt As')

plt.subplot(236)
plt.plot(d1[:,5],'-',d2[:,5],'--',d3[:,5],':')
plt.title('deriv wrt ns')
plt.legend(['dpar=par*10^-3','dpar=par*10^-4','dpar=par*10^-5'],loc='best')

plt.show()


