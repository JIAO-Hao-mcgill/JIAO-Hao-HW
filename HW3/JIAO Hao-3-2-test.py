import numpy as np
import camb
from matplotlib import pyplot as plt
import time


def get_spectrum(pars,lmax=1200):
    H0=pars[0]; ombh2=pars[1]; omch2=pars[2]
    tau=pars[3]; As=pars[4]; ns=pars[5]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]    #you could return the full power spectrum here if you wanted to do say EE
    return tt


plt.ion()

pars1=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])
pars2=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])
pars3=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])
# H0=65, omegab*h^2=0.02, omegac*h^2=0.1, tau=0.05, As=2e-9, ns=0.96
wmap=np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')

plt.clf();
#plt.errorbar(wmap[:,0],wmap[:,1],wmap[:,2],fmt='*')
plt.plot(wmap[:,0],wmap[:,1],'.')

# TT spectrum
cmb1=get_spectrum(pars1,lmax=1200); plt.plot(cmb1)
print(len(cmb1))
#cmb2=get_spectrum(pars2); plt.plot(cmb2)
#cmb3=get_spectrum(pars3); plt.plot(cmb3)
#plt.legend(['1','2','3'],loc='best')
plt.show()
