import numpy as np
import camb
from matplotlib import pyplot as plt
import time


def get_spectrum(pars,lmax=2000):
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

plt.ion()

pars=np.asarray([65, 0.02, 0.1, 0.05, 2e-9, 0.96])
# H0=65, omegab*h^2=0.02, omegac*h^2=0.1, tau=0.05, As=2e-9, ns=0.96
wmap=np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')

ndata=len(wmap[:,0])

TT=get_spectrum(pars,lmax=ndata) # TT spectrum

chisq=np.sum((wmap[:,1]-TT[:ndata])**2/wmap[:,2]**2)
print('The chi-square is',chisq)


