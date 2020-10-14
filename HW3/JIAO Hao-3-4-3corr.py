import numpy as np
import camb
import time
from matplotlib import pyplot as plt


def get_spectrum(pars,lmax=1199):
    H0=pars[0]; ombh2=pars[1]; omch2=pars[2]
    tau=pars[3]; As=pars[4]; ns=pars[5]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau) #fixed tau
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[2:lmax+2,0] # remove the first two entries
    return tt

def chisq(data,pars):
    x=data[:,0]
    y=data[:,1]
    noise=data[:,2]
    model=get_spectrum(pars)
    chisquare=np.sum((y-model)**2/noise**2)
    return chisquare

def run_mcmc_corr(pars,data,chifun,mycov,nstep=20000):
    n=0 # count the effective steps/samples
    npar=len(pars)
    chain=np.zeros([nstep,npar])
    chivec=np.zeros(nstep)
    chi_cur=chifun(data,pars)
    L=np.linalg.cholesky(mycov)
    for i in range(nstep):
        pars_trial=pars+L@np.random.randn(npar)
        if pars_trial[3]<0:
            pars_trial=pars
            n=n-1
        chi_trial=chifun(data,pars_trial)
        if np.random.rand(1)<np.exp(-0.5*(chi_trial-chi_cur)):
            pars=pars_trial
            chi_cur=chi_trial
            n=n+1
        chain[i,:]=pars
        chivec[i]=chi_cur
        print(i,n)
    return chain,chivec,n


par=np.asarray([65, 0.02, 0.1, 0.05, 2e-9, 0.96]) # Initial condition
# H0=65, omegab*h^2=0.02, omegac*h^2=0.1, tau=0.05, As=2e-9, ns=0.96
wmap=np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')
x=wmap[:,0]
y=wmap[:,1]
noise=wmap[:,2]
TT0=get_spectrum(par)
chisq0=np.sum((y-TT0)**2/noise**2)

chain1=np.loadtxt('chain-1(20000).txt')
chivec1=np.loadtxt('chivec-1(20000).txt')

# change the step of parameters
par_step=np.std(chain1[chain1.shape[0]//10:,:],axis=0)
print(par_step)

# correlated pars
delt=chain1.copy()
for i in range(delt.shape[1]):
    delt[:,i]=delt[:,i]-delt[:,i].mean()
mycov=delt.T@delt/chain1.shape[0]

chain,chivec,n=run_mcmc_corr(par,wmap,chisq,mycov)
print('The number of samples is',n)

plt.ion()
#plt.plot(x,TT0,'-',c='blue')
#plt.plot(x,TT,'-',c='green')
#plt.scatter(x,y,s=1,c='orange')
#plt.legend(['initial model','final model','data'],loc='best')
plt.plot(chain[:,0])
plt.show()

pars=np.zeros(6)
r=np.zeros(6)
for i in range(6):
    pars[i]=np.mean(chain[:,i])
    r[i]=np.std(chain[:,i])

