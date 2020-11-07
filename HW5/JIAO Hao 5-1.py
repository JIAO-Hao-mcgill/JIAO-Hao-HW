import numpy as np
from matplotlib import pyplot as plt
import h5py
import json
import sample_read_ligo as rl

plt.ion()

dic='LOSC_Event_tutorial/LOSC_Event_tutorial/'

# Livingston noise model
fnameL='L-L1_LOSC_4_V2-1126259446-32.hdf5'
strainL,dtL,utcL=rl.read_file(dic+fnameL)
# Hanford noise model
fnameH='H-H1_LOSC_4_V2-1126259446-32.hdf5'
strainH,dtH,utcH=rl.read_file(dic+fnameH)

template_name='GW150914_4_template.hdf5'
th,tl=rl.read_template(dic+template_name)




def window(a,win_type='tukey'):
    n=len(a)
    x=np.linspace(-1,1,n)
    if win_type=='cos':
        win=0.5+0.5*np.cos(np.pi*x)
    if win_type=='welch':
        win=1-x*x
    if win_type=='tukey':
        win=np.ones(n)
        for i in range(n):
            if np.abs(x[i])>0.5:
                win[i]=0.5+0.5*np.cos(2*np.pi*(np.abs(x[i])-0.5))
    return win*a

def smooth(a,npix,kind='boxcar'):
    aft=np.fft.rfft(a)
    if kind=='boxcar':
        vec=np.zeros(len(a))
        vec[:npix]=1
        vec[-npix+1:]=1
        vec=vec/np.sum(vec)
    vecft=np.fft.rfft(vec)
    return np.fft.irfft(vecft*aft,len(a))

def noise_model(strain,win_type='tukey',npix=5):
    strain_win=window(strain,win_type=win_type)
    sft_win=np.fft.rfft(strain_win)
    N=np.abs(sft_win)**2
    if npix!=0: # if npix==0, don't smooth
        N=smooth(N,npix)
    return N,sft_win

def whiten(sft,N): # sft & N are in Fourier space
    white_ft=sft/np.sqrt(N)
    return np.fft.irfft(white_ft)

n=len(strainH)
dt=1*dtH
freq=np.fft.rfftfreq(n,dt)   #frequency

# plot PS with three window functions
#N1,sftH=noise_model(strainH)
#N2,sftH=noise_model(strainH,win_type='cos')
#N3,sftH=noise_model(strainH,win_type='welch')
#plt.loglog(freq,N2,freq,N3,freq,N1,linewidth=1)
#plt.legend(['cos','welch','tukey'],loc='best')
#plt.xlim(20,2000) # if want, we can limit the frequancy
#plt.xlabel('frequency Hz')
#plt.ylabel('noise spectrum')

# noise model for Livingston and Hanford 
NH,sftH=noise_model(strainH)
NL,sftL=noise_model(strainL)
#plt.loglog(freq,NH,freq,NL,linewidth=1)
#plt.legend(['H1','L1'],loc='best')
#plt.xlim(20,2000)  # I didn't use this code when ploting the fig
#plt.xlabel('frequency Hz')
#plt.ylabel('noise spectrum')

# pre-whitened data
Aft=np.fft.rfft(window(th))
dataH_PW=whiten(sftH,NH)
dataH_ft_PW=np.fft.rfft(dataH_PW)
plt.subplot(311)
plt.plot(dataH_PW,linewidth=0.5)
plt.title('Whitened data')
plt.subplot(312)
plt.plot(freq,dataH_ft_PW,linewidth=0.5)
plt.title('Whitened FT of data')
plt.subplot(313)
plt.plot(freq,dataH_ft_PW,linewidth=0.5)
plt.title('Zoom in Whitened FT of data')
plt.xlim(0,20)





