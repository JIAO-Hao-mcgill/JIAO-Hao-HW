import numpy as np
from matplotlib import pyplot as plt
import h5py
import json
import sample_read_ligo as rl
plt.ion()

# the only difference between 5-2 is the last part (after line)

dic='LOSC_Event_tutorial/LOSC_Event_tutorial/'

eventname=['GW150914','LVT151012','GW151226','GW170104']

fnjson = "BBH_events_v3.json"
try:
    events = json.load(open(dic+fnjson,"r"))
except IOError:
    print("Cannot find resource file "+fnjson); quit()


# -------- function from 5-1 begin --------

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
    N=smooth(N,npix)
    return N,sft_win

def whiten(sft,N): # sft & N are in Fourier space
    white_ft=sft/np.sqrt(N)
    return np.fft.irfft(white_ft)

# -------- function from 5-1 end --------


def xcorr(a,b):
    return np.fft.irfft(np.fft.rfft(a)*np.conj(np.fft.rfft(b)))


for i in range(4):
    event = events[eventname[i]]
    # open the corresponding files
    strainH,dtH,utcH=rl.read_file(dic+event['fn_H1'])
    strainL,dtL,utcL=rl.read_file(dic+event['fn_L1'])
    th,tl=rl.read_template(dic+event['fn_template'])
    # noise mode
    NH,sftH=noise_model(strainH)
    NL,sftL=noise_model(strainL)
    # whiten data
    dataH_PW=whiten(sftH,NH)
    dataL_PW=whiten(sftL,NL)
    # whiten template
    AH_ft=np.fft.rfft(window(th))
    AH_PW=whiten(AH_ft,NH)
    AL_ft=np.fft.rfft(window(tl))
    AL_PW=whiten(AL_ft,NL)

    # matched filter
    mfH=xcorr(dataH_PW,AH_PW)
    mfL=xcorr(dataL_PW,AL_PW)
    n=len(mfH)
    posH=np.argmax(np.abs(mfH))
    posL=np.argmax(np.abs(mfL))
    mfH=np.abs(np.roll(mfH,n//2-posH))
    mfL=np.abs(np.roll(mfL,n//2-posL))
    mf=(mfH+mfL)/2

#------------------------------------------------------------------
    
    # Noise
    sigma_H=np.sqrt(np.average(np.fft.irfft(NH)))
    sigma_L=np.sqrt(np.average(np.fft.irfft(NL)))
    print('\nNoise for ',eventname[i],':')
    print('Real space:\tH1:',sigma_H,', L1:',sigma_L)
    print('Fourier space:\tH1:',np.sqrt(np.average(NH)),', L1:',np.sqrt(np.average(NL)))
    
    # Singal Noise Ratio
    # I have check that this result is almost equal to the result use std withuot abs
    SNR_H=np.max(mfH)/np.sqrt(np.sum(mfH*mfH/n))
    SNR_L=np.max(mfL)/np.sqrt(np.sum(mfL*mfL/n))
    SNR=np.max(mf)/np.sqrt(np.sum(mf*mf/n))
#    print('\nFor ',eventname[i],':')
#    print('SNR:\tH1:', SNR_H,', L1:',SNR_L,', H1+L1:',SNR)
    


















