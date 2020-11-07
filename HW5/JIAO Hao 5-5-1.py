import numpy as np
from matplotlib import pyplot as plt
import h5py
import json
import sample_read_ligo as rl
plt.ion()

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

    freq=np.fft.rfftfreq(len(AH_PW),dtH)
    AH_PW_ft=np.abs(np.fft.rfft(AH_PW))
    AL_PW_ft=np.abs(np.fft.rfft(AL_PW))

    n=len(AH_PW_ft)
    AH_cumulative=np.zeros(n)
    AL_cumulative=np.zeros(n)
    for j in range(n):
        AH_cumulative[j]=np.sum(np.abs(AH_PW_ft[:j]))/dtH
        AL_cumulative[j]=np.sum(np.abs(AL_PW_ft[:j]))/dtL

    plt.subplot(4,3,3*i+1)
    plt.plot(AH_PW,',')
    plt.plot(AL_PW,',')
    if i==0: plt.title('template_PW')
    
    plt.subplot(4,3,3*i+2)
    plt.loglog(freq,AH_PW_ft,',')
    plt.loglog(freq,AL_PW_ft,',')
    if i==0: plt.title('FT template_PW')
    
    plt.subplot(4,3,3*i+3)
    plt.loglog(freq,AH_cumulative,',')
    plt.loglog(freq,AL_cumulative,',')
    if i==0: plt.title('cumulative FT template_PW')
    
    AH_cumulative=np.abs(AH_cumulative-0.5*AH_cumulative[-1])
    index=np.argmin(AH_cumulative)
    fH=freq[index]
    AL_cumulative=np.abs(AL_cumulative-0.5*AL_cumulative[-1])
    index=np.argmin(AL_cumulative)
    fL=freq[index]
    print('\nThe middle frequency for '+eventname[i],':')
    print('H1:',fH,'Hz , L1:',fL,'Hz')
    












