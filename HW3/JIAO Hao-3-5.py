import numpy as np
import camb
import time
from matplotlib import pyplot as plt
import corner

chain=np.loadtxt('chain-1(20000).txt')

n=5319
ch=np.zeros([n,6])

j=0
ch[j,:]=chain[0,:]

# remove repeating steps, get sample
for i in range(1,len(chain[:,0])):
    if chain[i,0]!=chain[i-1,0]:
        j=j+1
        ch[j,:]=chain[i,:]
print(j)

def gauss(x,a,b):  # a=expectation , b=sigma
    y=np.exp(-(x-a)*(x-a)/(2*b*b))/(np.sqrt(2*np.pi)*b)
    return y

tau0=0.0544
dtau=0.0073
P_tau=np.zeros(n)
p_tau=gauss(ch[:,3],tau0,dtau)

par=np.zeros(6)
rms=np.zeros(6)
for i in range(6):
    par[i]=np.sum(ch[:,i]*p_tau)/np.sum(p_tau)
    rms[i]=np.sqrt(np.sum((ch[:,i]*p_tau-par[i])**2))/np.sum(p_tau)

print('H0=',par[0], ', \tsigma_H0=',rms[0])
print('ombh=',par[1], ', \tsigma_ombh=',rms[1])
print('omch=',par[2], ', \tsigma_omch=',rms[2])
print('tau=',par[3], ', \tsigma_tau=',rms[3])
print('As=',par[4], ', \tsigma_As=',rms[4])
print('ns=',par[5], ', \tsigma_ns=',rms[5])
