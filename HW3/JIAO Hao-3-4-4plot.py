import numpy as np
import camb
import time
from matplotlib import pyplot as plt
import corner

chain=np.loadtxt('chain-3(20000).txt')
chivec=np.loadtxt('chain-3(20000).txt')
parname=['H0','ombh','omch','tau','As','ns']

# best-fit pars of Newtom method
par=np.asarray([67.85007966125085, 0.022474746541453598, 0.11641429462470217,\
                0.04998121938569102, 2.0607555925550536e-09, 0.968576266340427])

# pars estimation with tau prior
par1=np.zeros([6,2])
par1[0,0]=69.58494933919009;    par1[0,1]=2.3056820437899064
par1[1,0]=0.022560706054178377; par1[1,1]=0.0007478180691303776
par1[2,0]=0.11384070220523158;  par1[2,1]=0.0037774632230811196
par1[3,0]=0.05360145222357419;  par1[3,1]=0.001796241854843383
par1[4,0]=2.058863059670732e-09;par1[4,1]=6.828519672104872e-11
par1[5,0]=0.971633499380878;    par1[5,1]=0.03219035346501125

plt.ion()
#figure=corner.corner(chain)
#assert(1==0)

I=6
J=6
for i in range(0,I):
    for j in range(0,i+1):
        plt.subplot(I,J+1,(J+1)*i+j+1)
        if i==j:
            plt.hist(chain[:,j],bins=20,color='white',edgecolor='steelblue')
        else:
            #plt.scatter(chain[:,j],chain[:,i],s=0.02,alpha=0.5)
            plt.plot(chain[:,j],chain[:,i],',')
            plt.plot(par[j],par[i],'+',c='red')
            plt.axvline(x=par1[j,0],color='green',linestyle='-')
            plt.axvline(x=par1[j,0]-par1[j,1],color='green',linestyle='-')
            plt.axvline(x=par1[j,0]-par1[j,1],color='green',linestyle='-')
            plt.axhline(y=par1[i,0],color='green',linestyle='-')
            plt.axhline(y=par1[i,0]-par1[i,1],color='green',linestyle=':')
            plt.axhline(y=par1[i,0]+par1[i,1],color='green',linestyle=':')
        if j==0:
            plt.ylabel(parname[i])
        else:
            plt.yticks([])
        if i==5:
            plt.xlabel(parname[j])
        else:
            plt.xticks([])
    plt.subplot(I,J+1,(J+2)*i+2)
    #plt.scatter(chain[:,i],s=0.02,c='lightblue')
    plt.plot(chain[:,i],',',c='lightblue')
    plt.title('MCMC result')
    plt.yticks([]); plt.xticks([])





plt.show()




