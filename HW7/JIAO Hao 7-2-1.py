import numpy as np
import matplotlib.pyplot as plt
plt.ion()

def make_Ax(x0,mask):
    x=x0.copy()
    x[mask]=0
    tot=np.roll(x,1,axis=0)+np.roll(x,-1,axis=0)+np.roll(x,1,axis=1)+np.roll(x,-1,axis=1)
    x=x-tot/4.  # Ax
    x[mask]=0
    return x

#def make_rhs(x0,mask,V00=1):
#    x=x0.copy()
#    not_mask=np.logical_not(mask)
#    x[not_mask]=0
#    tot=np.roll(x,1,axis=0)+np.roll(x,-1,axis=0)+np.roll(x,1,axis=1)+np.roll(x,-1,axis=1)
#    tot[mask]=0
#    return tot/4

def cg(b,x0,mask,ninter=601):
    Ax=make_Ax(x0,mask)
    rk=b-Ax
    pk=rk.copy()
    x=x0.copy()
    rtr=np.sum(rk*rk)
    for i in range(ninter):
        Apk=make_Ax(pk,mask)
        pAp=np.sum(pk*Apk)
        ak=rtr/pAp
        x=x+ak*pk
        rk_new=rk-ak*Apk
        rtr_new=np.sum(rk_new*rk_new)
        bk=rtr_new/rtr
        pk=rk_new+bk*pk
        rk=rk_new
        rtr=rtr_new
        if i%10==0:
            plt.clf()
            plt.imshow(rk)
            plt.colorbar()
            plt.pause(0.0001)
        if i%100==0:
            print('residue of',i,'is',rtr)
    return x




n=201
mask=np.zeros([n,n],dtype='bool')
bc=np.zeros([n,n])
mask[0,:]=True
mask[-1,:]=True
mask[:,0]=True
mask[:,-1]=True
bc[n//2,n//2]=1

V=cg(bc,0*bc,mask)
V=V-V[n//2,n//2]+1





